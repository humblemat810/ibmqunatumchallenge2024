
#%%
import os
os.environ['QXToken']="d1985953167baa593d331339a5c0cfa20858d94bfcc06c647cdc523d97896df43bd0e8ea9ef5451a060491a22446b1ac56693bbac30453a31a51b7def6aedcdb"


# Make sure there is no space between the equal sign
# and the beginning of your token

# %%
# qc-grader should be 0.18.11 (or higher)
import qc_grader
import qiskit_serverless
qiskit_serverless.core.client.IBMServerlessClient
qc_grader.__version__

# %% [markdown]
# Now, let's run our imports and setup the grader

# %%
# Import all in one cell

import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import warnings

from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import TwoLocal, EfficientSU2
from qiskit_ibm_runtime import QiskitRuntimeService, Estimator, Session, Options
from qiskit_serverless import QiskitFunction, save_result, get_arguments, save_result, distribute_task, distribute_qiskit_function, IBMServerlessClient, QiskitFunction
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_transpiler_service.transpiler_service import TranspilerService
from qiskit_aer import AerSimulator

# %%
# Import for grader
from qc_grader.challenges.iqc_2024 import grade_lab3_qs_ex1, grade_lab3_qs_ex2


service = QiskitRuntimeService(channel="ibm_quantum")
 
# Specify a system to use for transpilation
real_backend = service.backend("ibm_brisbane")


# %%
# Qiskit Pattern Step 1: Map quantum circuits and operators (Define Ansatz and operators)
num_qubits = 3 #Add your code here
rotation_blocks = ['ry','rz']#Add your code here
entanglement_blocks = 'cz' #Add your code here
entanglement = 'full'#Add your code here

# Define Ansatz
two = TwoLocal(num_qubits, rotation_blocks, entanglement_blocks, entanglement, reps=1, insert_barriers=True)
qc = QuantumCircuit(3)
qc2 = qc.compose(two)

ansatz = two

# Define parameters
num_params = ansatz.num_parameters

# Qiskit Pattern Step 2: Optimize the circuit for quantum execution
optimization_level = 2
pm = generate_preset_pass_manager(backend=real_backend, optimization_level=optimization_level)
isa_circuit = pm.run(ansatz)# Add your code here

# Define Hamiltonian for VQE
pauli_op = SparsePauliOp(['ZII', 'IZI', 'IIZ'])
hamiltonian_isa = pauli_op.apply_layout(layout=isa_circuit.layout)


# %%
# Setup Qiskit Serverless Client and Qiskit Runtime client
client = IBMServerlessClient("d1985953167baa593d331339a5c0cfa20858d94bfcc06c647cdc523d97896df43bd0e8ea9ef5451a060491a22446b1ac56693bbac30453a31a51b7def6aedcdb") # Add in your IBM Quantum Token to QiskitServerless Client


# For the challenge, we will be using QiskitRuntime Local testing mode. Change to True only if you wish to use real backend.
USE_RUNTIME_SERVICE = False

if USE_RUNTIME_SERVICE:
    service = QiskitRuntimeService(
        channel='ibm_quantum', 
        verify=False
    )
else:
    service = None


# %%
# Define the Qiskit Function
working_dir = os.path.join(os.getcwd())
if USE_RUNTIME_SERVICE:
    function = QiskitFunction(title= "run-vqe", entrypoint="vqe.py", working_dir=working_dir)
else:
    function = QiskitFunction(title= "run-vqe-qiskit-aer" , entrypoint="vqe.py", working_dir=working_dir,  dependencies=["qiskit_aer"])


# %%
import os
print(os.getcwd())
print(os.path.exists(os.path.join(os.getcwd(), "vqe.py")))
function.provider = 'debuguserprof'
# %%
import qiskit_serverless
job = qiskit_serverless.core.job.Job('runvqe', 'test')
# Upload the Qiskit Function using IBMServerlessClient
client.upload(function)

#%%
num_qubits = [41, 51, 61]
circuits = [EfficientSU2(nq, su2_gates=["rz","ry"], entanglement="circular", reps=1).decompose() for nq in num_qubits]

#%%
# Authenticate to the remote cluster and submit the pattern for remote execution if not done in previous exercise
serverless = IBMServerlessClient("d1985953167baa593d331339a5c0cfa20858d94bfcc06c647cdc523d97896df43bd0e8ea9ef5451a060491a22446b1ac56693bbac30453a31a51b7def6aedcdb")

#%%
optimization_levels = [1,2,3]# Add your code here
pass_managers = [{'pass_manager': generate_preset_pass_manager(optimization_level=level, backend=backend), 'optimization_level': level} for level in optimization_levels]

transpiler_services = [ 
        {'service': TranspilerService( optimization_level=3, ai=False,backend_name="ibm_brisbane" ), 'ai': False, 'optimization_level': 3},
        {'service': TranspilerService(  optimization_level=3, ai=True, backend_name="ibm_brisbane" ), 'ai': True, 'optimization_level': 3}
    ]

configs = pass_managers + transpiler_services

#%%
transpile_parallel_function = QiskitFunction(
    title="transpile_parallel",
    entrypoint="transpile_parallel.py",
    working_dir="./src",
    dependencies=["qiskit-transpiler-service"]
)
#%%
serverless.upload(transpile_parallel_function) # Add your code here )
#%%
# Get list of functions
serverless.list()
#%%
serverless.get("transpile_parallel")

## Fetch the specific function titled "transpile_parallel"
transpile_parallel_serverless: qiskit_serverless.core.function.QiskitFunction = serverless.get("transpile_parallel")# Add your code here

#%%
job2 = transpile_parallel_serverless.run(circuits= circuits, configs = configs)

#%%
arguments = dict(circuits= circuits)
job2 = client.run("transpile_parallel", arguments = arguments)
# %%
service = QiskitRuntimeService(channel="ibm_quantum")
backend = service.backend("ibm_brisbane")

# %%
grade_lab3_qs_ex2(optimization_levels, transpiler_services, transpile_parallel_function, transpile_parallel_serverless, job2)

# %%
