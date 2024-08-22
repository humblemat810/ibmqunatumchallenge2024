
# %% [markdown]
# # Lab 2: The Transpiler
# 
# 

# %% [markdown]
# # Table of contents
# 
# * [Prologue - What is the transpiler?](#prologue)   
# * [Transpile with Preset Pass Managers](#preset_passmanager)    
# * [Optimization levels](#optimization_level)
#     * [Optimization level = 0](#opt_lv_0)
#     * [Optimization level = 1](#opt_lv_1)
#     * [Optimization level = 2](#opt_lv_2)
#     * [Optimization level = 3](#opt_lv_3)    
# * [Transpiler Stage Details with Options](#transpiler_options)
#     * [Init stage](#init)
#     * [Layout Stage](#layout)
#     * [Routing Stage](#routing)
#     * [Translation Stage](#translation)
#     * [Optimization Stage](#optimization)
#     * [Scheduling Stage](#scheduling)    
# * [Build your own Pass Managers with Staged Pass Manager](#staged_pm)    
#     * [Build `Dynamic Decoupling` Pass](#dd)
# * [(Bonus) Ecosystem and Qiskit Transpiler Plugin](#plugin)  

# %% [markdown]
# <a id='toc2_'></a>
# <a name='toc2_'></a>
# 
# # Setup
# 
# To run this lab properly, we need several packages. If you haven't yet installed Qiskit and relevant packages, please run the cells below after uncommenting and restarting the kernel.

# %%
### Install Qiskit and relevant packages, if needed

!pip install qiskit[visualization]==1.0.2
!pip install qiskit_ibm_runtime
!pip install qiskit_aer
!pip install qiskit-transpiler-service
!pip install graphviz
!pip install git+https://github.com/qiskit-community/Quantum-Challenge-Grader.git

# %%
### Save API Token, if needed

%set_env QXToken=d1985953167baa593d331339a5c0cfa20858d94bfcc06c647cdc523d97896df43bd0e8ea9ef5451a060491a22446b1ac56693bbac30453a31a51b7def6aedcdb

# Make sure there is no space between the equal sign
# and the beginning of your token

# %% [markdown]
# Now, let's run our imports and setup the grader

# %%
# Imports

from qiskit.circuit.random import random_circuit
from qiskit.circuit.library import XGate, YGate
from qiskit_ibm_runtime.fake_provider import FakeTorino, FakeOsaka
from qiskit.transpiler import InstructionProperties, PassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler.preset_passmanagers.plugin import list_stage_plugins
from qiskit.transpiler.timing_constraints import TimingConstraints
from qiskit.transpiler.passes.scheduling import ASAPScheduleAnalysis,PadDynamicalDecoupling
from qiskit.visualization.timeline import draw, IQXStandard
from qiskit.transpiler import StagedPassManager
from qiskit.visualization import plot_circuit_layout
import matplotlib.pyplot as plt
import numpy as np

# %%
# Setup the grader
from qc_grader.challenges.iqc_2024 import (
    grade_lab2_ex1,
    grade_lab2_ex2,
    grade_lab2_ex3,
    grade_lab2_ex4,
    grade_lab2_ex5
)

# %% [markdown]
# This lab must use Qiskit v1.0.2. Before we start, let's check your environment.

# %%
from util import version_check

version_check()

# %% [markdown]
# # Prologue - What is the transpiler?<a name="prologue"></a>

# %% [markdown]
# Let's start with a hypothetical question:
# 
# When someone hands you their car keys and says "will you fill my car up with fuel?" - how do you know what to do?
# 
# Sure, you have your driver's license, but what type of car do they have? Where is their gear shifter? How do you turn on the blinker to turn the corner, or open the fuel tank once your arrive? What if it's an electric car that doesn't even *have* a fuel tank??
# 
# Luckily, the human brain is smart. It is able to take a set of instructions and adapt them to the vehicle being used.
# 
# That, in essence, is the transpiler.

# %% [markdown]
# Transpilation is the process of taking a given input circuit and rewriting it to an equivalent circuit for a specific quantum device, and/or to optimize the circuit for execution on a real quantum system.
# 
# This is necessary because not all quantum devices work the same way. The instructions you send to one device might not be compatible with a different quantum backend. Transpilation operates in terms of a device's basis gate set, the topology of the quantum chips, timing constraints, and more which we will explore in this lab.
# 
# The goal of a transpiler is to get the best performance from noisy quantum hardware. Most circuits must undergo a series of transformations that make them compatible with a given target device, and optimize them to reduce the effects of noise on the resulting outcomes.
# 
# **For example:** the process of transpilation can take a simple looking circuit that contains your instructions:

# %% [markdown]
# <img src="data:image/webp;base64,UklGRvYHAABXRUJQVlA4IOoHAABQMgCdASoxAa4APpFInkulpCKhodK6ELASCWNu4W0RDNpahb6r+q+g7WH8ztviLuuzGJ6hfMA5w3mA/bL1Y/xJ94/oAf0D/VdZd/ZP+B7C37R+nP7NH9z/6+VEeJ/6l2z/3HlH5Yphb7efhfyq5Adrb/I/lPwO/Lf5n/i/t051O2N45mgB/Fv6f/0f7X6/Wd76p9gz+Vf2H/m9gMYmsNTDs5CAksQMhPx2chASWIGQn47OGKFtWXUwXR3mmCiOvWKeQgYY7u0Ih2K2VY3HhePwW1m7Phns4fIwKTuWGpilRDgxwJaQxCM+ohwbP7fG9Ci0VrLt8whg0NZXpJdnVg6Ve8ZQMFJk4dfTKEF4LDboskxfu/u8dG/HZwqhs1bjv8ydnl+trkriBkJ+OzkH/CzFs5TszeIZGWIFxyz/yQewkns8D4GoVoE5mFtuW4k19C23mzwsvmRQmI2sK6hycaeqxAyPK/QF4b7zO/MvJluTPg/NZO5x8JNYr0gHXymZBoX4TvLmYXIGQgJLEEPMOSksQMhPx2chASWIFsAA/v5UgAAAi0eBE+4uxVHgKzVglvAwe7FPj5a5Z9t08KQniNYOvSVSzpN2tpIQxwJIxXUtaQ9w9Bdz1yVaWTBO0oa2RayJ3PGq9U/qriD+QDJ0uKrBvO6lh90NxQvvKsgKXpGjjLA6rw6fQ0ZFyZh4qv9JMwWw6K21HBLH0wPxn7YgenT/pfMpHqW+Nx5HduZLr73vbz84VLlXA6P9LnvPvWl6gtQHbC2BZdEO/EHWCpODaVPAp1IXZ/Hvl7oYo4lr1ee5KnjT4n9ROr0pbsMaZ8e+ocEACJxTv6GF2++cRIw7+iXCrEWIj//oIhH1ZLxSBkE4ua3z7AXkKsNoSf/8Ipdo3K17vc00e4Y7t/FZJYMkmV+hPe4Cx/i2oA+Vl61Mpyy5BpBGDry/4myebmdj8XwhxdrBeDXDZDb2eqkM4OSZcYt+NGH9vQzaBKlLiG4D/M8sC0EeYY96+B3busgZciqOJQFY6jHiqZAGITtUz5ayF/eZp/g9q/nO9SDcpj37MO7QTNZ0Jd28UWlVzz0wlgEYsoXj+gqWVUgaj7A/3nJXyvpAi8v7AZUZGHl48yCDAglbQv/DPYqHrzbLoPpYJM5ZsTWpISbkNaIehfNMJ3PM40FRm22QPv/1psRd8Oz+4LLQzt+jFdmT4d175rDdgrE/kj7cRxSdP/hP2kOjI9LqW1+9agakleZ69K/YVCzlcUW48EDv+wbnqbhurBW4Yphh6n9bem9bWXvJYp3OiDjRuDOzxOKzczKO01abVrLDg6MxWh88iSdVPMNVts9VXJTaKh4q8xn3FcIxMzv/QjgyYxEXoj7uGGf62mwOalh7OBKQNejETjfhH4kTTS/s/X+vdY8QOnEKyflm7hv3m3KdSo9nXUXLCcj0s4aWxCjoMwOipZF0szoFM8nOULdhyMxOebR06YlleGhaoCoF7xuubm5qssRZlInf10TaORcIo6m6t1e5+SmNWIJstX4ae3xP/npg6UEgOuUehVjL1dOg24z1EuTnGxXh3EfZ4ZwCfaW1ptYt0C3W96R1dNq0ePpYOJRfhrdEW1tPbUD/O4XXsAxKcRddwtEhoJyja8F2h8cNE0fycbxLu16oRGzP0Zj1c6KZbWFz12jXiblwdHKtOnuH1hpkulWinpji9nYfGW7OJId7m7rTUd1Uq+eoqQhKr6v4qkljB7kHDY9pPSdzuDDk5k3aXnhRrRsd4RKkLgUZeU1qyj+cwKp0gNeTXNkpQoRpKIYaeaj7JV1NiF0Be9DCgiyHIAAGi2WnMD0TNRm3rl9wkA0Z4g3BGyoG/AmoZW6egrl9TiVVwDphB0tOKLF12ckOxDMXxIIeP28uWd3omvwPDKCoE0fwZcb/kJ72Arv/Ftp7/+LZNmTePLodnb4iTAkeZuk/5wMzxqvl4E8qXH8cU62w2XgRxG5OZpyDPlWaCnB+w8XfN5hAW6yRctQqLX2Kl0AL6wvPM/I/avYfXDyy5SYgnwmUct2uyVXdXtsL5zeLFzg+xQjWXE8nGwKdX5W4BtRpwvcO+XV+4NsIxqG63jJwy0n+GewxzR37PinMYZsvNSiI2810M0kpTAufwKuNsYtg5CTyH9aSu0CMy5Az/Cg864OPH2N7v3LxXvuasTEHmlNKx92crt1hGyCe1GCY0ISTPXw9CDSQUFJlGmzxXPugbrpyLJ4A0pSKSAGFAgSClX803DHMvw3JZUiVfrlDe0Vb++NRZ5Effj8WiC8PCdYdtX/uMTe1eMBkNUkjuQSMV5hH913G/1pTUT/j+plWmX043BIPb6EkP6n4z3HPXvT2dWhd79B/CKIv/8T52m553ozCBWSl5iJZj7NsbfnkSTq+Fh3HddAX+v23VsnZQ2Kb8I2NrnfL+WpGr/bklHCkWOwnVL8rpNqdeYurkfr/YFoMSyBWNsNYidIGCK41CFhO7F0rSysEWGJadt+C01+6T667PJaN7KEY/aRCOBJqK5DXAZMELJC33dui+lBe2s0S5fkm+NGwE+Snf9E2IPyqoL9qbWOLXENJ+F3B0zwlrUsyBUtUb8Pyvu35kOE/U/+PPURdvN6uSsCh1FfFaJJuk8u6waf/vlKkgthm/OSZl/KfMYrB7xWbvIDw4GRCR847pEemP2mnTIgvDJVu0GaCAAAAAAAA" alt="" />

# %% [markdown]
# And transform it to provide the circuit you want, but by only using the basis gates or instructions that a given quantum computer is able to accept. It also will optimize those instructions in order to minimize the effects of noise:

# %% [markdown]
# <img src="data:image/webp;base64,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" alt="" />

# %% [markdown]
# A central component of the Qiskit SDK, the transpiler is designed for modularity and extensibility. Its main goal is to write new circuit transformations (known as transpiler passes), and combine them with other existing passes, greatly reducing the depth and complexity of quantum circuits. Which passes are chained together and in which order has a major effect on the final outcome. This pipeline is determined by the [PassManager](https://docs.quantum.ibm.com/api/qiskit/qiskit.transpiler.PassManager) and [StagedPassManager](https://docs.quantum.ibm.com/api/qiskit/qiskit.transpiler.StagedPassManager) objects.
# 
# The `StagedPassManager` orchestrates the execution of one or more `PassManagers` and determines the order in which they are executed, while the `PassManager` object is merely a collection of one or more passes. Think of the `StagedPassManager` as the conductor in an orchestra, the `PassManagers` as the different instrument sections, and the `Pass` objects as the individual musicians.
# 
# In this way, you can compose hardware-efficient quantum circuits that let you execute utility-scale work while keeping noise manageable.  For more details, visit the [Transpile](https://docs.quantum.ibm.com/transpile) section in the IBM Quantum Platform docs.

# %% [markdown]
# ## The six stages
# 
# Rewriting quantum circuits to match hardware constraints and optimizing for performance can be far from trivial. Qiskit provides users the standard six stages of compilation flows with four pre-built transpilation pipelines. By default, the preset pass managers are composed of six stages, with several options in each stages:
# 

# %% [markdown]
# 
# - `Init`: This pass runs any initial passes that are required before we start embedding the circuit to the system. This typically involves unrolling custom instructions and converting the circuit to all single- and two-qubit gates. (By default this will just validate the circuit instructions and translate multi-qubit gates into single- and two-qubit gates.)
#   
# - `Layout`: This stage applies a layout, mapping the virtual qubits in the circuit to the physical qubits on a backend.
# 
# - `Routing`: This stage runs after a layout has been applied and will inject gates (i.e. swaps) into the original circuit to make it compatible with the backend’s connectivity.
#   
# - `Translation`: This stage translates the gates in the circuit to the target backend’s basis set.
# - `Optimization`: This stage runs the main optimization loop repeatedly until a condition (such as fixed depth) is reached.
# - `Scheduling`: This stage is for any hardware-aware scheduling passes.
# 
# 
# Qiskit also provides four pre-defined levels of transpilation that users can choose according to their needs. You can modify these preset pass managers, and in addition, you can construct a pass manager to build an entirely custom pipeline for transforming input circuits.
# 
# For most users who are not familiar with quantum circuit optimization by transpiling, we suggest to use one of the ready-made routines. However in this lab we will be diving deep (deeeeeeeep!) into each stage and the options within.

# %% [markdown]
# <div class="alert alert-block alert-success">
# <a id='ex1'></a>
# <a name='ex1'></a>
# 
# ### Exercise 1:
# 
# **Your Task:** Please match the description of what happens in each stage with the corresponding transpilation stage name in the code cell below.
# 
# - **A.** This stage centers around reducing the number of circuit operations and the depth of circuits with several optimization options.
# - **B.** This stage inserts the appropriate number of SWAP gates in order to execute the circuits using the selected layout.
# - **C.** this stage is used to translate any gates that operate on more than two qubits, into gates that only operate on one or two qubits.
# - **D.** This stage executes a sequence of gates, a one-to-one mapping from the "virtual" qubits to the "physical" qubits in an actual quantum device.
# - **E.** this pass can be thought of as explicitly inserting hardware-aware operations like delay instructions to account for the idle time between gate executions.
# - **F.** This stage translates (or unrolls) the gates specified in a circuit to the native basis gates of a specified backend.
# 
# <b>HINT: </b> The answer will always be a single capital letter with quotation marks. For example: "A"
# 
# </div>

# %%
ans = {}

# Place the correct letter next to the corresponding stage, inside a parenthesis
# example:  ans["test"] = "M"

ans["init"] = 'C'
ans["layout"] = 'D'
ans["routing"] = 'B'
ans["translation"] = 'F'
ans["optimization"] = 'A'
ans["scheduling"] = 'E'

# %%
# Submit your answer using following code

grade_lab2_ex1(ans)

# %% [markdown]
# Good work checking your understanding on each transpiling stage. Next let's see how to use Qiskit's six transpile stages with `preset_pass_managers`.

# %% [markdown]
# <a name='preset_passmanager'></a>
# 
# # Transpile with preset pass managers

# %% [markdown]
# In this part, we will explore how to use Qiskit's standard six transpiler stages. We will focus first on the four pre-defined transpile pipelines, and see how to build your own PassManager through a practice that uses the features and options of each pass with a pre-defined pipeline.
# 
# First, let's look at what Passmanager and `generate_preset_pass_manager` are. This part and later parts refers frequently to the [IBM Quantum Platform docs](https://docs.quantum.ibm.com/transpile) and the [API reference documentation](https://docs.quantum.ibm.com/api/qiskit/transpiler) on the transpiler, so we recommend you look at them together.
# 
# ## What is a (staged) pass manager?
# 
# A pass manager is an object that stores a list of transpiler passes and can execute them on a circuit. You can create a pass manager by initializing a [PassManager](https://docs.quantum.ibm.com/api/qiskit/qiskit.transpiler.PassManager) with a list of transpiler passes. To run the transpilation on a circuit, call the [run](https://docs.quantum.ibm.com/api/qiskit/qiskit.transpiler.PassManager#run) method with a circuit as input.
# 
# A staged pass manager, represented by the [`StagedPassManager`](https://docs.quantum.ibm.com/transpile/transpiler-stages) class, is a special kind of pass manager that represents a level of abstraction above that of a normal pass manager. While a normal pass manager is composed of several transpiler passes, a staged pass manager is composed of several pass managers. This is a useful abstraction because transpilation typically happens in discrete stages, with each stage being represented by a pass manager.
# 

# %% [markdown]
# ## `Preset Passmanagers`
# 
# Preset Passmanagers (`qiskit.transpiler.preset_passmanagers`) contains functions for generating the preset pass managers for transpiling. The preset pass managers are instances of StagedPassManager, which are used to execute the circuit transformations at the different optimization levels in the pre-defined transpiling pipeline. Here we introduce the functions used to generate the entire pass manager by using `generate_preset_pass_manager`.

# %% [markdown]
# <div class="alert alert-block alert-success">
# 
# ### Metaphors
# 
# It's easy to get lost in the terminology here, so think back to our metaphor about an orchestra conductor, sections, and individual musicians.
# 
# **Bonus Exercise:** Come up with your own metaphor for the relationship between a `StagedPassManager`, a `PassManager`, and a `pass`.  
# 
# Feel free to share your metaphor with others in the IBM Quantum Challenge Discord, or with your friends.
# </div>

# %% [markdown]
# ## `generate_preset_pass_manager`
# 
# In Qiskit, `generate_preset_pass_manager` is used to quickly generate a preset pass manager. This function provides a convenient and simple method to construct a standalone `PassManager` object with **optimization** levels and options for each pass. Let's explore those next.

# %% [markdown]
# # Optimization levels <a name='optimization_level'></a>

# %% [markdown]
# The `generate_preset_pass_manager` function has one required positional argument, optimization_level, that controls how much effort the transpiler spends on optimizing circuits. This argument is an integer taking one of the values 0, 1, 2, or 3.
# 
# Higher optimization levels generate more optimized circuits at the expense of longer compile times, and vice versa.

# %% [markdown]
# <div class="alert alert-block alert-success">
# <a id='ex2'></a>
# <a name='ex2'></a>
# 
# ### Exercise 2: Build a function to `evaluate` transpiled circuit
# 
# 
# Since the goal of transpiling is to improve the actual execution performance of the circuit, your goal is to create a function that measures the performance of the translated circuit. You will use this function later in this lab.
# 
# **Your Task:** Create a function called `scoring`. The function should receives the transpiled circuit, its *final layout*, and its target backend as its inputs. The function should then return a circuit score. The closer the score is to 0, the better.
# 
# Other notes:
# * Please use `FakeTorino` from the `qiskit-ibm-runtime` package for this whole lab.
# * The algorithm for calculating the actual score in `util.py` is from [Mapomatic](https://github.com/qiskit-community/mapomatic), and the main code has been updated to suit PrimitiveV2.
# * We have constructed some of this function for you.
# * You will need to get the final layout of transpiled circuit. These two pages should help you complete the code below to finish the function.
#   * [The layout section from the QuantumCircuit API documentation](https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.QuantumCircuit#layout)
#   * [The TranspileLayout section from the Qiskit Transpiler API documentation](https://docs.quantum.ibm.com/api/qiskit/qiskit.transpiler.TranspileLayout)
# </div>
# Imports
#%%
from qiskit.circuit.random import random_circuit
from qiskit.circuit.library import XGate, YGate
from qiskit_ibm_runtime.fake_provider import FakeTorino, FakeOsaka
from qiskit.transpiler import InstructionProperties, PassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler.preset_passmanagers.plugin import list_stage_plugins
from qiskit.transpiler.timing_constraints import TimingConstraints
from qiskit.transpiler.passes.scheduling import ASAPScheduleAnalysis,PadDynamicalDecoupling
from qiskit.visualization.timeline import draw, IQXStandard
from qiskit.transpiler import StagedPassManager
from qiskit.visualization import plot_circuit_layout
import matplotlib.pyplot as plt
import numpy as np
# Setup the grader
from qc_grader.challenges.iqc_2024 import (
    grade_lab2_ex1,
    grade_lab2_ex2,
    grade_lab2_ex3,
    grade_lab2_ex4,
    grade_lab2_ex5
)
# %%
### Create the scoring function
from qiskit.circuit.quantumcircuit import QuantumCircuit

def scoring( qc:QuantumCircuit, backend):
    from util import transpile_scoring

    layout = qc.layout##your code here
    fidelity = transpile_scoring(qc, layout, backend)
    score = 1-fidelity##your code here

    return score


# %%
# Submit your answer using following code

grade_lab2_ex2(scoring)

# %% [markdown]
# Now you have a function to measure the performance of a transpiled circuit. Before we move on to the next part, let's finish setting up everything we need to properly test our circuit on a fake backend.
# 
# Namely, a circuit, and a fake backend!

# %%
### Create a random circuit

## DO NOT CHANGE THE SEED NUMBER
seed = 10000

## Create circuit

num_qubits = 6
depth = 4
qc = random_circuit(num_qubits,depth,measure=False, seed=seed)

qc.draw('mpl')

# %% [markdown]
# <div class="alert alert-block alert-warning">
# 
# <b>Be careful!</b>
#     
# To pass the grader, do not change the seed values for `seed` or `seed_transpiler` throughout this whole lab.
# </div>

# %% [markdown]
# To test the performance of each optimization level, call `FakeTorino` and save it as `backend`.
# 
# `FakeTorino` has the connectivity and noise features of ibm_torino, which is the latest IBM Quantum backend with a Heron processor.

# %%
## Save FakeTorino as backend

backend = FakeTorino()

# %% [markdown]
# We are going to walk through running all four optimization levels and then compare our results at the end. In order to do this, we will create a few arrays to hold the information as we work, then use them later on.
# 
# In the next code cell, we're simply constructing these arrays. You can run it and move on for now.

# %%
circuit_depths = {
    'opt_lv_0': None,
    'opt_lv_1': None,
    'opt_lv_2': None,
    'opt_lv_3': None,
}
gate_counts = {
    'opt_lv_0': None,
    'opt_lv_1': None,
    'opt_lv_2': None,
    'opt_lv_3': None,
}

scores = {
    'opt_lv_0': None,
    'opt_lv_1': None,
    'opt_lv_2': None,
    'opt_lv_3': None,
}

# %% [markdown]
# ## Optimization level = 0 <a name='opt_lv_0'></a>

# %% [markdown]
# <div class="alert alert-block alert-info">
# 
# If at any point during these four sections you need help or clarification, please refer to <a href="https://docs.quantum.ibm.com/transpile/set-optimization">this documentation</a> for a better understanding of optimization_level.
# 
# </div>

# %% [markdown]
# Optimization level 0 is intended for **device characterization experiments** and, as such, only maps the input circuit to the constraints of the target backend without performing any optimizations. It performs layout/routing with [TrivialLayout](https://docs.quantum.ibm.com/api/qiskit/qiskit.transpiler.passes.TrivialLayout#triviallayout), where it selects the same physical qubit numbers as virtual and inserts SWAPs to make it work (using [StochasticSwap](https://docs.quantum.ibm.com/api/qiskit/qiskit.transpiler.passes.StochasticSwap#stochasticswap)).
# 
# Let's make a pass manager with optimization level = 0 using our FakeTorino backend and see the result.

# %%
# Make a pass manager with our desired optimization level and backend
pm_lv0 = generate_preset_pass_manager(backend=backend, optimization_level=0, seed_transpiler=seed)

# Run for our random circuit
tr_lv0 = pm_lv0.run(qc)

# uncomment the next line to draw circuit
#tr_lv0.draw('mpl', idle_wires=False, fold=60)

# %% [markdown]
# As mentioned previously, optimization_level=0 is performing a basic gate decomposition by using basis gates of the backend and mapping logical qubits to physical qubits with a same order of number. It maps logical qubit 0 to physical qubit 0 and maps logical qubit 1 to physical qubit 1.

# %% [markdown]
# <div class="alert alert-block alert-success">
# <a id='ex3'></a>
# <a name='ex3'></a>
# 
# ### Exercise 3: (Start)
# 
# **Your Task:** Find the circuit depth of the random circuit, the sum of the total gate number, and compute the performance score of this circuit using `scoring`. Use the provided code to save each of these results to our previously made array.
# 
# </div>

# %% [markdown]
# <div class="alert alert-block alert-info">
# 
# Tips:
# - for the `circuit_depths`: [IBM Documentation](https://docs.quantum.ibm.com/api/qiskit/0.42/circuit)
# - for the `gate_counts`: Must be a number. [This discussion from stackexchange may help you](https://quantumcomputing.stackexchange.com/questions/25931/qiskit-count-of-each-gates), More powerful handy tip: [Circuit Property](https://docs.quantum.ibm.com/api/qiskit/0.42/circuit#quantum-circuit-properties)
# - for the `scores`: Use the `scoring` function you previously made
# 
# </div>

# %%
### Your code here ###

circuit_depths['opt_lv_0'] =tr_lv0.depth()
gate_counts['opt_lv_0'] = sum(dict(tr_lv0.count_ops()).values())
scores['opt_lv_0'] =scoring(tr_lv0, backend)

### Don't change code after this line ###

print("Optimization level 0 results")
print("====================")
print("Circuit depth:", circuit_depths['opt_lv_0'])
print("Gate count:", gate_counts['opt_lv_0'])
print("Score:", scores['opt_lv_0'])

# %% [markdown]
# 
# 
# ## Optimization level = 1 <a name='opt_lv_1'></a>
# 
# Optimization level 1 performs a **`light optimization`**. Here's what that means:
# 
# - Layout/Routing: Layout is first attempted with [TrivialLayout](https://docs.quantum.ibm.com/api/qiskit/qiskit.transpiler.passes.TrivialLayout#triviallayout). If additional SWAPs are required, a layout with a minimum number of SWAPs is found by using [SabreSWAP](https://docs.quantum.ibm.com/api/qiskit/qiskit.transpiler.passes.SabreSwap#sabreswap), then it uses [VF2LayoutPostLayout](https://docs.quantum.ibm.com/api/qiskit/qiskit.transpiler.passes.VF2PostLayout)` to try to select the best qubits in the graph.
# - [InverseCancellation](https://docs.quantum.ibm.com/api/qiskit/qiskit.transpiler.passes.InverseCancellation#inversecancellation)
# - [1Q gate optimization](https://docs.quantum.ibm.com/api/qiskit/qiskit.transpiler.passes.Optimize1qGates#optimize1qgates)
# 
# Try making a pass manager on your own this time. Once again use the FakeTorino backend with `generate_preset_pass_manager`. Please also set the `optimization_level` to 1, and `seed_transpiler` = `seed`

# %%
# Make a pass manager with our desired optimization level and backend
pm_lv1 = generate_preset_pass_manager(backend=backend, optimization_level=1, seed_transpiler=seed)


# Run for our random circuit
tr_lv1 = pm_lv1.run(qc)

# uncomment the next line to draw circuit
tr_lv1.draw('mpl', idle_wires=False, fold=60)

# %% [markdown]
# You should now see logical qubits mapped into different physical qubit sets and a smaller number of gates. Just like last time, let's once again find the circuit depth of the random circuit, the sum of the total gate number, and compute the performance score of this circuit using `scoring`.

# %%
### Your code here ###

circuit_depths['opt_lv_1'] =tr_lv1.depth()
gate_counts['opt_lv_1'] =sum(dict(tr_lv1.count_ops()).values())
scores['opt_lv_1'] =scoring(tr_lv1, backend)

### Don't change code after this line ###

print("Optimization level 1 results")
print("====================")
print("Circuit depth:", circuit_depths['opt_lv_1'])
print("Gate count:", gate_counts['opt_lv_1'])
print("Score:", scores['opt_lv_1'])

# %% [markdown]
# ## Optimization level = 2 <a name='opt_lv_2'></a>
# 
# Optimization level 2 performs a **`medium optimization`**, which means:
# 
# - Layout/Routing: Optimization level 1 (without trivial) + heuristic optimized with greater search depth and trials of optimization function. Because [TrivialLayout](https://docs.quantum.ibm.com/api/qiskit/qiskit.transpiler.passes.TrivialLayout#triviallayout) is not used, there is no attempt to use the same physical and virtual qubit numbers.
# - [CommutativeCancellation](https://docs.quantum.ibm.com/api/qiskit/qiskit.transpiler.passes.CommutativeCancellation#commutativecancellation)
# 
# Let's make a pass manager with `optimization_level` of 2 this time, again using `generate_preset_pass_manager` with the FakeTorino backend and `seed_transpiler` = `seed`.

# %%
# Make a pass manager with our desired optimization level and backend
pm_lv2 = generate_preset_pass_manager(backend=backend, optimization_level=2, seed_transpiler=seed)

# Run for our random circuit
tr_lv2 = pm_lv2.run(qc)

# uncomment the next line to draw circuit
#tr_lv2.draw('mpl', idle_wires=False, fold=60)

# %% [markdown]
# Even at a glance, we can see that the number of gates has decreased quite a bit. The physical qubit to which the logical qubit is mapped remains unchanged. Now, let's measure the performance of transpiling. It is the same code as above.

# %%
### Your code here ###

circuit_depths['opt_lv_2'] =tr_lv2.depth()
gate_counts['opt_lv_2'] =sum(dict(tr_lv2.count_ops()).values())
scores['opt_lv_2'] =scoring(tr_lv2, backend)

### Don't change code after this line ###

print("Optimization level 2 results")
print("====================")
print("Circuit depth:", circuit_depths['opt_lv_2'])
print("Gate count:", gate_counts['opt_lv_2'])
print("Score:", scores['opt_lv_2'])

# %% [markdown]
# ## Optimization level = 3 <a name='opt_lv_3'></a>
# 
# Optimization level 3 performs a **`high optimization`**:
# 
# - Optimization level 2 + heuristic optimized on layout/routing further with greater effort/trials
# - Resynthesis of two-qubit blocks using [Cartan's KAK Decomposition](https://arxiv.org/abs/quant-ph/0507171).
# - Unitarity-breaking passes:
#     - `OptimizeSwapBeforeMeasure`: Remove swaps in front of measurements by re-targeting
#     the classical bit of the measure instruction to avoid SWAPs
#     - `RemoveDiagonalGatesBeforeMeasure`: Remove diagonal gates (like RZ, T, Z, etc.) before
#     a measurement. Including diagonal 2Q gates.
# 
# You know what to do next!

# %%
pm_lv3 = generate_preset_pass_manager(backend=backend, optimization_level=3, seed_transpiler=seed)

# Run for our random circuit
tr_lv3 = pm_lv3.run(qc)

# uncomment to draw circuit
#tr_lv3.draw('mpl', idle_wires=False, fold=60)

# %% [markdown]
# Surprising! Now we have a smaller number of gates! Let's see how it worked...

# %%
### Your code here ###

circuit_depths['opt_lv_3'] =tr_lv3.depth()
gate_counts['opt_lv_3'] =sum(dict(tr_lv3.count_ops()).values())
scores['opt_lv_3'] =scoring(tr_lv3, backend)

### Don't change code after this line ###

print("Optimization level 3 results")
print("====================")
print("Circuit depth:", circuit_depths['opt_lv_3'])
print("Gate count:", gate_counts['opt_lv_3'])
print("Score:", scores['opt_lv_3'])

# %% [markdown]
# Now that we have all our results, let's graph how much the depth of the circuit, the number of gates, and the evaluation score have changed according to each optimization level.

# %%
colors = ['#FF6666', '#66B2FF']
ax = ["opt_lv_0", "opt_lv_1", "opt_lv_2", "opt_lv_3"]
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

# Plot 1: Circuit Depth
ax1.semilogy(ax, [circuit_depths[key] for key in ax],'o-',markersize=9, color='#FF6666', label="Depth")
ax1.set_xlabel("Optimization Level", fontsize=12)
ax1.set_ylabel("Depth", fontsize=12)
ax1.set_title("Circuit Depth", fontsize=14)
ax1.legend(fontsize=10)

# Plot 2: Total Number of Gates
ax2.semilogy(ax, [gate_counts[key] for key in ax],'^-',markersize=9, color='#66B2FF', label="Counts")
ax2.set_xlabel("Optimization Level", fontsize=12)
ax2.set_ylabel("Gate Count", fontsize=12)
ax2.set_title("Gate Count", fontsize=14)
ax2.legend(fontsize=10)

# Plot 3: Score of Transpiled Circuit
ax3.semilogy(ax, [scores[key] for key in ax],'*-',markersize=9, label="Score")
ax3.set_xlabel("Optimization Level", fontsize=12)
ax3.set_ylabel("Score", fontsize=12)
ax3.set_title("Score", fontsize=14)
ax3.legend(fontsize=10)

fig.tight_layout()
plt.show()

# %% [markdown]
# <div class="alert alert-block alert-success">
# 
# ### Exercise 3: (Finish)
# 
# **Your Task:** Above you did a great amount of work to construct your different pass managers, test them, and save information about each one to it's corresponding array. You graphed that information to visibly compare your results. Now, submit all four pass managers to the grader.
# 
# Make sure you didn't change any seed values!
# 
# </div>
# 

# %%
# Submit your answer using following code

ans = [pm_lv0, pm_lv1, pm_lv2, pm_lv3]

grade_lab2_ex3(ans)

# %% [markdown]
# # Transpiler stage details with options <a name='transpiler_options'></a>
# 
# So far, we've looked at four predefined transpiler pipelines offered by Qiskit using random circuits. One of the key features of the Qiskit SDK v1.0 transpiler is the ease with which users can build and configure a custom passmanager. Configure your own pass manager by using six standard passes along with the optimization level discussed above, but also choosing the option or plug-in for each pass as you like. Another way is to create your pass manager with as many passes as you want by using `StagedPassManager`. In addition, you can create and deploy a new pass manager and implement it in the the form of a [plugin](https://docs.quantum.ibm.com/transpile/create-a-transpiler-plugin) in Qiskit so that other users can use it.
# 
# Let's look at all three methods in the following sections, starting with the easiest way first.

# %% [markdown]
# ## Init stage <a name='init'></a>
# 
# This first stage does very little by default and is primarily useful if you want to include your own initial optimizations. Because most layout and routing algorithms are only designed to work with single- and two-qubit gates, this stage is also used to translate any gates that operate on more than two qubits, into gates that only operate on one or two qubits.
# 
# If you specify an argument at this stage for the qubits you want to use, that value overrides all the passes that could change it. You can find more details in the [Default settings and configuration options documentation](https://docs.quantum.ibm.com/transpile/defaults-and-configuration-options).
# 
# First, let's check the possible options of the init stage we can use.

# %%
list_stage_plugins("init")

# %% [markdown]
# Without extra plugins, only default options exist. This process includes several transpiler pass plugins depending on the `optimization_level` - let's take a look inside it.
# 
# For this, let's create a pass manager with `init_method="default"` options.

# %%
print("Plugins run by default init stage")
print("=================================")

for i in range(4):
    print(f"\nOptimization level {i}:")
    pm = generate_preset_pass_manager(backend=backend, optimization_level=i, init_method="default", seed_transpiler=1000)
    for task in pm.init.to_flow_controller().tasks:
        print(" -", type(task).__name__)

# %% [markdown]
# Here is a table of each plugin's API docs. Take a look if you want to dive more deeply into any of them.
# 
# | Plugin | Description | API docs link|
# |---- | ---- | ------|
# |UnitarySynthesis| Synthesize unitaries over some basis gates | https://docs.quantum.ibm.com/api/qiskit/qiskit.transpiler.passes.UnitarySynthesis#unitarysynthesis |
# |HighLevelSynthesis |Synthesize higher-level objects and unroll custom definitions| https://docs.quantum.ibm.com/api/qiskit/qiskit.transpiler.passes.HighLevelSynthesis#highlevelsynthesis |
# | BasisTranslator | Translates gates to a target basis by searching for a set of translations from a given EquivalenceLibrary | https://docs.quantum.ibm.com/api/qiskit/qiskit.transpiler.passes.BasisTranslator#basistranslator |
# | InverseCancellation| Cancel specific gates which are inverses of each other when they occur back-to- back | https://docs.quantum.ibm.com/api/qiskit/qiskit.transpiler.passes.InverseCancellation#inversecancellation |
# | OptimizeSwapBeforeMeasure | Moves the measurements around to avoid SWAPs | NA |
# | RemoveDiagonalGatesBeforeMeasure | Remove diagonal gates (including diagonal 2Q gates) before a measurement | https://docs.quantum.ibm.com/api/qiskit/qiskit.transpiler.passes.RemoveDiagonalGatesBeforeMeasure |
# 

# %% [markdown]
# ## Layout stage <a name='layout'></a>
# 
# The next stage involves the layout or connectivity of the backend a circuit will be sent to. In general, quantum circuits are abstract entities whose qubits are "virtual" or "logical" representations of actual qubits used in computations. To execute a sequence of gates, a one-to-one mapping from the "virtual" qubits to the "physical" qubits in an actual quantum device is necessary. This mapping is stored as a Layout object and is part of the constraints defined within a backend's instruction set architecture (ISA).
# 

# %% [markdown]
# ![layout_mapping](data:image/webp;base64,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)

# %% [markdown]
# 
# The choice of mapping is extremely important for minimizing the number of SWAP operations needed to map the input circuit onto the device topology and ensure the most well-calibrated qubits are used. Due to the importance of this stage, the preset pass managers try a few different methods to find the best layout. Typically this involves two steps: first, try to find a "perfect" layout (a layout that does not require any SWAP operations), and then, a heuristic pass that tries to find the best layout to use if a perfect layout cannot be found. There are two Passes typically used for this first step:
# 
# - [TrivialLayout](https://docs.quantum.ibm.com/api/qiskit/qiskit.transpiler.passes.TrivialLayout#triviallayout): Naively maps each virtual qubit to the same numbered physical qubit on the device (i.e., [0,1,1,3] -> [0,1,1,3]). This is historical behavior only used in optimzation_level=1 to try to find a perfect layout. If it fails, VF2Layout is tried next.
# - [VF2Layout](https://docs.quantum.ibm.com/api/qiskit/qiskit.transpiler.passes.VF2Layout): This is an AnalysisPass that selects an ideal layout by treating this stage as a subgraph isomorphism problem, solved by the VF2++ algorithm. If more than one layout is found, a scoring heuristic is run to select the mapping with the lowest average error.
#   
# Then for the heuristic stage, two passes are used by default:
# 
# - [DenseLayout](https://docs.quantum.ibm.com/api/qiskit/qiskit.transpiler.passes.DenseLayout): Finds the sub-graph of the device with the greatest connectivity and that has the same number of qubits as the circuit (used for optimization level 1 if there are control flow operations (such as IfElseOp) present in the circuit).
# - [SabreLayout](https://docs.quantum.ibm.com/api/qiskit/qiskit.transpiler.passes.SabreLayout): This pass selects a layout by starting from an initial random layout and repeatedly running the SabreSwap algorithm. This pass is only used in optimization levels 1, 2, and 3 if a perfect layout isn't found via the VF2Layout pass. For more details on this algorithm, refer to the paper [arXiv:1809.02573](arXiv:1809.02573).
# 
# We can find these four passes by using the same code as the init stage.

# %%
list_stage_plugins("layout")

# %% [markdown]
# First, let's check which tasks will be enabled by which optimization_level with the `layout_method='default'` option by running the code below.

# %%
print("Plugins run by default layout stage")
print("=================================")
for i in range(4):
    print(f"\nOptimization level {i}:")
    pm = generate_preset_pass_manager(backend=backend, optimization_level=i, layout_method='default', seed_transpiler=seed)
    qc_tr = pm.run(qc)

    for controller_group in pm.layout.to_flow_controller().tasks:
        tasks = getattr(controller_group, "tasks", [])
        for task in tasks:
            print(" - " , str(type(task).__name__))
    print(qc_tr.layout.final_index_layout())
    display(plot_circuit_layout(pm.run(qc), backend))

# %% [markdown]
# Now let's compare the score and layout of each option with optimization_level=3.

# %%
for option in list_stage_plugins("layout"):
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3, layout_method=option, seed_transpiler=seed)
    qc_tr = pm.run(qc)
    score = scoring(qc_tr, backend)

    print(f"Layout method = {option}")
    print(f"Score: {score:.6f}")
    print(f"Layout: {qc_tr.layout.final_index_layout()}\n")

# %% [markdown]
# ## Routing stage <a name='routing'></a>
# 
# In order to implement a two-qubit gate between qubits that are not directly connected on a quantum device, one or more SWAP gates must be inserted into the circuit to move the qubit states around until they are adjacent on the device gate map. Each SWAP gate represents an expensive and noisy operation to perform. Thus, finding the minimum number of SWAP gates needed to map a circuit onto a given device is an important step in the transpilation process. For efficiency, this stage is typically computed alongside the Layout stage by default, but they are logically distinct from one another. The Layout stage selects the hardware qubits to be used, while the Routing stage inserts the appropriate amount of SWAP gates in order to execute the circuits using the selected layout.
# 
# However, finding the optimal SWAP mapping is hard. In fact, it is an NP-hard problem, and is thus prohibitively expensive to compute for all but the smallest quantum devices and input circuits. To work around this, Qiskit uses a stochastic heuristic algorithm called SabreSwap to compute a good, but not necessarily optimal, SWAP mapping. The use of a stochastic method means that the circuits generated are not guaranteed to be the same over repeated runs. Indeed, running the same circuit repeatedly results in a distribution of circuit depths and gate counts at the output. It is for this reason that many users choose to run the routing function (or the entire StagedPassManager) many times and select the lowest-depth circuits from the distribution of outputs.

# %%
list_stage_plugins("routing")

# %%
print("Number of each gates of transpiled circuit and the score")
print("=================================")
for i in range(4):
    print(f"\nOptimization level {i}:")
    pm = generate_preset_pass_manager(backend=backend, optimization_level=i, routing_method='basic', seed_transpiler=seed)
    qc_tr = pm.run(qc)
    score = scoring(qc_tr, backend)
    for key, value in qc_tr.count_ops().items():
        print(key, ":", value)
    print(f"Score: {score:.6f}")

# %%
print("Plugins run by basic routing stage")
print("=================================")
for i in range(4):
    print(f"\nOptimization level {i}:")
    pm = generate_preset_pass_manager(backend=backend, optimization_level=i, routing_method='basic', seed_transpiler=seed)

    for controller_group in pm.routing.to_flow_controller().tasks:
        tasks = getattr(controller_group, "tasks", [])
        for task in tasks:
            print(" - " , str(type(task).__name__))
    display(pm.routing.draw())
    print(pm.run(qc).layout.final_index_layout())

# %%
## process stopped due to lookahead
options = ['basic','sabre', 'stochastic']

for option in options:
    print(f"Layout option = {option}:")
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3, routing_method=option, seed_transpiler=seed)
    qc_tr = pm.run(qc)
    score = scoring(qc_tr, backend)

    print(f"Score: {score:.6f}")

    for key, value in qc_tr.count_ops().items():
        print(key, ":", value)
    print("\n")

# %% [markdown]
# ## Translation stage <a name='translation'></a>
# 
# When writing a quantum circuit, you are free to use any quantum gate (unitary operation) that you like, along with a collection of non-gate operations such as qubit measurement or reset instructions. However, most quantum devices only natively support a handful of quantum gate and non-gate operations. These native gates are part of the definition of a target's ISA and this stage of the preset PassManagers translates (or unrolls) the gates specified in a circuit to the native basis gates of a specified backend. This is an important step, as it allows the circuit to be executed by the backend, but typically leads to an increase in the depth and number of gates.
# 
# Two special cases are especially important to highlight, and help illustrate what this stage does.
# 
# 1. If a SWAP gate is not a native gate to the target backend, this requires three CNOT gates: As a product of three CNOT gates, a SWAP is an expensive operation to perform on noisy quantum devices. However, such operations are usually necessary for embedding a circuit into the limited gate connectivities of many devices. Thus, minimizing the number of SWAP gates in a circuit is a primary goal in the transpilation process.
# 
# 2. A Toffoli, or controlled-controlled-not gate (ccx), is a three-qubit gate. Given that our basis gate set includes only single- and two-qubit gates, this operation must be decomposed. However, it is quite costly: For every Toffoli gate in a quantum circuit, the hardware may execute up to six CNOT gates and a handful of single-qubit gates. This example demonstrates that any algorithm making use of multiple Toffoli gates will end up as a circuit with large depth and will therefore be appreciably affected by noise.
# 
# Let's check how many options we can use.

# %%
list_stage_plugins("translation")

# %% [markdown]
# The basic options in Qiskit are 'translator' and 'synthesis'. Let's count the total gate numbers, circuit depth, and scores of transpiled circuits with the default option ('translator') and each optimization level.

# %%
print("Number of each gates of transpiled circuit")
print("=================================")

for i in range(4):
    print(f"\nOptimization level {i}:")
    pm = generate_preset_pass_manager(backend=backend, optimization_level=i, translation_method='translator', seed_transpiler=seed)
    qc_tr = pm.run(qc)
    score = scoring(qc_tr, backend)
    for key, value in qc_tr.count_ops().items():
        print(key, ":", value)
    print(f"Score: {score:.6f}")

# %% [markdown]
# Let's count total gate numbers, circuit depth, and scores of transpiled circuits with each option and optimization_level=3.

# %%
options = ['translator', 'synthesis']

print("Number of each gates of transpiled circuit")
print("=================================")

for option in options:
    print(f"Layout option = {option}:")
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3, translation_method=option, seed_transpiler=seed)
    qc_tr = pm.run(qc)
    score = scoring(qc_tr, backend)
    for key, value in qc_tr.count_ops().items():
        print(key, ":", value)
    print(f"Score: {score:.6f}")
    print("\n")

# %% [markdown]
# Let's plot it on a graph.

# %%
tr_depths = []
tr_gate_counts = []
tr_scores = []

options = ['translator', 'synthesis']

for i in range(4):
    for option in options:
        pm = generate_preset_pass_manager(backend=backend, optimization_level=i, translation_method=option, seed_transpiler=seed)

        tr_depths.append(pm.run(qc).depth())
        tr_gate_counts.append(sum(pm.run(qc).count_ops().values()))
        tr_scores.append(scoring(pm.run(qc), backend))

# %%
colors = ['#FF6666', '#66B2FF']
markers = [ '^', '*']
linestyles = ['-.', ':']

opt_list = []
for i in range(4):
    opt_list.append(f"Optimization Level {i}")

ax = opt_list
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

# Plot 1: Circuit Depth
for i in range(2):
    ax1.plot(ax, tr_depths[i:i+4], label=options[i], marker=markers[i], markersize=9, linestyle=linestyles[i], color=colors[i], linewidth=2)
ax1.set_xlabel("translation options", fontsize=12)
ax1.set_ylabel("Depth", fontsize=12)
ax1.set_title("Circuit Depth of Transpiled Circuit", fontsize=14)
ax1.legend(fontsize=10)

# Plot 2: Total Number of Gates
for i in range(2):
    ax2.plot(ax, tr_gate_counts[i:i+4], label=options[i], marker=markers[i], markersize=9, linestyle=linestyles[i], color=colors[i], linewidth=2)
ax2.set_xlabel("translation options", fontsize=12)
ax2.set_ylabel("# of Total Gates", fontsize=12)
ax2.set_title("Total Number of Gates of Transpiled Circuit", fontsize=14)
ax2.legend(fontsize=10)

# Plot 3: Score of Transpiled Circuit
for i in range(2):
    ax3.plot(ax, tr_scores[i:i+4], label=options[i], marker=markers[i],markersize=9, linestyle=linestyles[i], color=colors[i], linewidth=2)
ax3.set_xlabel("translation options", fontsize=12)
ax3.set_ylabel("Score of Transpiled Circuit", fontsize=12)
ax3.set_title("Score of Transpiled Circuit", fontsize=14)
ax3.legend(fontsize=10)

fig.tight_layout()
plt.show()

# %% [markdown]
# ## Optimization stage <a name='optimization'></a>
# 
# Decomposing quantum circuits into the basis gate set of the target device, and the addition of swap gates needed to match hardware topology, conspire to increase the depth and gate count of quantum circuits. Fortunately many routines for optimizing circuits by combining or eliminating gates exist. In some cases these methods are so effective the output circuits have lower depth than the inputs. In other cases, not much can be done, and the computation may be difficult to perform on noisy devices. Different gate optimizations are turned on with different optimization_level values.
# 
# - For optimization_level=1, this stage prepares [Optimize1qGatesDecomposition](https://docs.quantum.ibm.com/api/qiskit/qiskit.transpiler.passes.Optimize1qGatesDecomposition) and [CXCancellation](https://docs.quantum.ibm.com/api/qiskit/qiskit.transpiler.passes.CXCancellation), which combine chains of single-qubit gates and cancel any back-to-back CNOT gates.
# - For optimization_level=2, this stage uses the [CommutativeCancellation](https://docs.quantum.ibm.com/api/qiskit/qiskit.transpiler.passes.CommutativeCancellation) pass instead of `CXCancellation`, which removes redundant gates by exploiting commutation relations.
# - For optimization_level=3, this stage prepares the following passes- [Collect2qBlocks](https://docs.quantum.ibm.com/api/qiskit/qiskit.transpiler.passes.Collect2qBlocks), [ConsolidateBlocks](https://docs.quantum.ibm.com/api/qiskit/qiskit.transpiler.passes.ConsolidateBlocks), [UnitarySynthesis](https://docs.quantum.ibm.com/api/qiskit/qiskit.transpiler.passes.UnitarySynthesis), [Optimize1qGateDecomposition](https://docs.quantum.ibm.com/api/qiskit/qiskit.transpiler.passes.Optimize1qGatesDecomposition), [CommutativeCancellation](https://docs.quantum.ibm.com/api/qiskit/qiskit.transpiler.passes.CommutativeCancellation).
# Additionally, this stage also executes a few final checks to make sure that all instructions in the circuit are composed of the basis gates available on the target backend.
# 
# 
# At this stage, we can set up two options:
# 
# - approximation_degree (float, in the range 0-1 | None) - Heuristic dial used for circuit approximation (1.0 = no approximation, 0.0 = maximal approximation). The default value is 1.0. Specifying None sets the approximation degree to the reported error rate. See the Approximation degree section for more details.
# - optimization_method (str) - The plugin name to use for the optimization stage. By default an external plugin is not used. You can see a list of installed plugins by running list_stage_plugins() with optimization for the stage_name argument.
# 
# 

# %% [markdown]
# <div class="alert alert-block alert-info">
#     
# #### Bonus exercise (not graded)
# 
# As this stage is strongly related to optimizing gates of circuits, let's try to plot the following with each optimization_level
# 
# 1. total gate number
# 2. circuit depth
# 3. score of the transpiled circuit
# 
# with different approximation_degree values - $[ 0, 0.1, 0.2, ... ,0.9, 1]$ by finishing below codes.
# 
# </div>

# %%
tr_depths = []
tr_gate_counts = []
tr_scores = []

approximation_degree_list = np.linspace(0,1,10)

for i in range(4):
    for j in approximation_degree_list:
        # your code here #
        
        print(f"\nOptimization level {i}, approximate degree {j}:")
        pm = generate_preset_pass_manager(backend=backend, optimization_level=i, translation_method='translator', seed_transpiler=seed, approximation_degree=j)
        qc_tr = pm.run(qc)
    

        tr_depths.append(qc_tr.depth())
        tr_gate_counts.append(sum(qc_tr.count_ops().values()))
        tr_scores.append(scoring(qc_tr, backend))
        # your code here #

# %%
colors = ['#FF6666', '#FFCC66', '#99FF99', '#66B2FF']
markers = ['o', 's', '^', '*']
linestyles = ['-', '--', '-.', ':']

ax = approximation_degree_list
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

# Plot 1: Circuit Depth
for i in range(4):
    ax1.plot(ax, tr_depths[i::4], label=f"Optimization Level {i}", marker=markers[i], markersize=9, linestyle=linestyles[i], color=colors[i], linewidth=2)
ax1.set_xlabel("Approximation Degree", fontsize=12)
ax1.set_ylabel("Depth", fontsize=12)
ax1.set_title("Circuit Depth of Transpiled Circuit", fontsize=14)
ax1.legend(fontsize=10)

# Plot 2: Total Number of Gates
for i in range(4):
    ax2.plot(ax, tr_gate_counts[i::4], label=f"Optimization Level {i}", marker=markers[i], markersize=9, linestyle=linestyles[i], color=colors[i], linewidth=2)
ax2.set_xlabel("Approximation Degree", fontsize=12)
ax2.set_ylabel("# of Total Gates", fontsize=12)
ax2.set_title("Total Number of Gates of Transpiled Circuit", fontsize=14)
ax2.legend(fontsize=10)

# Plot 3: Score of Transpiled Circuit
for i in range(4):
    ax3.plot(ax, tr_scores[i::4], label=f"Optimization Level {i}", marker=markers[i],markersize=9, linestyle=linestyles[i], color=colors[i], linewidth=2)
ax3.set_xlabel("Approximation Degree", fontsize=12)
ax3.set_ylabel("Score of Transpiled Circuit", fontsize=12)
ax3.set_title("Score of Transpiled Circuit", fontsize=14)
ax3.legend(fontsize=10)

fig.tight_layout()
plt.show()

# %% [markdown]
# ## Scheduling stage <a name='scheduling'></a>
# 
# This last stage is only run if it is explicitly called for (similar to the init stage) and does not run by default (though a method can be specified by setting the 1scheduling_method1 argument when calling `generate_preset_pass_manager`). The scheduling stage is typically used once the circuit has been translated to the target basis, mapped to the device, and optimized. These passes focus on accounting for all the idle time in a circuit. At a high level, the scheduling pass can be thought of as explicitly inserting delay instructions to account for the idle time between gate executions and to inspect how long the circuit will be running on the backend.
# 
# First, let'sc check which options we can use.

# %%
list_stage_plugins("scheduling")

# %% [markdown]
# To use scheduling options, let's first prepare `timing_constraints`, which contains relevant information about the backend to get the optimized pulse schedule.

# %%
backend_timing = backend.target.timing_constraints()
timing_constraints = TimingConstraints(
    granularity=backend_timing.granularity,
    min_length=backend_timing.min_length,
    pulse_alignment=backend_timing.pulse_alignment,
    acquire_alignment=backend_timing.acquire_alignment )

# %%
# Run with optimization level 3 and 'asap' scheduling pass
pm_asap = generate_preset_pass_manager(
    optimization_level=3,
    backend=backend,
    timing_constraints=timing_constraints,
    scheduling_method="asap",
    seed_transpiler=seed,
)

# %%
my_style = {
    'formatter.general.fig_width': 40,
    'formatter.general.fig_unit_height': 1,
}

draw(pm_asap.run(qc), style=IQXStandard(**my_style), show_idle=False, show_delays=True)

# %%
pm_alap = generate_preset_pass_manager(
    optimization_level=3,
    backend=backend,
    timing_constraints=timing_constraints,
    scheduling_method="alap",
    seed_transpiler=seed,
)
draw(pm_alap.run(qc), style=IQXStandard(**my_style), show_idle=False, show_delays=True)

# %% [markdown]
# As you can see, these two circuits have a lot of delays, but the position and order are different. Depends on circuit and backend, this scheduling can bring differences in performance. Let's check and compare the scores of these two circuits.

# %%
print("Score")
print("===============")
print(f"asap: {scoring(pm_asap.run(qc), backend):.6f}")
print(f"alap: {scoring(pm_alap.run(qc), backend):.6f}")

# %% [markdown]
# <div class="alert alert-block alert-success">
# <a id='ex4'></a>
# <a name='ex4'></a>
# 
# ### Exercise 4:
# 
# **Your Task:** At this point, you should feel like a professional at constructing different pass managers. Please make a pass manager with the following options:
# 
# 1. optimization level = 3
# 2. "sabre" layout
# 3. "sabre" routing
# 4. "synthesis" translation
#    
# </div>

# %%
pm_ex4 = generate_preset_pass_manager(
    backend=backend,
    routing_method='sabre',
    layout_method="sabre",
    translation_method="synthesis",
    optimization_level = 3
    ### Write your code below here ###



    ### Don't change any code past this line ###
)

# %%
# Submit your answer using following code

grade_lab2_ex4(pm_ex4)

# %% [markdown]
# # Build your own pass managers with staged pass manager <a name='staged_pm'></a>
# 
# One of the powerful features of the Qiskit v1.0 transpiler is its flexibility. It allows you compose a `PassManager` with only two or three stages. It also allows you to put your own `Pass` at desired stages.

# %% [markdown]
# ## Build `Dynamical Decoupling` pass <a name='dd'></a>
# 
# Here, we will try to build a custom `scheduling` pass to perform `Dynamical Decoupling`, which works by adding pulse sequences (known as dynamical decoupling sequences) to idle qubits to flip them around the Bloch sphere, cancelling the effect of noise channels and thereby suppressing decoherence. These pulse sequences are similar to refocusing pulses used in nuclear magnetic resonance. For a full description, see [A Quantum Engineer's Guide to Superconducting Qubits](https://arxiv.org/abs/1904.06560). To learn more, you can hear from Nick Bronn about dynamic decoupling in this [Qiskit video](https://www.youtube.com/watch?v=67jRWQuW3Fk).
# 
# For more details, you can check out the [Create a pass manager for dynamical decoupling](https://docs.quantum.ibm.com/transpile/dynamical-decoupling-pass-manager) documentation.
# 
# We will continue to use our same quantum circuit. A dynamical decoupling sequence is a series of gates that compose to the identity and are spaced regularly in time. For example, start by creating a simple sequence called XY4 consisting of four gates.

# %%
X = XGate()
Y = YGate()

dd_sequence = [X, Y, X, Y]

# %% [markdown]
# Because of the regular timing of dynamical decoupling sequences, information about the YGate must be added to the target because it is not a basis gate, whereas the XGate is. We know a priori that the YGate has the same duration and error as the XGate, however, so we can just retrieve those properties from the target and add them back for the YGate. This is also why the basis_gates were saved separately, since we are adding the YGate instruction to the target, although it is not an actual basis gate of our ibm_torino backend (here FakeTorino, which has a same configurations including basis gate sets of real ibm_torino).

# %%
backend=FakeTorino()

target = backend.target

y_gate_properties = {}
for qubit in range(target.num_qubits):
    y_gate_properties.update(
        {
            (qubit,): InstructionProperties(
                duration=target["x"][(qubit,)].duration,
                error=target["x"][(qubit,)].error,
            )
        }
    )

target.add_instruction(YGate(), y_gate_properties)

# %% [markdown]
# Next, execute the custom passes.

# %% [markdown]
# <div class="alert alert-block alert-success">
# <a id='ex5'></a>
# <a name='ex5'></a>
# 
# ### Exercise 5:
# 
# **Your Task:** Instantiate the PassManager with `ASAPScheduleAnalysis`s and `PadDynamicalDecoupling`. Run `ASAPScheduleAnalysis` first to add timing information about the quantum circuit before the regularly-spaced dynamical decoupling sequences can be added. These passes are run on the circuit with .run().
# </div>

# %%
dd_pm = PassManager(
    [
        ## your code here
        ASAPScheduleAnalysis(target = target),
        
        
        ## your code here
        PadDynamicalDecoupling(dd_sequence = dd_sequence, target = target),
        
        
        
    ]
)

# %% [markdown]
# Now let's see how it works by comparing it with a `Timing` of `asap` scheduling option. First, let's bring the timing drawing from above.
# 

# %%
draw(pm_asap.run(qc), style=IQXStandard(**my_style), show_idle=False, show_delays=True)

# %% [markdown]
# Now, let's make a custom scheduling `Pass` by using the function we've created.
# 
# For this new `scheduling pass` we will use `StagedPassManager`. We can make a PassManager that only has one pass. After making a custom staged PassManager, we will apply this to the transpiled circuit with pm_asap, that we already created above.

# %%
staged_pm_dd = StagedPassManager(
    stages=["scheduling"],
    scheduling=dd_pm
)

# %%
qc_tr = pm_asap.run(qc)
draw(staged_pm_dd.run(qc_tr), style=IQXStandard(**my_style), show_idle=False, show_delays=True)

# %%
# Submit your answer using following code

grade_lab2_ex5(staged_pm_dd)

# %% [markdown]
# ## (Bonus) Ecosystem and Qiskit Transpiler plugin <a name='plugin'></a>
# 
# To facilitate the development and reuse of custom transpilation code by the wider community of Qiskit users, the Qiskit SDK supports a plugin interface that enables third-party Python packages to declare that they provide extended transpilation functionality accessible via Qiskit.
# 
# Currently, third-party plugins can provide extended transpilation functionality in three ways:
# 
# - [A transpiler stage plugin](https://docs.quantum.ibm.com/api/qiskit/transpiler_plugins) provides a pass manager that can be used in place of one of the six stages of a preset staged pass manager: init, layout, routing, translation, optimization, and scheduling.
# - [A unitary synthesis plugin](https://docs.quantum.ibm.com/api/qiskit/qiskit.transpiler.passes.synthesis.plugin.UnitarySynthesisPlugin) provides extended functionality for unitary gate synthesis.
# - [A high-level synthesis plugin](https://docs.quantum.ibm.com/api/qiskit/qiskit.transpiler.passes.synthesis.plugin.HighLevelSynthesisPlugin) provides extended functionality for synthesizing "high-level objects" such as linear functions or Clifford operators. High-level objects are represented by subclasses of the [Operation](https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.Operation) class.
# 
# 
# Refer to [this page](https://docs.quantum.ibm.com/transpile/transpiler-plugins) for more details, including how to install and use plugins.
# 
# Also, you can be a `contributor` of these plugins! Creating a transpiler plugin is a great way to share your transpilation code with the wider Qiskit community, allowing other users to benefit from the functionality you've developed. [Here](https://docs.quantum.ibm.com/transpile/create-a-transpiler-plugin) you can find guidelines and instructions on how to contribute to the Qiskit community by providing nice transpiler plugins.

# %% [markdown]
# # Additional information
# 
# **Created by:** Sophy Shin, Sumit Suresh Kale, Abby Cross
# 
# **Advised by:** Va Barbosa, Junye Huang, Brian Ingmanson
# 
# **Version:** 1.1.0



# %%
