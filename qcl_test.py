# import numpy as np
# import matplotlib.pyplot as plt
#
#
# # Function to define the quantum control landscape
# def quantum_control_landscape(a1, a2):
#     # Replace this with your actual landscape function
#     J = np.sin(a1) + np.cos(a2)
#     return J
#
#
# # Generate values for a1 and a2
# a1_values = np.linspace(-2.5, 2.5, 100)
# a2_values = np.linspace(-2.5, 2.5, 100)
#
# # Create a meshgrid for a1 and a2
# a1_mesh, a2_mesh = np.meshgrid(a1_values, a2_values)
#
# # Calculate J values for each combination of a1 and a2
# J_values = quantum_control_landscape(a1_mesh, a2_mesh)
#
# # Plot the quantum control landscape with color intensity levels
# plt.contourf(a1_mesh, a2_mesh, J_values, cmap='viridis')
# plt.colorbar(label='J Value')  # Add a color bar to the plot
#
# # Set labels for axes
# plt.xlabel('a1')
# plt.ylabel('a2')
# plt.title('Quantum Control Landscape')
#
# plt.show()


import numpy as np
import matplotlib.pyplot as plt
from qutip import *


# Define the Hamiltonian function
def hamiltonian(a1, a2):
    # Replace this with your actual Hamiltonian
    H = Qobj(np.array([[1, a1], [a2, -1]]))
    return H


# Define the time evolution operator
def time_evolution_operator(H, T):
    return (-1j * H * T).expm()


# Define the quantum landscape function with T fixed to pi
def quantum_landscape(a1, a2):
    # Define initial and final states |0⟩ and |1⟩
    initial_state = basis(2, 0)
    final_state = basis(2, 1)

    # Calculate the time evolution operators with T fixed to pi
    U1 = time_evolution_operator(hamiltonian(a1, a2), np.pi / 2)
    U2 = time_evolution_operator(hamiltonian(a1, a2), np.pi / 2)

    # Calculate the expectation value
    expectation_value = expect(final_state, U2 * final_state * U1 * initial_state).real

    return expectation_value

#
# # Generate values for a1 and a2
# a1_values = np.linspace(-5, 5, 100)
# a2_values = np.linspace(-5, 5, 100)
#
# # Create an empty array for J_values
# J_values = np.empty((len(a1_values), len(a2_values)))
#
# # Calculate the quantum landscape values using nested loops
# for i, a1 in enumerate(a1_values):
#     for j, a2 in enumerate(a2_values):
#         J_values[i, j] = quantum_landscape(a1, a2)
#
# # Plot the quantum landscape
# plt.imshow(J_values, extent=(a1_values.min(), a1_values.max(), a2_values.min(), a2_values.max()), origin='lower',
#            aspect='auto', cmap='viridis')
# plt.colorbar(label='J Value')
# plt.xlabel('a1')
# plt.ylabel('a2')
# plt.title('Quantum Landscape for T = π')
# plt.show()



