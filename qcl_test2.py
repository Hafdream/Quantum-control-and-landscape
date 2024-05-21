import numpy as np
import matplotlib.pyplot as plt
from qutip import *


# Define the Landau-Zener Hamiltonian
def landau_zener_hamiltonian(delta, v):
    H = Qobj(np.array([[delta / 2, v], [v, -delta / 2]]))
    return H


# Define the time evolution operator
def time_evolution_operator(H, T):
    return (-1j * H * T).expm()


# Define the Landau-Zener control landscape function
def landau_zener_control_landscape(delta_values, v_values, T):
    landscape = np.zeros((len(delta_values), len(v_values)))

    for i, delta in enumerate(delta_values):
        for j, v in enumerate(v_values):
            H = landau_zener_hamiltonian(delta, v)
            U = time_evolution_operator(H, T)

            # Calculate the probability of non-adiabatic transition
            probability = abs(U[0, 1]) ** 2
            landscape[i, j] = probability

    return landscape


# Generate values for delta and v
delta_values = np.linspace(-5, 5, 100)
v_values = np.linspace(-5, 5, 100)

# Set a fixed value for T
T = 0.7*np.pi

# Calculate the Landau-Zener control landscape
lz_landscape = landau_zener_control_landscape(delta_values, v_values, T)

# Plot the Landau-Zener control landscape
plt.imshow(lz_landscape, extent=(v_values.min(), v_values.max(), delta_values.min(), delta_values.max()),
           origin='lower', aspect='auto', cmap='jet')
plt.colorbar(label='Transition Probability')
plt.xlabel('v')
plt.ylabel('Delta')
plt.title(f'Quantum Control Landscape for T = {T}')
plt.show()
