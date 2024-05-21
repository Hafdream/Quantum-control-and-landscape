import numpy as np
from qutip import *

# Define Pauli matrices
sigma_x = sigmax()
sigma_z = sigmaz()

# Define initial state |psi(0)> = I
initial_state = basis(2, 0)

# Define the Hamiltonian components
H0 = sigma_z
Hc_coeff = 1.0  # Choose a coefficient for the control field
Hc = Hc_coeff * sigma_x

# Define the time-dependent control field amplitude
def omega(t, args):
    return 1.0  # You can modify this function to achieve specific evolution

# Time evolution operator
T = 1.0  # Total evolution time
tlist = np.linspace(0, T, 100)
result = mesolve((H0, [Hc, omega]), initial_state, tlist, [], [sigmax(), sigmay(), sigmaz()])

# Extract the final state
final_state = result.states[-1]

# Display the final state
print("Final State:")
print(final_state)
