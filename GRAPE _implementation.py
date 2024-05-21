import numpy as np
from scipy.optimize import minimize
from qutip import *
import matplotlib.pyplot as plt

# Define the quantum system parameters
H0 = 10 * sigmaz()  # Hamiltonian for a two-level system
Hc = sigmax()       # Control Hamiltonian

# Define the target unitary operator
target_U = identity(2)

# Define the GRAPE objective function (fidelity)
def grape_objective(pulse_params):
    U = propagator([H0, Hc * pulse_params], 0.1, unitary_mode='series')
    fidelity = np.abs(np.trace(target_U.dag() * U)) / 2.0
    return 1 - fidelity  # Minimize 1 - fidelity

# Initial guess for the control pulse parameters
initial_pulse_params = np.zeros(100)

# Run the GRAPE optimization
result = minimize(grape_objective, initial_pulse_params, method='L-BFGS-B')

# Extract the optimized pulse parameters
optimized_pulse_params = result.x

# Plot the optimized pulse
time = np.linspace(0, 1, len(optimized_pulse_params))
optimized_pulse = np.array([Hc * p for p in optimized_pulse_params])

plt.figure()
plt.plot(time, optimized_pulse_params, label='Optimized Pulse')
plt.xlabel('Time')
plt.ylabel('Control Amplitude')
plt.legend()
plt.title('Optimized Pulse')
plt.show()
