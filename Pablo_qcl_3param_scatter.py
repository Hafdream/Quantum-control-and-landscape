import matplotlib.pyplot as plt
import numpy as np
from qutip import *

# Define Pauli matrices
sigma_x = sigmax()
sigma_z = sigmaz()


def hamiltonian(a, delta):
    # Define the Hamiltonian as a function of the control parameter 'a'
    return 0.5 * delta * sigma_x + a * sigma_z


def objective_function(a1, a2, a3, T, delta):
    # Define the initial and target states
    initial_state = basis(2, 0)
    target_state = basis(2, 1)

    # Construct the total Hamiltonian for each time segment
    H1 = hamiltonian(a1, delta)
    H2 = hamiltonian(a2, delta)
    H3 = hamiltonian(a3, delta)

    # Time evolution operators for each segment
    U1 = (-1j * H1 * T / 3).expm()
    U2 = (-1j * H2 * T / 3).expm()
    U3 = (-1j * H3 * T / 3).expm()

    # Construct the total evolution operator
    evolution_operator = U3 * U2 * U1

    # Calculate the overlap between the final and initial states
    overlap = abs((target_state.dag() * evolution_operator * initial_state).full()) ** 2

    return overlap


def landscape_plot(T, t_min, delta):
    # Generate a grid of values for a1, a2, and a3
    a1_values = np.linspace(0, 5, 50)
    a2_values = np.linspace(0, 5, 50)
    a3_values = np.linspace(0, 5, 25)

    # Create a 3D grid
    a1_mesh, a2_mesh, a3_mesh = np.meshgrid(a1_values, a2_values, a3_values)
    T_value = T[3]
    # Compute the values of J at each (x, y, z) point
    j_values = np.vectorize(objective_function)(a1_mesh, a2_mesh, a3_mesh, T_value * t_min, delta)

    j_max = np.amax(j_values)
    print("J(E) max: {}".format(j_max))
    # Flatten the arrays for the scatter plot
    a1_flat = a1_mesh.flatten()
    a2_flat = a2_mesh.flatten()
    a3_flat = a3_mesh.flatten()
    j_values_flat = j_values.flatten()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(a1_flat, a2_flat, a3_flat, c=j_values_flat, cmap='jet', vmin=0, vmax=1)
    ax.set_xlabel('a1')
    ax.set_ylabel('a2')
    ax.set_zlabel('a3')
    ax.set_title("3D plot for T/t_min={:.2f}, \nMax J(E): {:.5f}".format(T_value, j_max))
    # Add a colorbar
    fig.colorbar(scatter, ax=ax, label='J(E)', orientation='vertical')
    plt.show()


if __name__ == "__main__":
    # Set fixed values for T and delta
    delta = 1
    t_min = np.pi / delta
    T = [0.7, 1, 2, 10]  # times t_min
    landscape_plot(T, t_min, delta)
