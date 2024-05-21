import random

import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from mpl_toolkits.mplot3d import Axes3D

# Define Pauli matrices
sigma_x = sigmax()
sigma_z = sigmaz()


def hamiltonian(a, delta):
    # Define the Hamiltonian as a function of the control parameter 'a'
    # return 0.5 * delta * sigma_x + a * sigma_z
    sigma_x = sigmax() / 2
    sigma_z = sigmaz() / 2
    return sigma_x + 4 * a * sigma_z


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
    overlap = abs((target_state.dag() * evolution_operator * initial_state).full()[0, 0]) ** 2

    return overlap


def landscape_plot(T, t_min, delta):
    # Generate a grid of values for a1, a2, and a3
    a_values = np.linspace(-1, 1, 100)
    # a1_mesh, a2_mesh, a3_mesh = np.meshgrid(a_values, a_values, a_values)

    rows_ = int(np.sqrt(len(a_values)))
    cols_ = int(np.ceil(len(a_values) / rows_))

    a3_values = a_values
    T_value = T[2]
    j_max = [0, 0, 0, 0]  # [J_value, a1, a2, a3]
    for i, a3 in enumerate(a3_values):
        # Calculate the objective function values for each combination of a1, a2, and a3
        # a3 = np.random.choice(a3_values)
        a1_values = a2_values = np.linspace(-1, 1, 100)
        J_values = np.zeros((len(a1_values), len(a2_values)))

        # J_values = np.vectorize(objective_function)(a1_mesh, a2_mesh, a3_mesh, T[2]*t_min, delta)
        for z, a1 in enumerate(a1_values):
            for j, a2 in enumerate(a2_values):
                J_values[z, j] = objective_function(a1, a2, a3, T_value * t_min, delta)
                if J_values[z, j] > j_max[0]:
                    j_max[0] = J_values[z, j]
                    j_max[1] = a1
                    j_max[2] = a2
                    j_max[3] = a3
        # contour = axs[i].contourf(a1_mesh[:, :, 0], a2_mesh[:, :, 0], J_values[:, :, 0], cmap='jet')
        plt.subplot(rows_, cols_, i + 1)
        plt.imshow(J_values, extent=(a2_values.min(), a2_values.max(), a1_values.min(), a1_values.max()),
                   origin='lower', aspect='auto', cmap='jet', vmin=0, vmax=1)

        # plt.xlabel('a1')
        # plt.ylabel('a2')
        plt.xticks([])
        plt.yticks([])

        plt.title("a3 = {:.1f}".format(a3))
        # if i >= 3:
        #     break
        # i += 1
    plt.colorbar()
    plt.suptitle("T/t_min={:.2f} \n Max J(E):{:.5f}".format(T_value, j_max[0]))
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Set fixed values for T and delta
    delta = 1
    t_min = np.pi / delta
    T = [0.7, 1, 2, 10]  # times t_min
    landscape_plot(T, t_min, delta)
