import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# Define Pauli matrices
sigma_x = sigmax()
sigma_z = sigmaz()


# The Hamiltonian function
def hamiltonian(a, delta):
    # return delta / 2 * sigma_x + a * sigma_z
    return delta / 2 * sigma_x + 4 * a * sigma_z/2


# Time evolution operator
def time_evolution_operator(H, T):
    reduced_plank_constant = 1.0  # 6.582119e-16
    return ((-1j/reduced_plank_constant) * H * T).expm()


# The quantum landscape function
def quantum_landscape(a1, a2, delta, T):
    # Initial and final states
    initial_state = basis(2, 0)
    final_state = basis(2, 1)

    # Calculate the time evolution operators
    H1 = hamiltonian(a1, delta)
    H2 = hamiltonian(a2, delta)
    U1 = time_evolution_operator(H1, T / 2)
    U2 = time_evolution_operator(H2, T / 2)

    # Calculate the quantum landscape
    # result_state_u1 = U1 * initial_state
    # result_state_u2 = final_state.dag() * U2
    # result_state = result_state_u2 * result_state_u1

    # print("Result state: ", np.transpose(final_state) * U2 * result_state_u1)
    # print("Result state abs: ", (final_state.dag() * U2 * result_state_u1).full())

    # Construct the total evolution operator
    evolution_operator = U2 * U1

    # Calculate the overlap between the final and initial states
    landscape = abs((final_state.dag() * evolution_operator * initial_state).full()) ** 2

    # landscape = abs(result_state.full()) ** 2
    # print("Landscape: ", landscape)
    return landscape


def landscape_plot(T, t_min, delta):
    # Generate values for a1 and a2,
    a1_values = np.linspace(-1, 1, 100)
    a2_values = np.linspace(-1, 1, 100)

    plt.figure(figsize=(16, 4))
    # Calculate the quantum landscape values

    for z in range(len(T)):
        J_values = np.zeros((len(a1_values), len(a2_values)))
        # J_values = np.vectorize(quantum_landscape)(a1_values, a2_values, delta, T[z]*t_min)

        # Record the optimal J[E] for different values of a1 and a2
        j_max = [0, 0, 0]  # J[E], a1 ,a2
        for i, a1 in enumerate(a1_values):
            for j, a2 in enumerate(a2_values):
                J_values[i, j] = quantum_landscape(a1, a2, delta, T[z]*t_min)
                if J_values[i, j] > j_max[0]:
                    j_max = [J_values[i, j], a1, a2]

        print("For T/Tmin: {:.2f}, delta: {:.1f}: \n\tMax J[E]:{:.5f} \n\tat a1:{:.5f} and a2: {:.5f}".format(T[z], delta, j_max[0], j_max[1], j_max[2]))
        # Plot the quantum landscape
        plt.subplot(1, len(T), z+1)
        if z == 0:
            plt.ylabel('a2')

        plt.suptitle(f'Quantum Landscape for T = {T} * t_min;  t_min = π/delta; delta={delta} ')
        plt.imshow(J_values, extent=(a2_values.min(), a2_values.max(), a1_values.min(), a1_values.max()),
                   origin='lower', aspect='auto', cmap='jet', vmin=0, vmax=1)
        plt.xlabel('a1')
        plt.title("T/T_min: {:.2f},  Max J(E):{:.4f}".format(T[z], j_max[0]))
    plt.colorbar()
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    # Set fixed values for T and delta
    delta = 1
    t_min = np.pi / delta
    # Result reproduced for T/Tmin:0.6
    T = [2]  # [0.7, 1, 2, 10]  # times t_min
    landscape_plot(T, t_min, delta)



