import pandas as pd
import numpy as np
from qutip import *
import matplotlib.pyplot as plt
import math

# from RL_state_preparation.single_qubit.SGD import gradient_descent

# Define Pauli matrices
sigma_x = 0.5 * sigmax()
sigma_z = 0.5 * sigmaz()


def hamiltonian(a):
    # Define the Hamiltonian as a function of the control parameter 'a'
    return sigma_x + 4 * a * sigma_z


def objective_function(seq, T):
    # Define the initial and target states
    initial_state = basis(2, 0)
    target_state = basis(2, 1)
    U = identity(2)

    for i in seq:
        H = hamiltonian(i)  # Hamiltonian --> H[J(t)] = 4J(t)σz + hσx
        U = ((-1j * H * T / len(seq)).expm()) * U
    # Calculate the overlap between the final and initial states
    fidelity_ = abs((target_state.dag() * U * initial_state).full()[0, 0]) ** 2
    return fidelity_


def objective_function_linear_ramp(seq, T):
    """
    hamiltonian(a) = sigma_x + 4 * a * sigma_z * i * T / len(seq)  ; i = 1, 2, 3, 4, ...
    """
    # Define the initial and target states
    initial_state = basis(2, 0)
    target_state = basis(2, 1)
    U = identity(2)
    for i in range(len(seq)):
        H = hamiltonian(seq[i] * (i + 1) * T / len(seq))  # Hamiltonian --> H[J(t)] = 4J(t)σz + hσx; J(t) = i * delta_t
        U = ((-1j * H * T / len(seq)).expm()) * U
    fidelity_ = abs((target_state.dag() * U * initial_state).full()[0, 0]) ** 2
    return fidelity_


def gradient_descent(x, dim, learning_rate, momentum, num_iterations, min_err, dt, interval_p):
    # print("Initial X: ", x)
    velocity = np.zeros_like(x)
    for i in range(num_iterations):
        v = 2 * np.random.uniform(0, 1, dim) - 1
        # v = np.random.rand(dim)
        xp = x + v * interval_p
        xm = x - v * interval_p
        error_derivative = (objective_function(xp, dt) - objective_function(xm, dt)) / (2 * interval_p)

        # momentum
        velocity = momentum * velocity - learning_rate * error_derivative * v
        x = x + velocity
        # x = x - learning_rate * error_derivative * v
        # cost_hist.append(cost(xp))
        if (1 - objective_function(x, dt)) < min_err:
            iteration = i
            print("\t\tIter: ", i)
            # print("\t\tPulse: ", cost(x)[1])
            break

    return objective_function(x, dt)


def landscape_plot(T, t_min):
    # Generate a grid of values for a1, a2, and a3
    a_values = np.linspace(-5, 5, 25)
    # a1_mesh, a2_mesh, a3_mesh = np.meshgrid(a_values, a_values, a_values)

    # rows_ = int(np.sqrt(len(a_values)))
    # cols_ = int(np.ceil(len(a_values) / rows_))

    a3_values = a_values
    T_value = T[2]
    j_max = []  # [J_value, a1, a2, a3]
    for i, a3 in enumerate(a3_values):
        a1_values = a2_values = np.linspace(-5, 5, 100)
        J_values = np.zeros((len(a1_values), len(a2_values)))

        for z, a1 in enumerate(a1_values):
            for j, a2 in enumerate(a2_values):
                J_values[z, j] = objective_function([a1, a2, a3], T_value * t_min)
                if J_values[z, j] > 0.99:  # j_max[0]:
                    j_max.append([J_values[z, j], a1, a2, a3])


def objective_function_linear_ramp_with_initial_value(seq, T, slope, plot_pulse):
    """
    hamiltonian(a) = sigma_x + 4 * sigma_z * (b1 + b2 * i * T/partitions)  ; i = 1, 2, 3, 4, ...
    """
    # Define the initial and target states
    initial_state = basis(2, 0)
    target_state = basis(2, 1)
    U = identity(2)
    pulse_ = []
    a3_values = np.linspace(-1, 1, 50)
    part_ = len(a3_values)
    for i in range(part_):
        pulse_.append(seq[0] + seq[1] * (i + 1) * (T / part_) * slope)
        H = hamiltonian(seq[0] + seq[1] * (i + 1) * (T / part_) * slope)
        U = ((-1j * H * T / part_).expm()) * U
    fidelity_ = abs((target_state.dag() * U * initial_state).full()[0, 0]) ** 2
    if plot_pulse:
        plot_pulse_simple(pulse_, len(pulse_))
    return fidelity_


def objective_function_exp_ramp_3_param(seq, T, slope, plot_pulse):
    """
    hamiltonian(a) = sigma_x + 4 * sigma_z * (b1 + b2 * (T/partitions)^b3)
    """
    # Define the initial and target states
    initial_state = basis(2, 0)
    target_state = basis(2, 1)
    U = identity(2)
    pulse_ = []
    a3_values = np.linspace(-1, 1, 50)
    part_ = len(a3_values)
    for i in range(part_):
        pulse_.append(seq[0] + seq[1] * (((i+1)*(T/part_)) ** seq[2]) * slope)
        H = hamiltonian(seq[0] + seq[1] * (((i+1)*(T/part_)) ** seq[2]) * slope)#(seq[0] + seq[1] * ((T / part_) ** a3_values[i]) * slope)
        U = ((-1j * H * T / part_).expm()) * U
    fidelity_ = abs((target_state.dag() * U * initial_state).full()[0, 0]) ** 2
    # print("Fidelity: ", fidelity_)
    # print("Pulse: ", pulse_)
    if plot_pulse:
        plot_pulse_simple(pulse_, len(pulse_))

    return fidelity_


def objective_function_exp_ramp_4_param(seq, T, slope, plot_pulse):
    """
    hamiltonian(a) = sigma_x + 4 * sigma_z * (...)
    """
    # Define the initial and target states
    initial_state = basis(2, 0)
    target_state = basis(2, 1)
    U = identity(2)
    pulse_ = []
    a3_values = np.linspace(-1, 1, 50)
    part_ = len(a3_values)
    for i in range(part_):  # ((seq[0] * seq[1] + seq[2]) * (T / part_) + seq[3] * math.sin(a3_values[i]) * slope)
        pulse_.append((seq[0] * seq[1] + seq[2]) * (((i+1)*T)/part_) ** seq[3] * slope)  # (seq[0] * seq[1] + (((i+1)*(T/part_)) ** seq[2]) * slope)
        H = hamiltonian((seq[0] * seq[1] + seq[2]) * (((i+1)*T)/part_) ** seq[3] * slope)  # (seq[0] + seq[1] * ((T / part_) ** a3_values[i]) * slope)
        U = ((-1j * H * T / part_).expm()) * U
    fidelity_ = abs((target_state.dag() * U * initial_state).full()[0, 0]) ** 2
    # print("Fidelity: ", fidelity_)
    # print("Pulse: ", pulse_)
    if plot_pulse:
        plot_pulse_simple(pulse_, len(pulse_))

    return fidelity_


def objective_function_sinusoidal_ramp_2_param(seq, T, slope, plot_pulse):
    """
    hamiltonian(a) =
    """
    # Define the initial and target states
    initial_state = basis(2, 0)
    target_state = basis(2, 1)
    U = identity(2)
    pulse_ = []
    a3_values = np.linspace(-5, 5, 50)
    part_ = len(a3_values)
    for i in range(part_):
        pulse_.append((seq[0] + seq[1] * (T / part_) + math.sin(a3_values[i]) * slope))  #(seq[0] + seq[1] * (T / part_) *  math.sin(seq[2] * a3_values[i]) * slope)
        H = hamiltonian(seq[0] + seq[1] * (T / part_) + math.sin(a3_values[i]) * slope)  # (seq[0] + seq[1] * (T / part_) * math.sin(a3_values[i]) * slope)
        U = ((-1j * H * T / part_).expm()) * U
    fidelity_ = abs((target_state.dag() * U * initial_state).full()[0, 0]) ** 2
    # print("Fidelity: ", fidelity_)
    # print("Pulse: ", pulse_)
    if plot_pulse:
        plot_pulse_simple(pulse_, len(pulse_))

    return fidelity_


def objective_function_sinusoidal_ramp_3_param(seq, T, slope, plot_pulse):
    """
    hamiltonian(a) = sigma_x + 4 * sigma_z * (b1 + b2 * (T/partitions)*sin(b3))
    """
    # Define the initial and target states
    initial_state = basis(2, 0)
    target_state = basis(2, 1)
    U = identity(2)
    pulse_ = []
    a3_values = np.linspace(-5, 5, 50)
    part_ = len(a3_values)
    for i in range(part_):
        pulse_.append((seq[0] + seq[1] * (T / part_) + seq[2] * math.sin(a3_values[i]) * slope))  #(seq[0] + seq[1] * (T / part_) *  math.sin(seq[2] * a3_values[i]) * slope)
        H = hamiltonian(seq[0] + seq[1] * (T / part_) + seq[2] * math.sin(a3_values[i]) * slope)  # (seq[0] + seq[1] * (T / part_) * math.sin(a3_values[i]) * slope)
        U = ((-1j * H * T / part_).expm()) * U
    fidelity_ = abs((target_state.dag() * U * initial_state).full()[0, 0]) ** 2
    # print("Fidelity: ", fidelity_)
    # print("Pulse: ", pulse_)
    if plot_pulse:
        plot_pulse_simple(pulse_, len(pulse_))

    return fidelity_


def objective_function_sinusoidal_ramp_4_param(seq, T, slope, plot_pulse):
    """
    hamiltonian(a) = sigma_x + 4 * sigma_z * (...)
    """
    # Define the initial and target states
    initial_state = basis(2, 0)
    target_state = basis(2, 1)
    U = identity(2)
    pulse_ = []
    a3_values = np.linspace(-5, 5, 50)
    part_ = len(a3_values)
    for i in range(part_):
        pulse_.append(((seq[0] * seq[1] + seq[2]) * (T / part_) + seq[3] * math.sin(a3_values[i]) * slope))  #(seq[0] + seq[1] * (T / part_) *  math.sin(seq[2] * a3_values[i]) * slope)
        H = hamiltonian(((seq[0] * seq[1] + seq[2]) * (T / part_) + seq[3] * math.sin(a3_values[i]) * slope))  # (seq[0] + seq[1] * (T / part_) * math.sin(a3_values[i]) * slope)
        U = ((-1j * H * T / part_).expm()) * U
    fidelity_ = abs((target_state.dag() * U * initial_state).full()[0, 0]) ** 2
    print("\tFidelity: ", fidelity_)
    # print("\tPulse: ", pulse_)

    if plot_pulse:
        plot_pulse_simple(pulse_, len(pulse_))

    return fidelity_


def calculate_fid(T, t_min, pulse_type):
    a1_values = np.linspace(-1, 1, 100)
    a2_values = np.linspace(-1, 1, 100)
    a3_values = np.linspace(-1, 1, 100)
    a4_values = np.linspace(-1, 1, 100)
    a5_values = np.linspace(-1, 1, 100)
    # print(a1_values)

    T_value = T[2]
    res = []
    # for x in a5_values:
    #     for z in a4_values:
    #         for j in a3_values:
    #             for i in a2_values:
    #                 for c in a1_values:
    #                     seq = [x, z, j, i, c]
    #                     fid_ = objective_function(seq, T_value*t_min)
    #                     if fid_ >= 0.9999:
    #                         print("Fid: {}, {}".format(fid_, seq))
    #                     res.append([fid_, x, z, j, i, c])
    if pulse_type == "rand":
        max_ = [0, []]
        for j, val1 in enumerate(a1_values):
            for i, val2 in enumerate(a2_values):
                seq = [val1, val2]
                fid_linear = objective_function_linear_ramp_with_initial_value(seq, T_value * t_min, 1, False)
                print("fid: ", fid_linear)
                if fid_linear > max_[0]:
                    max_[0] = fid_linear
                    max_[1] = seq
                if fid_linear >= 0.9999:
                    print("Fid: {}, {}".format(fid_linear, seq))
                res.append([fid_linear, j, i])
        print("max fid: ", max_)

    elif pulse_type == "linear":
        update_val = 0.02020202020202022  # a3_values[1] - a3_values[0]
        print("Update interval: ", update_val)
        slope = 18
        max_ = [0, [float("-inf") * 4]]
        for j in a3_values:
            i = j + update_val * slope  # random.choice([j, a3_values[-1]])  #
            c = i + update_val * slope  # random.choice([i, a3_values[-1]])  #
            d = c + update_val * slope
            e = d + update_val * slope
            seq = [j, i]  # , c, d]
            fid_ = objective_function(seq, T_value * t_min)
            print("fid: ", fid_)
            if fid_ > max_[0]:
                max_[0] = fid_
                max_[1] = [j, i]  # , c, d]
            if fid_ >= 0.99:
                print("Fid: {}, {}".format(fid_, seq))
            res.append([fid_, j, i])  # , c, d])
        print("max fid: ", max_)
    # columns = ["Fidelity", "a2", "a1"]
    df = pd.DataFrame(res)
    file_path = "./results/2_param_fidelity_results_100_t2_linear_ramp_with_initial_value.csv"
    if pulse_type == "linear":
        plot_pulse(res, 2)
    else:
        plot_pulse(res, 50)

    # Writing to CSV file
    df.to_csv(file_path, index=False)
    print(df.head())


def linear_input_landscape(T, t_min):
    a1_values = np.linspace(-1, 1, 50)
    a2_values = np.linspace(-1, 1, 50)
    a3_values = np.linspace(-1, 1, 50)
    a4_values = np.linspace(-1, 1, 50)
    plt.figure(figsize=(16, 4))
    # Calculate the quantum landscape values
    res_data = []
    for z in range(len(T)):
        res = np.zeros((len(a1_values), len(a2_values)))

        j_max = [0, 0, 0]
        for i, a1 in enumerate(a1_values):
            print("Outer loop: ", i)
            for j, a2 in enumerate(a2_values):
                seq = [a1, a2]
                # for x, a3 in enumerate(a3_values):
                #     for zz, a4 in enumerate(a4_values):
                #         seq = [a1, a2]#, a3, a4]
                #         # res[i, j] = objective_function_linear_ramp_with_initial_value(seq, T[z] * t_min, 1, False)
                #
                #         # res[i, j] = objective_function_linear_ramp_3_param(seq, T[z] * t_min, 1, True)
                #         # res[i, j] = objective_function_sinusoidal_ramp_3_param(seq, T[z] * t_min, 1, True)
                #         # fid_3param = objective_function_exp_ramp_3_param(seq, T[z] * t_min, 1, False)
                #         # fid_3param = objective_function_sinusoidal_ramp_3_param(seq, T[z] * t_min, 1, True)
                #         # fid_3param = objective_function_sinusoidal_ramp_4_param(seq, T[z] * t_min, 1, False)
                fid_3param = objective_function_sinusoidal_ramp_2_param(seq, T[z] * t_min, 1, False)
                res[i, j] = fid_3param
                if T[z] == 2:
                    res_data.append([fid_3param, a1, a2])

                # print("Fid: ", res[i,j])
                if res[i, j] > j_max[0]:
                    j_max = [res[i, j], a1, a2]
                    print("max fid: ", res[i, j])

        print(
            "For T/Tmin: {:.2f}, delta: {:.1f}: \n\tMax J[E]:{:.5f} \n\tat a1:{:.5f} and a2: {:.5f}".format(T[z], delta,
                                                                                                            j_max[0],
                                                                                                            j_max[1],
                                                                                                            j_max[2]))
        # Plot the quantum landscape
        plt.subplot(1, len(T), z + 1)
        if z == 0:
            plt.ylabel('a2')
        plt.suptitle(f'Quantum Landscape for T = {T} * t_min;  t_min = π/delta; delta={delta} ')
        plt.imshow(res, extent=(a2_values.min(), a2_values.max(), a1_values.min(), a1_values.max()),
                   origin='lower', aspect='auto', cmap='jet', vmin=0, vmax=1)
        plt.xlabel('a1')
        plt.title("T/T_min: {:.2f},  Max J(E):{:.4f}".format(T[z], j_max[0]))
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    df = pd.DataFrame(res_data)
    file_path = "./results/2_param_fidelity_T2_results_50_sinusoidal.csv"
    # Writing to CSV file
    df.to_csv(file_path, index=False)
    print(df.head())


def linear_input_landscape_plot_with_slope(T, t_min):
    a1_values = np.linspace(-1, 1, 50)
    a2_values = np.linspace(-1, 1, 50)
    slopes = np.linspace(-10, 10, 21)
    rows_ = int(np.ceil(np.sqrt(len(slopes))))
    cols_ = int(np.ceil(len(slopes) / rows_))
    print("Rows: {}, Cols: {}".format(rows_, cols_))
    T_value = T[3]
    j_max_ = [0, 0, 0]
    for z, slope in enumerate(slopes):
        res = np.zeros((len(a1_values), len(a2_values)))
        j_max = [0, 0, 0]
        for i, a1 in enumerate(a1_values):
            for j, a2 in enumerate(a2_values):
                seq = [a1, a2]
                res[i, j] = objective_function_linear_ramp_with_initial_value(seq, T_value * t_min, slope, True)
                # print("Fid: ", res[i,j])
                if res[i, j] > j_max[0]:
                    j_max = [res[i, j], a1, a2]
        print(
            "For T/Tmin: {:.2f}, slope: {:.1f}: \n\tMax J[E]:{:.5f} \n\tat a1:{:.5f} and a2: {:.5f}".format(T[2],
                                                                                                            slope,
                                                                                                            j_max[0],
                                                                                                            j_max[1],
                                                                                                            j_max[2]))

        plt.subplot(rows_, cols_, z + 1)
        plt.imshow(res, extent=(a2_values.min(), a2_values.max(), a1_values.min(), a1_values.max()),
                   origin='lower', aspect='auto', cmap='jet', vmin=0, vmax=1)

        plt.xticks([])
        plt.yticks([])

        plt.title("S:{:.1f}, F:{:.4f}".format(int(slope), j_max[0]))
        if j_max[0] > j_max_[0]:
            j_max_ = j_max
    plt.colorbar()
    plt.suptitle("T/T_min: {:.2f},  Max Fid:{:.4f}, \nS=Slope, F=Fidelity".format(T_value, j_max_[0]))
    plt.tight_layout()
    plt.show()


def plot_pulse(pulse, N):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 4))
    x_values = [i + 1 for i in range(N)]
    fid_val = []
    for z in range(len(pulse)):
        fid_val.append(pulse[z][0])
    for i in range(len(pulse)):
        fid_i = pulse[i][0]
        if fid_i > 0.009:
            pulse_ = pulse[i][1:]
            ax.step(x_values, pulse_, where='mid')
    ax.set_xticks(x_values)
    ax.set_xlabel('time steps')
    ax.set_ylabel('Pulse amplitude')
    plt.show()


def plot_pulse_simple(pulse, N):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
    x_values = [i + 1 for i in range(N)]
    max_val = max(max(x_values), max(pulse))
    ax.step(x_values, pulse, where='mid')
    # ax.set_xticks(x_values)
    # ax.axis('equal')
    # ax.set_xlim(1, max_val)
    # ax.set_ylim(0, max_val)
    ax.set_xlabel('time steps')
    ax.set_ylabel('Pulse amplitude')
    plt.show()


if __name__ == "__main__":
    # Set fixed values for T and delta
    delta = 1
    t_min = np.pi / delta
    T = [2] # [0.7, 1, 2, 10]  # [10, 100, 200, 400]  #   times t_min
    pulse_type = "rand"  # "linear" #
    # calculate_fid(T, t_min, pulse_type)
    linear_input_landscape(T, t_min)
    # linear_input_landscape_plot_with_slope(T, t_min)
