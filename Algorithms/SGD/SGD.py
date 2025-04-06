import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
import json
from datetime import datetime
import random
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


def cost(seq):

    dt = 2 * np.pi/len(seq)

    sx = 1 / 2 * np.mat([[0, 1], [1, 0]], dtype=complex)
    sz = 1 / 2 * np.mat([[1, 0], [0, -1]], dtype=complex)
    # print("Sima z: ", sz, "\n Sima X: ", sx)
    U = np.matrix(np.identity(2, dtype=complex))  # initial Evolution operator

    J = 4  # control field strength
    pulse_ = []
    for i in seq:
        H = i * J * sz + 1 * sx  # Hamiltonian --> H[J(t)] = 4J(t)σz + hσx
        U = expm(-1j * H * dt) * U  # Evolution operator
        pulse_.append(i)
    p0 = np.mat([[1], [0]], dtype=complex)  # initial state
    pt = U * p0  # final state

    target = np.mat([[0], [1]], dtype=complex)
    # .H is used to get the conjugate transpose of a numpy matrix when working with complex numbers
    err = 1 - (np.abs(pt.H * target) ** 2).item(0).real  # infidelity (to make it as small as possible)

    return err, pulse_, U


delta = 0.01  # 0.01
cost_hist = []


def gradient_descent(x, dim, learning_rate, momentum, num_iterations, min_err):
    # print("Initial X: ", x)
    velocity = np.zeros_like(x)
    iteration = num_iterations
    for i in range(num_iterations):
        v = 2 * np.random.uniform(0, 1, dim) - 1
        # v = np.random.rand(dim)
        xp = x + v * delta
        xm = x - v * delta
        if xp[0] < -1:
            xp[0] = -1
        if xp[0] > 1:
            xp[0] = 1
        if xm[0] < -1:
            xm[0] = -1
        if xm[0] > 1:
            xm[0] = 1

        if xp[1] < -1:
            xp[1] = -1
        if xp[1] > 1:
            xp[1] = 1
        if xm[1] < -1:
            xm[1] = -1
        if xm[1] > 1:
            xm[1] = 1
        if len(xp) > 2:
            if xp[2] < -1:
                xp[2] = -1
            if xp[2] > 1:
                xp[2] = 1
            if xm[2] < -1:
                xm[2] = -1
            if xm[2] > 1:
                xm[2] = 1
        if len(xp) > 3:
            if xp[3] < -1:
                xp[3] = -1
            if xp[3] > 1:
                xp[3] = 1
            if xm[3] < -1:
                xm[3] = -1
            if xm[3] > 1:
                xm[3] = 1

        error_derivative = (cost(xp)[0] - cost(xm)[0]) / (2 * delta)
        # print("xp, xm", xp, xm,)
        # momentum
        velocity = momentum * velocity - learning_rate * error_derivative * v
        x = x + velocity
        # x = x - learning_rate * error_derivative * v
        # cost_hist.append(cost(xp))
        if x[0] < -1:
            x[0] = -1
        if x[0] > 1:
            x[0] = 1
        if x[1] < -1:
            x[1] = -1
        if x[1] > 1:
            x[1] = 1
        if len(x) > 2:
            if x[2] < -1:
                x[2] = -1
            if x[2] > 1:
                x[2] = 1
        if len(x) > 3:
            if x[3] < -1:
                x[3] = -1
            if x[3] > 1:
                x[3] = 1
        if cost(x)[0] < min_err:
            iteration = i
            print("\t\tIter: ", i)
            # print("\t\tPulse: ", cost(x)[1])
            break

    return cost(x), iteration


def plot_iterations():
    """
    """
    iter = [[2728, 3431, 4152, 5725, 5850, 9533, 8322, 10000, 9288, 8885],
            [541, 178, 516, 431, 442, 673, 714, 842, 856, 1258],
            [1255, 195, 167, 282, 371, 435, 611, 664, 674, 863],
            [1105, 142, 158, 153, 142, 139, 128, 175, 133, 131]]
    N = [2]  # [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    fig, ax = plt.subplots(figsize=(16, 8))

    ax.plot(N, iter[0], linestyle='--', color='#0e42ed', label='[0,1] vanilla SGD', marker='x')
    ax.plot(N, iter[1], linestyle='--', color='#f5be0a', label='[-1,1] vanilla SGD', marker='o')
    ax.plot(N, iter[2], linestyle='--', color='#660a32', label='[0,1] SGD with Momentum', marker='d')
    ax.plot(N, iter[3], linestyle='--', color='#069e2f', label='[-1,1] SGD with Momentum', marker='+')

    ax.set_xlabel('N (Number of partitions)')
    ax.set_ylabel('iterations')
    ax.set_title('SGD iterations for different values of N.')
    ax.legend()
    plt.show()


def plot_result(fid, N, iterations):
    fig, ax = plt.subplots(figsize=(16, 8))
    # ax.plot.rc('font', size=12)
    ax.plot(N, fid, linestyle='--', label='dashed', marker='o')
    for i, (xi, yi) in enumerate(zip(N, fid)):
        ax.text(xi, yi, f'{yi:.3f}', ha='left', va='bottom')
        ax.text(xi, yi-0.01, f'i : {iterations[i]:.0f}', ha='left', va='bottom')
    percentage_of_max = 0.8
    ymin = min(ax.get_ylim()[0], percentage_of_max * max(fid))
    # min_y_value = 0.5
    ax.set_ylim(ymin, ax.get_ylim()[1]*1.05)
    ax.set_xlabel('N (Number of partitions)')
    ax.set_xticks(N)
    ax.set_ylabel('Fidelity')
    ax.set_title('SGD Fidelity for different values of N.')
    ax.text(N[len(N)//2],  1.05 * ymin, 'i -> iterations', ha='center', va='bottom')

    plt.show()


def plot_pulse(pulse, N):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    x_values = [i+1 for i in range(N[0])]
    # x_values2 = [i+1 for i in range(N[1])]
    ax[0].step(x_values, pulse[0], where='mid', color='b')
    ax[0].set_xticks(x_values)
    # ax[1].step(x_values2, pulse[1], where='pre', color='g')
    # ax[1].set_xticks(x_values2)
    ax[0].set_xlabel('time steps')
    ax[0].set_ylabel('Pulse amplitude')
    ax[0].set_title(f'Pulse profile for N={N[0]}')
    # ax[1].set_xlabel('time steps')
    # ax[1].set_ylabel('Pulse amplitude')
    # ax[1].set_title('Pulse amplitudes for N=20')

    plt.show()


def main(totalTests, partitions, save_res, min_err, plotPulse=False):
    tot_fid = []
    iterations = []
    final_pulse = []
    pulse_fid = []
    for z in range(totalTests):
        fid = []
        print("Running Test: ", z)
        iter = []
        for i in partitions:
            print("\tCalculating fid for N: ", i)
            seq = 2 * np.random.uniform(0, 1, i) - 1  # np.random.uniform(-1, 1, i)

            ep_max = 10000
            time_Strt = datetime.now()
            cost_, iter_ = gradient_descent(seq, i, 0.01, 0.949, ep_max, min_err)
            print("\t\tTime cost: ", (datetime.now() - time_Strt).total_seconds())
            fidelity = 1 - cost_[0]
            pulse_ = cost_[1]
            print("Pulse: ", pulse_)

            loadings_ = np.load("../../QuTip/results/3param_PCA_loadings.npy")
            print("Loadings: ", loadings_)
            pulse_proj = np.dot(pulse_, loadings_.T)
            print("Projected pulse:", pulse_proj)

            final_pulse.append(pulse_)

            if True:  # not (pulse_proj[0] > 1 or pulse_proj[1] > 1 or pulse_proj[0] < -1 or pulse_proj[1] < -1):
                fid_p = [fidelity]
                fid_p.extend(pulse_proj)
                pulse_fid.append(fid_p)
            fid.append(fidelity)
            iter.append(iter_)
            print('\t\tFinal_fidelity=', fidelity)
            # print("Uf: \n", cost_[2])
        tot_fid.append(fid)
        iterations.append(iter)
    final_fid = np.mean(tot_fid, axis=0)
    final_iter = np.mean(iterations, axis=0).astype(int)

    # let's extract the final pulse for plotting
    final_pulse_ = final_pulse[-1]
    df = pd.DataFrame(pulse_fid)
    file_path = "../../QuTip/results/3param_PCA_SGD_from_loadings3.csv"
    # df.to_csv(file_path, index=False)

    if save_res:
        with open('../results/SGD_1Q_Iteration_info_[-1,1].json', 'w') as f:
            # Use json.dump to write data to a JSON file
            data = {'N': partitions,
                    'Iter': final_iter.tolist()}
            json.dump(data, f, indent=2)
        with open('../results/SGD_1Q_data.json', 'w') as f:
            # Use json.dump to write data to a JSON file
            data = {'N': partitions,
                    'avg_fid': final_fid.tolist()}
            json.dump(data, f, indent=2)
    # if not plotPulse:
    #     plot_result(final_fid, partitions, final_iter)
    if plotPulse:
        print("Final pulse: ", final_pulse_)
    return final_pulse_


def effect_of_pulse_reordering(seq):
    seq_ = seq[::-1]
    seq_3 = seq.copy()
    random.shuffle(seq_3)
    fid_1, pulse_1, U_1 = cost(seq)
    fid_2, pulse_2, U_2 = cost(seq_)
    fid_3, pulse_3, U_3 = cost(seq_3)
    print("Fidelity for different reordering of pulses")
    print("\tOrg: ", 1 - fid_1, "\t Pulse: ", pulse_1)
    print("\tRev: ", 1 - fid_2, "\t Pulse: ", pulse_2)
    print("\tShu: ", 1 - fid_3, "\t Pulse: ", pulse_3)


if __name__ == "__main__":
    total_tests = 1  # 1000  # 10
    minErr = 10E-5

    final_pulse = []
    plotPulse = False
    if plotPulse:
        N = [[3]]  # [[5], [20]]  # [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        for z in range(len(N)):
            final_pulse_ = main(total_tests, N[z], True, minErr, plotPulse=plotPulse)
            final_pulse.append(final_pulse_)
        plot_pulse(final_pulse, N[0])
    else:
        N_ = [3]  # [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        final_pulse_ = main(total_tests, N_, True, minErr)


