from QTable import QLearningTable
import numpy as np
from env import Maze
import matplotlib.pyplot as plt
import pandas as pd


def run(N):
    env = Maze(N)
    ep_max = 500  # 5000  # 500
    RL = QLearningTable(actions=env.action_space)  # list(range(env.n_actions)))
    fid_10 = 0
    actions_ = []
    for episode in range(ep_max):
        done = False
        observation = env.reset()
        print_once = True
        iter_ = 0
        # fid = 0
        actions_ = []
        print(f"Episode: {episode}, for N={N}")
        while iter_ < N:
            action = RL.choose_action(str(observation))
            actions_.append(action)
            # print(f"\tAction {iter_}: ", action)
            observation_, reward, done, fid = env.step(action, N)
            # print("Observation: ", observation_)
            RL.learn(str(observation), action, reward, str(observation_), print_once)
            print_once = False
            observation = observation_
            if done and len(actions_) == N:
                print(f"\tDone?: {done}, action: {actions_} ")
                # if episode >= ep_max - 11:
                # HOW TO MAKE SURE THE FIDELITY IS FOR THE CORRECT PULSE?????? - solved by using done variable
                # The last value of done(for the current episode) will decide whether to go for next episode or not
                fid_10 = fid  # max(fid_10, fid)
                # break
            iter_ += 1
            print(f"\tEp (iter)-{iter_} fid: {fid}")
        if done:
            break
    # while len(actions_) < N:
    #     actions_.append(0)  # Add a pulse of zero amplitude if the episode terminates early (to have N param Q landscape)
    # print('\t\tFinal_fidelity=', fid_10)
    return fid_10, actions_


def plot_result(fid, N):
    fig, ax = plt.subplots()
    ax.plot(N, fid, linestyle='--', color='#f5be0a', label='dashed', marker='o')
    ax.set_xlabel('N (Number of partitions)')
    ax.set_ylabel('Fidelity')
    ax.set_title('QL Fidelity for different values of N')
    plt.show()


def main(totalTests, N):
    tot_fid = []
    fid_and_pulse = []
    fid_and_pulse_from_PCA_loadings = []
    for z in range(totalTests):
        fid = []
        pulses = {}
        print("Running Test: ", z)
        for i in N:
            print("Calculating fid for N: ", i)
            fidelity, pulses_ = run(i)  # 1 - run(i)
            fid.append(fidelity)
            pulses[i] = pulses_

            fid_pulse = [fidelity]
            fid_pulse.extend(pulses_)
            fid_and_pulse.append(fid_pulse)

            # run Dimensionality reduction using using PCA loadings
            loadings_ = np.load("../../results/2param_PCA_loadings.npy")
            print("Loadings: ", loadings_)
            pulse_proj = np.dot(pulses_, loadings_.T)
            print("Projected pulse:", pulse_proj)
            fid_pulse_loadings = [fidelity]
            fid_pulse_loadings.extend(pulse_proj)
            fid_and_pulse_from_PCA_loadings.append(fid_pulse_loadings)

            print('\t\tFinal_fidelity = ', fidelity)
        tot_fid.append(fid)

    final_fid = np.mean(tot_fid, axis=0)
    print("FID: ", final_fid)
    plot_result(final_fid, N)
    df = pd.DataFrame(fid_and_pulse)
    file_path = "../../results/2param_QL_e_greedy_0p9_with_experience_donotuse.csv"
    df.to_csv(file_path, index=False)

    df2 = pd.DataFrame(fid_and_pulse_from_PCA_loadings)
    file_path2 = "../../results/2param_QL_e_greedy_0p9_from_PCA_loadings_with_experience_donotuse.csv"
    df2.to_csv(file_path2, index=False)


if __name__ == "__main__":
    total_tests = 1000  # 20
    N = [2]  # , 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    main(total_tests, N)
