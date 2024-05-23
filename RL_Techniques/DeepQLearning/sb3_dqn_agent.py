from quantumEnv import QuantumControlEnv
import gymnasium as gym
import stable_baselines3 as sb3
from stable_baselines3.common.env_checker import check_env
import numpy as np
from qutip import *
from scipy.linalg import expm
import pandas as pd


def hamiltonian(a):
    # Define the Hamiltonian as a function of the control parameter 'a'
    sigma_x = 0.5 * sigmax()
    sigma_z = 0.5 * sigmaz()
    return sigma_x + 4 * a * sigma_z


def objective_function_test(seq, T, steps):
    # Define the initial and target states
    initial_state = basis(2, 0)
    target_state = basis(2, 1)
    U = identity(2)

    for i in seq:
        H = hamiltonian(i)  # Hamiltonian --> H[J(t)] = 4J(t)σz + hσx
        # U = ((-1j * H * T / len(seq)).expm()) * U
        U = ((-1j * H * T / steps).expm()) * U
    # Calculate the overlap between the final and initial states
    fidelity_ = abs((target_state.dag() * U * initial_state).full()[0, 0]) ** 2

    # init_st = np.mat([[1], [0]], dtype=complex)
    # target = np.mat([[0], [1]], dtype=complex)
    #
    # U2 = expm(-1j * hamiltonian(seq[0])*(2*np.pi/len(seq))) * init_st
    # # print("Fiiiiid1111:", U2, (np.abs(U2.H * target) ** 2).item(0).real)
    # U3 = expm(-1j * hamiltonian(seq[1])*(2*np.pi/len(seq))) * U2
    # # print("Fiiiiid2222:",  U3, (np.abs(U3.H * target) ** 2).item(0).real)
    # U4 = expm(-1j * hamiltonian(seq[2]) * (2 * np.pi / len(seq))) * U3
    # # print("Fiiiiid3333:", U4, (np.abs(U4.H * target) ** 2).item(0).real)
    #
    # fid_ = (np.abs(U4.H * target) ** 2).item(0).real
    fid_ = 0.0
    return fidelity_, fid_


def simple_check():
    done = False
    while not done:
        # step_count = 0
        # observation = env.reset()
        actions_ = []
        # while step_count <= 3:
        action = env.action_space.sample()
        observation, reward, fid, done, info = env.step(action)
        print(f"\nAction: {action}")
        print(f"Obs: {observation, reward, fid, done, info}")
        step_count = info["step"]
        actions_.append(action[0])
    # if done:
    #     print(actions_)


def train_agent(env, total_time_steps, lr, exploration_fraction, weights_path, verbose):
    check_env(env, True, True)
    custom_net_arch = [64, 512, 256]  # [64, 64, 64]  #
    agent = sb3.DQN('MlpPolicy', env, policy_kwargs=dict(net_arch=custom_net_arch), train_freq=(2, "episode"),
                    gamma=0.005, verbose=verbose, learning_rate=lr, exploration_fraction=exploration_fraction)

    print("NET: ", agent.q_net)
    print("TARGET NET: ", agent.q_net_target)

    agent.learn(total_timesteps=total_time_steps)
    mean_reward, std_reward = sb3.common.evaluation.evaluate_policy(agent, env, render=False, n_eval_episodes=10)
    print("Mean Reward: {} +/- {}".format(mean_reward, std_reward))
    agent.save(weights_path)


def run_agent(env, episode_max, weights_path):
    obs, _ = env.reset()
    agent = sb3.dqn.DQN.load(weights_path)
    z = 0
    tot_fid = []
    fid_pulse = []
    fid_and_pulse_from_PCA_loadings = []
    while z < episode_max:
        z += 1
        terminate = False
        truncate = False
        obs, _ = env.reset()
        actions_ = []
        i = 0
        while not (terminate or truncate):
            if i == 0:
                action = env.action_space.sample()
            else:
                action, _ = agent.predict(obs, deterministic=False)
            i += 1
            actions_.append(env.map_action_to_pulse_amplitudes(action))
            obs, reward, terminate, truncate, info = env.step(action)
            # print(f"\tOBS:{list(obs), reward, terminate, truncate}, STEP:{info['step']}, FID: {info['fidelity']}")
            # steps = info['step']

        fid, fid2 = objective_function_test(actions_, 2*np.pi, env.num_partitions)
        fid_p = [fid]
        fid_p.extend(actions_)
        fid_pulse.append(fid_p)

        # run Dimensionality reduction using using PCA loadings
        loadings_ = np.load("../../results/4param_PCA_loadings.npy")
        # print("Loadings: ", loadings_)
        pulse_proj = np.dot(actions_, loadings_.T)
        # print("Projected pulse:", pulse_proj)
        fid_pulse_loadings = [fid]
        fid_pulse_loadings.extend(pulse_proj)
        fid_and_pulse_from_PCA_loadings.append(fid_pulse_loadings)

        print(f"Fid calc: {fid}, fid returned:{info['fidelity']}, for action: {actions_} \n-----------------------")
        tot_fid.append(info['fidelity'])
    Average_fid = sum(tot_fid)/len(tot_fid)

    print(f"\tAverage_fid:{Average_fid}, Pulse:{actions_}")
    df = pd.DataFrame(fid_pulse)
    file_path = "../../results/4param_DQN_2.csv"
    # df.to_csv(file_path, index=False)

    df2 = pd.DataFrame(fid_and_pulse_from_PCA_loadings)
    file_path2 = "../../results/4param_dqn_pca_loadings2.csv"
    df2.to_csv(file_path2, index=False)


if __name__ == "__main__":
    for zz in [4]:  # [2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:

        weights_path = "./dqn_agent_4param_2.zip"
        is_train = True  # False
        if is_train:
            env = gym.make("QuantumControl-v0", num_actions=100, num_partitions=zz, max_infidelity=0.005, is_train=True)
            train_agent(env, total_time_steps=5000000, lr=0.0001, exploration_fraction=0.25, weights_path=weights_path, verbose=1)
        else:
            env = gym.make("QuantumControl-v0", num_actions=100, num_partitions=zz, max_infidelity=0.005, is_train=False)
        print(f"\n--------------- TRAINING COMPLETE! -----------------\n")
        print(f"Running test for {zz} partitions")
        run_agent(env, episode_max=1000, weights_path=weights_path)
