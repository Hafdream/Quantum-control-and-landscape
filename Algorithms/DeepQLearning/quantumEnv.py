import gymnasium as gym
import numpy as np
from scipy.linalg import expm


sx = 1/2 * np.mat([[0, 1], [1, 0]], dtype=complex)
sz = 1/2 * np.mat([[1, 0], [0, -1]], dtype=complex)


def hamiltonian(a):
    J = 4  # control field strength
    H = J * a * sz + 1 * sx
    return H


class QuantumControlEnv(gym.Env):
    def __init__(self, num_actions, num_partitions, max_infidelity, is_train=True):
        super(gym.Env, self).__init__()
        # self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.info = None
        self.n_actions = num_actions
        self.action_space = gym.spaces.Discrete(self.n_actions)  # np.linspace(-1, 1, num_partitions)  #
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

        self.num_partitions = num_partitions
        self.state = np.array([1, 0, 0, 0])
        self.nstep = 0
        self.dt = (2 * np.pi) / self.num_partitions
        self.max_infidelity = max_infidelity
        self.action_rand = {}  # 0
        self.tmp_buffer = []
        self.action_list = []
        self.is_train = is_train

    def reset(self, **kwargs):
        # Reset the environment to an initial state
        self.state = np.array([1, 0, 0, 0], dtype=np.float32)
        self.nstep = 0
        self.info = {"steps": self.nstep, "fidelity": 0.0}
        return (self.state, self.info)

    def step(self, action):
        pulse_amplitude = self.map_action_to_pulse_amplitudes(action)
        self.action_rand[pulse_amplitude] = 1 + self.action_rand.get(pulse_amplitude, 0)
        if self.action_rand[pulse_amplitude] >= 100 * self.num_partitions and self.nstep == 0 and self.is_train:
            action = self.action_space.sample()
            pulse_amplitude = self.map_action_to_pulse_amplitudes(action)
            self.action_rand[pulse_amplitude] = 0
            print(f"ACTION RAND:{self.map_action_to_pulse_amplitudes(action)}, {self.action_rand[pulse_amplitude]}")
        # pulse_amplitude = self.map_action_to_pulse_amplitudes(action)

        psi = np.array(
            [self.state[0:int(len(self.state) / 2)] + self.state[int(len(self.state) / 2):int(len(self.state))] * 1j])
        psi = psi.T
        psi = np.mat(psi)
        target = np.mat([[0], [1]], dtype=complex)

        H = hamiltonian(pulse_amplitude)
        U = expm(-1j * H * self.dt)
        psi = U * psi  # final state

        err = 1 - (np.abs(psi.H * target) ** 2).item(0).real
        # rwd = 1 * (err < 0.5) + 10 * (err < 0.15) + 5000 * (err < self.max_infidelity)
        self.nstep += 1

        # self.action_rand += 1

        if self.nstep < self.num_partitions:
            self.action_list.append(pulse_amplitude)
        if err < self.max_infidelity and self.action_list and (self.action_list[0] not in self.tmp_buffer) and (
                self.nstep == self.num_partitions):
            rwd = 1 * (err < 0.5) + 10 * (err < 0.15) + 500 * (err < self.max_infidelity) + 5000 * (
                        err < self.max_infidelity)
            self.tmp_buffer.append(self.action_list[0])
            print(f"MAX RWD: {rwd}")
        elif self.action_rand[pulse_amplitude] >= (100 * self.num_partitions - self.num_partitions) and self.nstep == 1 and self.action_list and (self.action_list[0] in self.tmp_buffer):
            rwd = 1
            print(f"MIN RWD: {rwd}")
        else:
            rwd = 1 * (err < 0.5) + 10 * (err < 0.15) + 500 * (err < self.max_infidelity)
        if self.nstep == self.num_partitions:
            self.action_list = []
        if len(self.tmp_buffer) > 200000:
            self.tmp_buffer = self.tmp_buffer[10:]

        truncate = (self.nstep >= self.num_partitions)  # self.num_partitions)  # max(10, self.num_partitions))
        terminate = (err < self.max_infidelity) and truncate

        psi = np.array(psi)
        psi_T = psi.T
        self.state = np.array(psi_T.real.tolist()[0] + psi_T.imag.tolist()[0], dtype=np.float32)
        self.info = {"step": self.nstep, "fidelity": 1 - err, "reward":rwd}
        if self.nstep == 1:
            print(f"ACTION:{pulse_amplitude}, State: {self.state}, ACTION_RAND:{self.action_rand[pulse_amplitude]}")
        print(f"\tInfo: {self.info}")
        return self.state, rwd, terminate, truncate, self.info

    def map_action_to_pulse_amplitudes(self, action):
        # Map discrete actions from discrete actions to pulse amplitudes
        min_amplitude = -1.0
        max_amplitude = 1.0
        action_range = max_amplitude - min_amplitude
        action_value = min_amplitude + (action / (self.n_actions - 1)) * action_range
        return action_value


gym.envs.register(
    id='QuantumControl-v0',
    entry_point='quantumEnv:QuantumControlEnv',
)
