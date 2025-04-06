import math
import cmath
import numpy as np
from scipy.linalg import expm

sx = 1 / 2 * np.mat([[0, 1], [1, 0]], dtype=complex)
sy = 1 / 2 * np.mat([[0, -1j], [1j, 0]], dtype=complex)
sz = 1 / 2 * np.mat([[1, 0], [0, -1]], dtype=complex)


def hamiltonian(j):
    H = sx + 4*j*sz
    return H


psi_target = np.mat([[0], [1]], dtype=complex)
psi_0 = np.mat([[1], [0]], dtype=complex)

dt = 2 * np.pi  # / 20  # Values of N added at run time programmatically

def phase2(z):
    """
    return phase angle in [0, 2pi]
    """
    phase = cmath.phase(z)
    if phase < 0:
        phase += 2 * math.pi
    return phase


def get_state_representation(state):
    density_matrix = state * (state.T.conj())
    expectation_value = np.trace((density_matrix * sz))
    return state # expectation_value#.real


# class Maze(object):             # for Python 2
class Maze:
    # qubit in the Bloch Maze
    def __init__(self, N):

        self.counter = None
        self.state = None
        self.N = N
        self.action_space = np.linspace(-1, 1, 100)  # np.linspace(-1, 1, self.N)  # ['0', '1']
        self.n_actions = len(self.action_space)
        self._build_maze()

    def _build_maze(self):
        self.state = psi_0

    def reset(self):
        self.state = psi_0
        self.counter = 0
        # print(dt)
        return get_state_representation(self.state)  # state_to_lattice_point(self.state)

    def step(self, action, N):
        done = False
        U = expm(-1j * hamiltonian(action) * (dt / N))
        self.state = U.dot(self.state)
        self.counter += 1
        s_ = self.state
        # print("STATE: ", s_, s_[0, 0].real)
        xx = s_.H * psi_target  # * psi_0.T
        fidelity = (np.abs(xx[0, 0])) ** 2
        error = 1 - fidelity

        if error < 0.001:  # 10e-3
            reward = 500
            done = True
            s_lattice = 'terminal'
            print(f"\t\t Fidelity_: {fidelity}")
        else:
            reward = -1*(error >= 0.5) + 10*(error < 0.5) + 100*(error < 0.1)
            # reward = 10 * (error < 0.5) + 100 * (error < 0.1)
            # done = (self.counter >= np.pi / (dt / N))  # (self.counter >= 25)  #
            s_lattice = get_state_representation(s_)
        # print("Observation_: ", s_lattice)
        return s_lattice, reward, done, fidelity
