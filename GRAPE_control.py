import numpy as np
import matplotlib.pyplot as plt
import datetime
from qutip import Qobj, identity, sigmax, sigmaz,basis
from qutip.qip import hadamard_transform
import qutip.logging_utils as logging
logger = logging.get_logger()
#  Set this to None or logging.WARN for 'quiet' execution
log_level = logging.INFO
#  QuTiP control modules
import qutip.control.pulseoptim as cpo

def run_optimizer(pulse_type):
    example_name = 'Hadamard'
    # Drift Hamiltonian
    H_d = 0.5 * sigmax()
    # The (single) control Hamiltonian
    H_c = 4 * [sigmaz()]
    # start point for the gate evolution
    U_0 = basis(2, 0)  # identity(2)
    # Target for the gate evolution Hadamard gate
    U_targ = basis(2, 1)  # hadamard_transform(1)

    # Number of time slots
    n_ts = 10  # 50
    # Time allowed for the evolution
    evo_time = 2*np.pi  # 10

    # Fidelity error target
    fid_err_targ = 10E-5
    # Maximum iterations for the optimisation algorithm
    max_iter = 10000
    # Maximum (elapsed) time allowed in seconds
    max_wall_time = 120
    # Minimum gradient (sum of gradients squared)
    # as this tends to 0 -> local minima has been found
    min_grad = 1e-20

    # pulse type alternatives: RND|ZERO|LIN|SINE|SQUARE|SAW|TRIANGLE|

    # Set to None to suppress output files
    f_ext = "{}_n_ts{}_ptype{}.txt".format(example_name, n_ts, pulse_type)

    result = cpo.optimize_pulse_unitary(H_d, H_c, U_0, U_targ, n_ts, evo_time,
                                        fid_err_targ=fid_err_targ, min_grad=min_grad,
                                        max_iter=max_iter, max_wall_time=max_wall_time,
                                        init_pulse_type=pulse_type,
                                        log_level=log_level, gen_stats=True)  # out_file_ext=f_ext,

    result.stats.report()
    print("Final evolution\n{}\n".format(result.evo_full_final))
    print("*"*20 + " Summary " + "*"*20)
    print("Final fidelity error: {}".format(result.fid_err))
    print("Final gradient normal: {}".format(result.grad_norm_final))
    print("Terminated due to: {}".format(result.termination_reason))
    print("Number of iterations: {}".format(result.num_iter))
    print("Completed in: {} (HH:MM:SS.US)".format(datetime.timedelta(seconds=result.wall_time)))

    return result, pulse_type


def plot_result(result, p_type):
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(2, 1, 1)
    ax1.set_title("Initial control amps: ("+p_type+")")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Control amplitude")
    ax1.step(result.time, np.hstack((result.initial_amps[:, 0], result.initial_amps[-1, 0])), where='post')

    ax2 = fig1.add_subplot(2, 1, 2)
    ax2.set_title("Optimised Control Sequences")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Control amplitude")
    ax2.step(result.time, np.hstack((result.final_amps[:, 0], result.final_amps[-1, 0])), where='post')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    input_type = ["RND", "SINE", "SQUARE", "SAW", "TRIANGLE", "ZERO", "LIN"]
    for i in range(len(input_type)):
        time_Strt = datetime.datetime.now()
        res, in_type = run_optimizer(input_type[i])
        print("Total time: ", (datetime.datetime.now() - time_Strt).total_seconds())
        plot_result(res, in_type)




