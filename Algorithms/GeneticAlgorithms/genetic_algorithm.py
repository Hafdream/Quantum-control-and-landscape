import random
import numpy as np
from qutip import *
import pandas as pd


MAX_GENERATION = 50  # Maximum number of generations
num_partitions = 4  # [2,3,4] --> number of time steps the control pulse is divided into
MUT_RATE = 0.3  #
TARGET = 0.999  # Fidelity target value, if this value is reached exit optimization loop
num_tests = 1000
GENES = np.linspace(-1, 1, 100)
POP_SIZE = len(GENES) ** num_partitions

sigma_x = 0.5 * sigmax()
sigma_z = 0.5 * sigmaz()


def initialize_pop():
    population = list()
    tar_len = num_partitions

    for i in range(POP_SIZE):
        temp = list()
        for j in range(tar_len):
            temp.append(random.choice(GENES))
        population.append(temp)

    return population


def crossover(selected_chromo, CHROMO_LEN, population):
    offspring_cross = []
    for i in range(int(POP_SIZE)//2):
        parent1 = random.choice(selected_chromo)
        parent2 = random.choice(selected_chromo)  # random.choice(population[:int(POP_SIZE)])

        p1 = parent1[0]
        p2 = parent2[0]

        crossover_point = random.randint(1, CHROMO_LEN - 1)
        child = p1[:crossover_point] + p2[crossover_point:]
        offspring_cross.extend([child])
    return offspring_cross


def mutate(offspring, MUT_RATE):
    mutated_offspring = []

    for arr in offspring:
        for i in range(len(arr)):
            if random.random() < MUT_RATE:
                arr[i] = random.choice(GENES)
        mutated_offspring.append(arr)
    return mutated_offspring


def selection(population):
    sorted_chromo_pop = sorted(population, key=lambda x: x[1])
    sorted_chromo_pop = sorted_chromo_pop[::-1]
    first_30 = sorted_chromo_pop[:int(0.3 * POP_SIZE)]
    last_20 = sorted_chromo_pop[int(0.8 * POP_SIZE):]
    selected_pop = first_30 + last_20

    return selected_pop

def hamiltonian(a):
    # Define the Hamiltonian as a function of the control parameter 'a'
    return sigma_x + 4 * a * sigma_z

def fitness_fid_cal(chromo_from_pop, T):
    # Define the initial and target states
    initial_state = basis(2, 0)
    target_state = basis(2, 1)
    U = identity(2)

    for i in chromo_from_pop:
        H = hamiltonian(i)  # Hamiltonian --> H[J(t)] = 4J(t)σz + hσx
        U = ((-1j * H * T / len(chromo_from_pop)).expm()) * U

    fidelity_ = abs((target_state.dag() * U * initial_state).full()[0, 0]) ** 2

    return [chromo_from_pop, fidelity_]

def replace(new_gen, population):
    new_gen.extend(population)
    temp_sorted = sorted(new_gen, key=lambda x: x[1])
    temp_sorted = temp_sorted[::-1]

    return temp_sorted[:len(population)]

def main(MUT_RATE, TARGET):
    initial_population = initialize_pop()

    loadings_ = np.load("../../results_old/4param_PCA_loadings.npy")
    found_count = 0
    fid_and_pulse = []
    fid_and_pulse_from_PCA_loadings = []
    while found_count < num_tests:  # not found:
        population = []
        generation = 1
        for _ in range(len(initial_population)):
            population.append(fitness_fid_cal(initial_population[_], 2 * np.pi))

        while True:
            selected = selection(population)
            cross_overed = crossover(selected, num_partitions, population)
            mutated = mutate(cross_overed, MUT_RATE)
            new_gen = []
            for _ in mutated:
                new_gen.append(fitness_fid_cal(_, 2*np.pi))
            population = replace(new_gen, selected)

            if population[0][1] >= TARGET:
                print('Target Found | Sequence: ' + str(population[0][0]) +
                      ' Generation: ' + str(generation) + ' Fitness/Fidelity: ' +
                      str(population[0][1]))

                fid_pulse_ = [population[0][1]]
                fid_pulse_.extend(population[0][0])
                fid_and_pulse.append(fid_pulse_)

                pulse_proj = np.dot(population[0][0], loadings_.T)
                fid_pulse_loadings = [population[0][1]]
                fid_pulse_loadings.extend(pulse_proj)
                fid_and_pulse_from_PCA_loadings.append(fid_pulse_loadings)
                break
            print('\tSequence: ' + str(population[0][0]) + ' Generation: ' +
                  str(generation) + ' Fitness/fidelity: ' + str(population[0][1]))
            generation += 1
            if generation > MAX_GENERATION:
                fid_pulse_ = [population[0][1]]
                fid_pulse_.extend(population[0][0])
                fid_and_pulse.append(fid_pulse_)
                # print("Loadings: ", loadings_)
                pulse_proj = np.dot(population[0][0], loadings_.T)
                fid_pulse_loadings = [population[0][1]]
                fid_pulse_loadings.extend(pulse_proj)
                fid_and_pulse_from_PCA_loadings.append(fid_pulse_loadings)
                break
        found_count = len(fid_and_pulse_from_PCA_loadings)  # += 1
        print(f"Iter: {found_count}")
    df = pd.DataFrame(fid_and_pulse)
    file_path = "../../results_old/4param_ga_20240922.csv"
    df.to_csv(file_path, index=False)

    df2 = pd.DataFrame(fid_and_pulse_from_PCA_loadings)
    file_path2 = "../../results_old/4param_ga_pca_loadings_20240922.csv"
    df2.to_csv(file_path2, index=False)


if __name__ == "__main__":
    main(MUT_RATE, TARGET)
