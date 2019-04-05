import random
from ga_struct import GaStruct
import math
import time
from fitness import Fitness
from selection import Selection
from elitism import Elitism
from crossover import CrossOver
import numpy as np


GA_POPSIZE = 2048
GA_MAXITER = 16384
GA_ELITRATE = 0.1
GA_MUTATIONRATE = 0.25
GA_MUTATION = random.randint(0, 32767) * GA_MUTATIONRATE
GA_TARGET = "Hello World!"
GA_AGE = 8
GA_K = 2
CLOCKS_PER_SECOND = 1000


def init_population(population, buffer):
    tsize = len(GA_TARGET)
    for i in range(GA_POPSIZE):
        citizen = GaStruct('', 0)
        for j in range(tsize):
            citizen.str = citizen.str + str(chr((random.randint(0, 32767) % 90) + 32))
        population.append(citizen)
    for i in range(GA_POPSIZE):
        citizen = GaStruct('', 0)
        buffer.append(citizen)


def calc_distance_fitness(population):
    target = GA_TARGET
    tsize = len(target)
    for i in range(GA_POPSIZE):
        fitness = 0
        for j in range(tsize):
            fitness = fitness + abs(int(ord(population[i].str[j]) - ord(target[j])))
        population[i].fitness = fitness


def rws_selection(population, max_weight):
    keys = []
    weights = []
    for i, citizen in enumerate(population):
        keys.append(i)
        weights.append(max_weight - citizen.fitness)
    probs = np.array(weights, dtype=float) / float(max_weight)
    probs = np.array(probs, dtype=float) / float(sum(probs))
    sample_np = np.random.choice(keys, 1, p=probs)
    return sample_np[0]


def calc_bulls_n_cows_fitness(population):
    target = GA_TARGET
    tsize = len(target)
    not_found_penalty = 50
    incorrect_place_penalty = 20
    for i in range(GA_POPSIZE):
        fitness = 0
        for j in range(tsize):
            if population[i].str[j] != target[j]:
                if population[i].str[j] in target:
                    fitness = fitness + incorrect_place_penalty
                else:
                    fitness = fitness + not_found_penalty
        population[i].fitness = fitness


def elitism_by_rate(population, buffer, esize):
    for i in range(esize):
        buffer[i].str = population[i].str
        buffer[i].fitness = population[i].fitness
    return esize


def elitism_by_aging(population, buffer, esize):
    count = 0
    for i in range(esize):
        if population[i].age > GA_AGE:
            buffer[count].str = population[i].str
            buffer[count].fitness = population[i].fitness
            buffer[count].age = population[i].age + 1
            count = count + 1
    return count


def mutate(member):
    tsize = len(GA_TARGET)
    ipos = int(random.randint(0, 32767) % tsize)
    delta = int((random.randint(0, 32767) % 90) + 32)
    string_list = list(member.str)
    string_list[ipos] = str(chr(((ord(string_list[ipos]) + delta) % 122)))
    member.str = ''.join(string_list)


def random_selection():
    i1 = int(random.randint(0, 32767) % (GA_POPSIZE / 2))
    i2 = int(random.randint(0, 32767) % (GA_POPSIZE / 2))
    return i1, i2


def tournament_selection(population):
    k_citizen_entries = []
    for i in range(GA_K):
        entry = int(random.randint(0, 32767) % GA_POPSIZE)
        if entry not in k_citizen_entries:
            k_citizen_entries.append(entry)
    min = math.inf
    best = 0
    for i in range(len(k_citizen_entries)):
        if population[k_citizen_entries[i]].fitness < min:
            min = population[k_citizen_entries[i]].fitness
            best = k_citizen_entries[i]
    return best


def fitness_sum(population):
    sum = 0
    for citizen in population:
        sum = sum + citizen.fitness
    return sum


def reverse_weights(population, max_weight):
    for citizen in population:
        citizen.fitness = abs(max_weight - citizen.fitness)


def select_parents(population, selection_type):
    if selection_type == Selection.RANDOM:
        i1, i2 = random_selection()
        return i1, i2
    elif selection_type == Selection.RWS:
        max_weight = max(population).fitness
        i1 = rws_selection(population, max_weight)
        i2 = rws_selection(population, max_weight)
        return i1, i2
    elif selection_type == Selection.TOURNAMENT:
        i1 = tournament_selection(population)
        i2 = tournament_selection(population)
        return i1, i2


def elitism(population, buffer, esize, elitism_type):
    if elitism_type == Elitism.ELITE_RATE:
        return elitism_by_rate(population, buffer, esize)
    elif elitism_type == Elitism.AGING:
        return elitism_by_aging(population, buffer, esize)


def crossover(population, i1, i2, crossover_type):
    tsize = len(GA_TARGET)
    age = min(population[i1].age, population[i2].age) + 1
    if crossover_type == CrossOver.ONE_POINT:
        spos = int(random.randint(0, 32767) % tsize)
        return GaStruct(population[i1].str[0:spos] + population[i2].str[spos:tsize], 0, age)
    elif crossover_type == CrossOver.TWO_POINT:
        spos1 = int(random.randint(0, 32767) % tsize)
        spos2 = int(random.randint(0, 32767) % tsize)
        if spos1 == spos2:
            if spos2 == 0:
                spos2 = spos2 + 1
            else:
                spos1 = spos1 - 1
        elif spos1 > spos2:
            spos1, spos2 = spos2, spos1

        return GaStruct(population[i1].str[0:spos1] + population[i2].str[spos1:spos2] +
                        population[i1].str[spos2:tsize], 0, age)


def mate(population, buffer, selection_type, elitism_type, crossover_type):
    esize = int(GA_POPSIZE * GA_ELITRATE)
    esize = elitism(population, buffer, esize, elitism_type)
    for i in range(esize, GA_POPSIZE):
        i1, i2 = select_parents(population, selection_type)
        buffer[i] = crossover(population, i1, i2, crossover_type)
        if random.randint(0, 32767) < GA_MUTATION:
            mutate(buffer[i])


def average(population):
    sum = 0
    for i in range(GA_POPSIZE):
        sum = sum + population[i].fitness
    return sum / GA_POPSIZE


def deviation(population):
    avg = average(population)
    sum = 0
    for i in range(GA_POPSIZE):
        sum = sum + ((avg - population[i].fitness) ** 2)
    sum = math.sqrt(sum / GA_POPSIZE)
    return sum


def print_best(gav):
    print("Best: " + gav[0].str + " (" + str(gav[0].fitness) + ")")


def print_data(population, start_time, iter_num):
    print("    Average: " + str(average(population)))
    print("    Deviation: " + str(deviation(population)))
    run_time = time.time() - start_time
    clock_ticks = run_time * CLOCKS_PER_SECOND
    print("    Clock ticks elapsed: " + str(clock_ticks))
    print("    Time elapsed: " + str(run_time))
    print("    Iteration number: " + str(iter_num))


def swap(population, buffer):
    temp = population
    population = buffer
    buffer = temp
    return population, buffer


def calc_fitness(population, fitness_type):
    if fitness_type == Fitness.DISTANCE:
        calc_distance_fitness(population)
    else:
        calc_bulls_n_cows_fitness(population)


def main():
    fitness_type = Fitness.DISTANCE
    selection_type = Selection.RWS
    elitism_type = Elitism.ELITE_RATE
    crossover_type = CrossOver.ONE_POINT
    overall_time = time.time()
    pop_alpha = []
    pop_beta = []
    init_population(pop_alpha, pop_beta)
    population = pop_alpha
    buffer = pop_beta
    iter_num = 0
    for i in range(GA_MAXITER):
        start_time = time.time()
        calc_fitness(population, fitness_type)
        population.sort()
        print_best(population)
        if population[0].fitness == 0:
            break
        mate(population, buffer, selection_type, elitism_type, crossover_type)
        buffer, population = population, buffer
        print_data(population, start_time, i)
        iter_num = i

    overall_time = time.time() - overall_time
    overall_clock_ticks = overall_time * CLOCKS_PER_SECOND
    print("Overall clock ticks: " + str(overall_clock_ticks))
    print("Overall time: " + str(overall_time))
    print("Overall iterations: " + str(iter_num))


main()
