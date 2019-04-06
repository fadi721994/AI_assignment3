import random
from citizen import Citizen
import math
import time
from fitness import Fitness
from selection import Selection
from crossover import CrossOver
import argparse
import sys
import os
import cProfile, pstats, io


GA_POPSIZE = 2048
GA_MAXITER = 16384
GA_ELITRATE = 0.1
GA_MUTATIONRATE = 0.25
GA_MUTATION = random.randint(0, 32767) * GA_MUTATIONRATE
GA_TARGET = "Hello World!"
GA_AGE = 4
GA_K = 2
CLOCKS_PER_SECOND = 1000
AGED_CITIZENS = []
CHOSEN_PAIRS = []

def init_population(population, buffer):
    tsize = len(GA_TARGET)
    for i in range(GA_POPSIZE):
        citizen = Citizen()
        for j in range(tsize):
            citizen.str = citizen.str + str(chr((random.randint(0, 32767) % 90) + 32))
        population.append(citizen)
    for i in range(GA_POPSIZE):
        citizen = Citizen()
        buffer.append(citizen)


def calc_distance_fitness(population):
    target = GA_TARGET
    tsize = len(target)
    for i in range(GA_POPSIZE):
        fitness = 0
        for j in range(tsize):
            fitness = fitness + abs(int(ord(population[i].str[j]) - ord(target[j])))
        population[i].fitness = fitness


def rws_selection(weights, overall_weights):
    f = random.randint(0, overall_weights)
    for i, weight in enumerate(weights):
        f = f - weight
        if f <= 0:
            return i
    return 0


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
    best = 0
    for i in range(GA_K):
        entry = int(random.randint(0, 32767) % GA_POPSIZE)
        if i == 0 or population[entry].fitness < population[best].fitness:
            best = entry
    return best


def fitness_sum(population):
    sum = 0
    for citizen in population:
        sum = sum + citizen.fitness
    return sum


def select_parents(population, selection_type):
    if selection_type == Selection.RANDOM:
        i1, i2 = random_selection()
        return i1, i2
    elif selection_type == Selection.RWS:
        max_weight = population[GA_POPSIZE - 1].fitness
        weights = []
        for i, citizen in enumerate(population):
            weights.append(max_weight - citizen.fitness)
        overall_weights = sum(weights)
        i1 = rws_selection(weights, overall_weights)
        i2 = rws_selection(weights, overall_weights)
        return i1, i2
    elif selection_type == Selection.AGING:
        if len(AGED_CITIZENS) < 2:
            return select_parents(population, Selection.RANDOM)
        else:
            for i in range(5):
                i1 = random.choice(AGED_CITIZENS)
                i2 = random.choice(AGED_CITIZENS)
                if i2 > i1:
                    i1, i2 = i2, i1
                if (i1, i2) not in CHOSEN_PAIRS:
                    CHOSEN_PAIRS.append((i1, i2))
                    return i1, i2
            return select_parents(population, Selection.RANDOM)
    elif selection_type == Selection.TOURNAMENT:
        i1 = tournament_selection(population)
        i2 = tournament_selection(population)
        return i1, i2
    return 0, 1


def elitism(population, buffer, esize):
    for i in range(esize):
        buffer[i].str = population[i].str
        buffer[i].fitness = population[i].fitness
        buffer[i].age = population[i].age + 1
    return esize


def crossover(first_parent, second_parent, crossover_type):
    tsize = len(GA_TARGET)
    if crossover_type == CrossOver.ONE_POINT:
        spos = int(random.randint(0, 32767) % tsize)
        return Citizen(first_parent.str[0:spos] + second_parent.str[spos:tsize])
    elif crossover_type == CrossOver.TWO_POINT:
        spos1 = int(random.randint(0, 32767) % tsize)
        spos2 = int(random.randint(0, 32767) % tsize)
        if spos1 > spos2:
            spos1, spos2 = spos2, spos1
        return Citizen(first_parent.str[0:spos1] + second_parent.str[spos1:spos2] +
                        first_parent.str[spos2:tsize])


def initialize_aging_data(population):
    AGED_CITIZENS.clear()
    for i, citizen in enumerate(population):
        if citizen.age > GA_AGE:
            AGED_CITIZENS.append(i)
    CHOSEN_PAIRS.clear()


def mate(population, buffer, selection_type, crossover_type):
    esize = int(GA_POPSIZE * GA_ELITRATE)
    elitism(population, buffer, esize)
    for i in range(esize, GA_POPSIZE):
        if selection_type == Selection.AGING:
            initialize_aging_data(population)
        i1, i2 = select_parents(population, selection_type)
        buffer[i] = crossover(population[i1], population[i2], crossover_type)
        if random.randint(0, 32767) < GA_MUTATION:
            mutate(buffer[i])


def average(population):
    return fitness_sum(population) / GA_POPSIZE


def deviation(population):
    avg = average(population)
    sum = 0
    for i in range(GA_POPSIZE):
        sum = sum + ((avg - population[i].fitness) ** 2)
    sum = math.sqrt(sum / GA_POPSIZE)
    return sum


def print_best(gav, iter_num):
    print("Best: " + gav[0].str + " (" + str(gav[0].fitness) + ")")
    with open("output.txt", 'a') as file:
        file.write("Best: " + gav[0].str + " (" + str(gav[0].fitness) + ")\n")
        file.write("    Iteration number: " + str(iter_num) + "\n")
        file.write("    Fitness average: " + str(round(average(gav), 3)) + "\n")
        file.write("    Fitness deviation: " + str(round(deviation(gav), 3)) + "\n")


def print_data(start_time):
    run_time = time.clock() - start_time
    clock_ticks = run_time * CLOCKS_PER_SECOND
    with open("output.txt", 'a') as file:
        file.write("    Clock ticks elapsed: " + str(round(clock_ticks, 3)) + "\n")
        file.write("    Time elapsed: " + str(round(run_time, 3)) + "\n")


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


def delete_files():
    if os.path.isfile("./output.txt"):
        os.remove("./output.txt")


def main():
    delete_files()
    fitness_type, selection_type, crossover_type = parse_cmd()
    overall_time = time.clock()
    population = []
    buffer = []
    init_population(population, buffer)
    iter_num = 0
    for i in range(GA_MAXITER):
        iter_num = i
        start_time = time.clock()
        calc_fitness(population, fitness_type)
        population.sort()
        print_best(population, i)
        if population[0].fitness == 0:
            print_data(start_time)
            break
        mate(population, buffer, selection_type, crossover_type)
        buffer, population = population, buffer
        print_data(start_time)

    overall_time = time.clock() - overall_time
    overall_clock_ticks = overall_time * CLOCKS_PER_SECOND
    with open("output.txt", 'a') as file:
        file.write("Overall clock ticks: " + str(overall_clock_ticks) + "\n")
        file.write("Overall time: " + str(overall_time) + "\n")
        file.write("Overall iterations: " + str(iter_num) + "\n")


def parse_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument('-F', default=0, help='Fitness function: 0 for ASCII distance from target string, '
                                              '1 for bulls and cows.')
    parser.add_argument('-S', default=0, help='Selection type: 0 for Random from the top half, 1 for RWS, '
                                              '2 for aging, 3 for tournament.')
    parser.add_argument('-C', default=0, help='Crossover type: 0 for one-point crossover, 1 for two-point crossover.')
    args = parser.parse_args()
    try:
        fitness_function = int(args.F)
        if fitness_function != 0 and fitness_function != 1:
            print("Fitness function can only be 0 or 1.")
            sys.exit()
    except ValueError:
        print("Fitness function can only be 0 or 1.")
        sys.exit()
    try:
        selection_type = int(args.S)
        if selection_type != 0 and selection_type != 1 and selection_type != 2 and selection_type != 3:
            print("Selection type can only be 0, 1, 2 or 3.")
            sys.exit()
    except ValueError:
        print("Selection type can only be 0, 1, 2 or 3.")
        sys.exit()
    try:
        crossover_type = int(args.C)
        if crossover_type != 0 and crossover_type != 1:
            print("Crossover type can only be 0 or 1.")
            sys.exit()
    except ValueError:
        print("Crossover type can only be 0 or 1.")
        sys.exit()
    return Fitness(fitness_function), Selection(selection_type), CrossOver(crossover_type)


# pr = cProfile.Profile()
# pr.enable()
main()
# pr.disable()
# s = io.StringIO()
# sortby = 'cumulative'
# ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
# ps.print_stats()
# print(s.getvalue())
