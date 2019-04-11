import random
from citizen import Citizen
import math
import time
from fitness import Fitness
from selection import Selection
from crossover import CrossOver
from algorithm import Algorithm
from genetic_algorithm import GeneticAlgorithm
from nqueens_mutation import NQueensMutation
from nqueens_crossover import NQueensCrossover
import copy
import argparse
import sys
import os
import cProfile, pstats, io


print_avgs = []
GA_POPSIZE = 2048
GA_MAXITER = 16384
GA_ELITRATE = 0.1
GA_MUTATIONRATE = 0.25
GA_MUTATION = random.randint(0, 32767) * GA_MUTATIONRATE
GA_TARGET = "Hello World!"
GA_AGE = 5
GA_K = 2
CLOCKS_PER_SECOND = 1000
AGED_CITIZENS = []
CHOSEN_PAIRS = []
ALGORITHM = Algorithm.GENETIC
GENETIC_ALGORITHM_TYPE = GeneticAlgorithm.StringSearch
QUEENS_NUM = 8
MUTATION_TYPE = NQueensMutation.EXCHANGE
CROSSOVER_TYPE = NQueensCrossover.PMX
ITEMS = 10
CAPACITY = 165
WEIGHTS = [23, 31, 29, 44, 53, 38, 63, 85, 89, 82]
VALUE = [92, 57, 49, 68, 60, 43, 67, 84, 87, 72]
OVERALL_WEIGHT = 0


def init_population(population, buffer):
    global GENETIC_ALGORITHM_TYPE
    tsize = len(GA_TARGET)
    for i in range(GA_POPSIZE):
        citizen = Citizen()
        if GENETIC_ALGORITHM_TYPE == GeneticAlgorithm.StringSearch:
            for j in range(tsize):
                citizen.str = citizen.str + str(chr((random.randint(0, 32767) % 90) + 32))
        elif GENETIC_ALGORITHM_TYPE == GeneticAlgorithm.Knapsack:
            global ITEMS, WEIGHTS, CAPACITY
            overall = 0
            for j in range(ITEMS):
                choose = (random.randint(0, 32767) % 2)
                if choose == 1:
                    citizen.knapsack.append(1)
                else:
                    citizen.knapsack.append(0)
            validate_and_update_sack(citizen)
        else:
            global QUEENS_NUM
            # Create entries
            for j in range(QUEENS_NUM):
                citizen.board.append(j)
            # Randomize entries
            for j in range(QUEENS_NUM):
                k = (random.randint(0, 32767) % QUEENS_NUM)
                citizen.board[j] = k
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


def exchange_mutation(citizen):
    global QUEENS_NUM
    col_1 = int(random.randint(0, 32767) % QUEENS_NUM)
    col_2 = int(random.randint(0, 32767) % QUEENS_NUM)
    while col_1 == col_2:
        col_2 = int(random.randint(0, 32767) % QUEENS_NUM)
    citizen.board[col_1], citizen.board[col_2] = citizen.board[col_2], citizen.board[col_1]


def simple_inversion_mutation(citizen):
    global QUEENS_NUM
    col_1 = int(random.randint(0, 32767) % QUEENS_NUM)
    col_2 = int(random.randint(0, 32767) % QUEENS_NUM)
    while col_1 == col_2:
        col_2 = int(random.randint(0, 32767) % QUEENS_NUM)
    for_range = int(abs(col_2 - col_1) / 2) + 1
    if col_1 > col_2:
        col_1, col_2 = col_2, col_1
    for i in range(for_range):
        citizen.board[col_1+i], citizen.board[col_2-i] = citizen.board[col_2-i], citizen.board[col_1+i]


def validate_and_update_sack(citizen):
    global WEIGHTS, ITEMS, CAPACITY
    overall_weight = 0
    for i in range(ITEMS):
        if citizen.knapsack[i] == 1:
            overall_weight = overall_weight + WEIGHTS[i]
    while overall_weight > CAPACITY:
        chosen_items = []
        for i in range(ITEMS):
            if citizen.knapsack[i] == 1:
                chosen_items.append(i)
        min_weight = math.inf
        min_weight_index = 0
        for item in chosen_items:
            if WEIGHTS[item] < min_weight:
                min_weight_index = item
                min_weight = WEIGHTS[item]
        overall_weight = overall_weight - WEIGHTS[min_weight_index]
        citizen.knapsack[min_weight_index] = 0
    citizen.capacity = overall_weight


def knapsack_exchange_mutation(citizen):
    global ITEMS
    item1 = int(random.randint(0, 32767) % ITEMS)
    item2 = int(random.randint(0, 32767) % ITEMS)
    while item1 == item2:
        item2 = int(random.randint(0, 32767) % ITEMS)
    citizen.knapsack[item1], citizen.knapsack[item2] = citizen.knapsack[item2], citizen.knapsack[item1]
    validate_and_update_sack(citizen)


def mutate(citizen):
    global GENETIC_ALGORITHM_TYPE
    if GENETIC_ALGORITHM_TYPE == GeneticAlgorithm.NQueens:
        global MUTATION_TYPE
        if MUTATION_TYPE == NQueensMutation.EXCHANGE:
            exchange_mutation(citizen)
        else:
            simple_inversion_mutation(citizen)
    elif GENETIC_ALGORITHM_TYPE == GeneticAlgorithm.Knapsack:
        knapsack_exchange_mutation(citizen)
    else:
        tsize = len(GA_TARGET)
        ipos = int(random.randint(0, 32767) % tsize)
        delta = int((random.randint(0, 32767) % 90) + 32)
        string_list = list(citizen.str)
        string_list[ipos] = str(chr(((ord(string_list[ipos]) + delta) % 122)))
        citizen.str = ''.join(string_list)


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
        for citizen in population:
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
        buffer[i].board = population[i].board
        buffer[i].knapsack = population[i].knapsack
        buffer[i].capacity = population[i].capacity
        buffer[i].age = population[i].age + 1
    return esize


def ox_crossover(first_parent, second_parent):
    global QUEENS_NUM
    chosen_entries = 0
    citizen = Citizen()
    first_parent_entries = []
    first_parent_chosen_values = []
    for i in range(QUEENS_NUM):
        citizen.board.append(-1)
    while chosen_entries != QUEENS_NUM / 2:
        entry = int(random.randint(0, 32767) % QUEENS_NUM)
        if entry not in first_parent_entries:
            first_parent_entries.append(entry)
            first_parent_chosen_values.append(first_parent.board[entry])
            chosen_entries = chosen_entries + 1
            citizen.board[entry] = first_parent.board[entry]
    second_parent_chosen_values = []
    for i in range(QUEENS_NUM):
        if citizen.board[i] == -1:
            for j in range(QUEENS_NUM):
                second_parent_value = second_parent.board[j]
                if second_parent_value not in first_parent_chosen_values:
                    if second_parent_value not in second_parent_chosen_values:
                        citizen.board[i] = second_parent_value
                        second_parent_chosen_values.append(second_parent_value)
                        break
    for i in range(QUEENS_NUM):
        if citizen.board[i] == -1:
            citizen.board[i] = int(random.randint(0, 32767) % QUEENS_NUM)
    return citizen


def pmx_crossover(first_parent, second_parent):
    global QUEENS_NUM
    citizen = Citizen()
    for i in range(3):
        entry = int(random.randint(0, 32767) % QUEENS_NUM)
        for i in range(QUEENS_NUM):
            if first_parent.board[i] == second_parent.board[entry]:
                citizen.board.append(first_parent.board[entry])
            else:
                citizen.board.append(first_parent.board[i])
        citizen.board[entry] = second_parent.board[entry]
    return citizen


def nqueens_crossover(first_parent, second_parent):
    global CROSSOVER_TYPE
    if CROSSOVER_TYPE == NQueensCrossover.PMX:
        return pmx_crossover(first_parent, second_parent)
    else:
        return ox_crossover(first_parent, second_parent)


def knapsack_crossover(first_parent, second_parent):
    global ITEMS, WEIGHTS
    citizen = Citizen()
    overall_weights = 0
    for i in range(ITEMS):
        if first_parent.knapsack[i] == 1 or second_parent.knapsack[i] == 1:
            citizen.knapsack.append(1)
            overall_weights = overall_weights + WEIGHTS[i]
        else:
            citizen.knapsack.append(0)
    citizen.capacity = overall_weights
    validate_and_update_sack(citizen)
    return citizen


def crossover(first_parent, second_parent, crossover_type):
    global GENETIC_ALGORITHM_TYPE
    if GENETIC_ALGORITHM_TYPE == GeneticAlgorithm.NQueens:
        return nqueens_crossover(first_parent, second_parent)
    if GENETIC_ALGORITHM_TYPE == GeneticAlgorithm.Knapsack:
        return knapsack_crossover(first_parent, second_parent)
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


def mate(population, buffer, selection_type, crossover_type):
    esize = int(GA_POPSIZE * GA_ELITRATE)
    elitism(population, buffer, esize)
    if selection_type == Selection.AGING:
        initialize_aging_data(population)
    for i in range(esize, GA_POPSIZE):
        CHOSEN_PAIRS.clear()
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


def print_sack(sack):
    global ITEMS, VALUE, WEIGHTS
    overall_value = 0
    overall_weight = 0
    for i in range(ITEMS):
        if sack[i] == 1:
            overall_value = overall_value + VALUE[i]
            overall_weight = overall_weight + WEIGHTS[i]
            print("Item " + str(i) + " with value " + str(VALUE[i]))
    print("Overall value is " + str(overall_value))
    print("Overall weight is " + str(overall_weight))


def print_best(gav, iter_num):
    global GENETIC_ALGORITHM_TYPE
    if GENETIC_ALGORITHM_TYPE == GeneticAlgorithm.NQueens:
        print("Best: ")
        print_board(gav[0].board)
        print("Fitness: " + str(gav[0].fitness))
    elif GENETIC_ALGORITHM_TYPE == GeneticAlgorithm.Knapsack:
        print_sack(gav[0].knapsack)
    else:
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


def calc_conflicts(checked_queen_col, board):
    global QUEENS_NUM
    conflicts_num = 0
    checked_queen_row = board[checked_queen_col]
    for current_queen_col in range(QUEENS_NUM):
        # If same queen, don't calculate conflicts
        if current_queen_col == checked_queen_col:
            continue
        current_queen_row = board[current_queen_col]
        if current_queen_row == checked_queen_row or\
                abs(checked_queen_row-current_queen_row) == abs(checked_queen_col-current_queen_col):
            conflicts_num = conflicts_num + 1
    return conflicts_num


def print_board(board):
    global QUEENS_NUM
    for row in range(QUEENS_NUM):
        print_str = ''
        for col in range(QUEENS_NUM):
            if board[col] == row:
                print_str = print_str + 'Q '
            else:
                print_str = print_str + '. '
        print(print_str)


def calc_overall_conflicts(board):
    fitness = 0
    for i in range(QUEENS_NUM):
        fitness = fitness + calc_conflicts(i, board)
    return fitness


def calc_nqueens_fitness(population):
    global QUEENS_NUM
    for citizen in population:
        citizen.fitness = calc_overall_conflicts(citizen.board)


def calc_knapsack_fitness(population):
    global VALUE, ITEMS
    overall_value = sum(VALUE)
    for citizen in population:
        sack_value = 0
        for i in range(ITEMS):
            if citizen.knapsack[i] == 1:
                sack_value = sack_value + VALUE[i]
        citizen.fitness = overall_value - sack_value


def calc_fitness(population, fitness_type):
    global GENETIC_ALGORITHM_TYPE
    if GENETIC_ALGORITHM_TYPE == GeneticAlgorithm.NQueens:
        calc_nqueens_fitness(population)
    elif GENETIC_ALGORITHM_TYPE == GeneticAlgorithm.Knapsack:
        calc_knapsack_fitness(population)
    elif fitness_type == Fitness.DISTANCE:
        calc_distance_fitness(population)
    else:
        calc_bulls_n_cows_fitness(population)


def delete_files():
    if os.path.isfile("./output.txt"):
        os.remove("./output.txt")


def knapsack_fitness_done(citizen):
    global WEIGHTS, VALUE, ITEMS
    global GENETIC_ALGORITHM_TYPE
    if GENETIC_ALGORITHM_TYPE != GeneticAlgorithm.Knapsack:
        return False
    overall_value = sum(VALUE)
    citizen_value = 0
    for i in range(ITEMS):
        if citizen.knapsack[i] == 1:
            citizen_value = citizen_value + VALUE[i]
    if citizen_value / overall_value > 0.75:
        return True
    return False


def genetic_algorithm():
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
        if population[0].fitness == 0 or knapsack_fitness_done(population[0]):
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
    # with open("something", 'a') as file:
    #     for a in print_avgs:
    #         file.write(str(a) + "\n")


def is_solved(board):
    fitness = 0
    for i in range(QUEENS_NUM):
        fitness = fitness + calc_conflicts(i, board)
    if fitness == 0:
        return True
    return False


def find_best_row(board, random_col):
    global QUEENS_NUM
    new_board = copy.deepcopy(board)
    min_conflicts = math.inf
    for i in range(QUEENS_NUM):
        new_board[random_col] = i
        conflicts = calc_overall_conflicts(new_board)
        if conflicts < min_conflicts:
            min_conflicts = conflicts
    best_rows = []
    for i in range(QUEENS_NUM):
        new_board[random_col] = i
        conflicts = calc_overall_conflicts(new_board)
        if conflicts == min_conflicts:
            best_rows.append(i)
    best_row = random.choice(best_rows)
    return best_row


def csp_algorithm():
    global QUEENS_NUM
    overall_time = time.clock()
    board = []
    iter_num = 0
    for i in range(QUEENS_NUM):
        row = (random.randint(0, 32767) % QUEENS_NUM)
        board.append(row)
    while not is_solved(board):
        print("Iteration number " + str(iter_num))
        print_board(board)
        print()
        random_col = (random.randint(0, 32767) % QUEENS_NUM)
        board[random_col] = find_best_row(board, random_col)
        iter_num = iter_num + 1
    overall_time = time.clock() - overall_time
    overall_clock_ticks = overall_time * CLOCKS_PER_SECOND
    print_board(board)
    with open("output.txt", 'a') as file:
        file.write("Overall clock ticks: " + str(overall_clock_ticks) + "\n")
        file.write("Overall time: " + str(overall_time) + "\n")
        file.write("Overall iterations: " + str(iter_num) + "\n")


def main():
    global AlGORITHM
    parse_cmd()
    delete_files()
    if ALGORITHM == Algorithm.GENETIC:
        genetic_algorithm()
    else:
        csp_algorithm()


def parse_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument('-A', default=0, help='Algorithm type: 0 for genetic algorithm, 1 for CSP')
    parser.add_argument('-GA', default=1, help='Genetic algorithm type: 0 for string search, 1 for N-queens, 2 0-1 knapsack')
    parser.add_argument('-N', default=16, help='Queens number')
    parser.add_argument('-QM', default=1, help='Mutation type: 0 for exchange mutation, 1 for simple inversion mutation')
    parser.add_argument('-QC', default=0, help='Crossover type: 0 for PMX crossover, 1 for OX crossover')
    parser.add_argument('-F', default=0, help='Fitness function: 0 for ASCII distance from target string, '
                                              '1 for bulls and cows.')
    parser.add_argument('-S', default=0, help='Selection type: 0 for Random from the top half, 1 for RWS, '
                                              '2 for aging, 3 for tournament.')
    parser.add_argument('-C', default=0, help='Crossover type: 0 for one-point crossover, 1 for two-point crossover.')
    args = parser.parse_args()
    try:
        algorithm_type = int(args.A)
        if algorithm_type != 0 and algorithm_type != 1:
            print("Algorithm can only be 0 or 1.")
            sys.exit()
    except ValueError:
        print("Algorithm can only be 0 or 1.")
        sys.exit()
    try:
        genetic_alg = int(args.GA)
        if genetic_alg != 0 and genetic_alg != 1 and genetic_alg != 2:
            print("Genetic algorithm can only be 0, 1 or 2.")
            sys.exit()
    except ValueError:
        print("Genetic algorithm can only be 0, 1 or 2.")
        sys.exit()
    try:
        queens_number = int(args.N)
        if queens_number < 1 or queens_number > 100:
            print("Queens number can only be a number between 1 and 100.")
            sys.exit()
    except ValueError:
        print("Queens number can only be a number between 1 and 100.")
        sys.exit()
    try:
        mutation_number = int(args.QM)
        if mutation_number != 0 and mutation_number != 1:
            print("Mutation number can only be 0 or 1.")
            sys.exit()
    except ValueError:
        print("Mutation number can only be 0 or 1.")
        sys.exit()
    try:
        crossover_number = int(args.QC)
        if crossover_number != 0 and crossover_number != 1:
            print("Crossover number can only be 0 or 1.")
            sys.exit()
    except ValueError:
        print("Crossover number can only be 0 or 1.")
        sys.exit()
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
    global ALGORITHM, GENETIC_ALGORITHM_TYPE, QUEENS_NUM, MUTATION_TYPE, CROSSOVER_TYPE
    ALGORITHM = Algorithm(algorithm_type)
    GENETIC_ALGORITHM_TYPE = GeneticAlgorithm(genetic_alg)
    QUEENS_NUM = queens_number
    MUTATION_TYPE = NQueensMutation(mutation_number)
    CROSSOVER_TYPE = NQueensCrossover(crossover_number)
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
