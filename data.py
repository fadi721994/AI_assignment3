import random
import argparse
import sys
from algorithm import Algorithm
from genetic_problem import GeneticProblem
from nqueens_crossover import NQueensCrossover
from nqueens_mutation import NQueensMutation
from string_search_fitness import StringSearchFitness
from string_search_crossover import StringSearchCrossOver
from selection import Selection


class Data:
    def __init__(self):
        # General initialization
        self.algorithm = Algorithm.GENETIC
        self.genetic_problem = GeneticProblem.STRING_SEARCH
        self.clocks_per_second = 1000

        # General genetic algorithm initialization
        self.ga_popsize = 2048
        self.ga_maxiter = 16384
        self.ga_elitrate = 0.1
        self.ga_mutationrate = 0.25
        self.ga_mutation = 32767 * self.ga_mutationrate

        # Init parent selection method
        self.selection = Selection.RANDOM
        self.original_selection = Selection.RANDOM

        # Specific string-search problem initialization
        self.ga_target = "Hello World!"
        self.ga_age = 5
        self.ga_k = 2
        self.aged_citizens = []
        self.chosen_pairs = []
        self.string_search_fitness = StringSearchFitness.DISTANCE
        self.string_search_crossover = StringSearchCrossOver.ONE_POINT

        # Specific n-queens problem initialization
        self.queens_num = 8
        self.queens_mutation = NQueensMutation.EXCHANGE
        self.queens_crossover = NQueensCrossover.PMX

        # Specific knapsack problem initialization
        self.knapsack_problem = 1

        self.parse_cmd()

    def parse_cmd(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-A', default=0, help='Algorithm type: 0 for genetic algorithm, 1 for CSP')
        parser.add_argument('-GP', default=0,
                            help='Genetic algorithm type: 0 for string search, 1 for N-queens, 2 0-1 knapsack')
        parser.add_argument('-KP', default=1, help='Knapsack problem number. Can be between 1 and 8')
        parser.add_argument('-QN', default=8, help='Queens number')
        parser.add_argument('-QM', default=0,
                            help='N-queens mutation type: 0 for exchange mutation, 1 for simple inversion mutation')
        parser.add_argument('-QC', default=0, help='N-queens crossover type: 0 for PMX crossover, 1 for OX crossover')
        parser.add_argument('-SF', default=0, help='String search fitness function: 0 for ASCII distance from target '
                                                   'string, 1 for bulls and cows.')
        parser.add_argument('-SC', default=0,
                            help='String search crossover type: 0 for one-point crossover, 1 for two-point crossover.')
        parser.add_argument('-S', default=0, help='Selection type: 0 for Random from the top half, 1 for RWS, '
                                                  '2 for aging, 3 for tournament.')
        parser.add_argument('-PS', default=2048, help='Population size')
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
            genetic_prob = int(args.GP)
            if genetic_prob != 0 and genetic_prob != 1 and genetic_prob != 2:
                print("Genetic algorithm can only be 0, 1 or 2.")
                sys.exit()
        except ValueError:
            print("Genetic algorithm can only be 0, 1 or 2.")
            sys.exit()

        try:
            knapsack_problem = int(args.KP)
            if knapsack_problem != 1 and knapsack_problem != 2 and knapsack_problem != 3 and knapsack_problem != 4\
                    and knapsack_problem != 5 and knapsack_problem != 6 and knapsack_problem != 7\
                    and knapsack_problem != 8:
                print("Knapsack problem number can only be a number between 1 and 8 (including).")
                sys.exit()
        except ValueError:
            print("Knapsack problem number can only be a number between 1 and 8 (including).")
            sys.exit()

        try:
            queens_number = int(args.QN)
            if queens_number < 1 or queens_number > 100:
                print("Queens number can only be a number between 1 and 100.")
                sys.exit()
        except ValueError:
            print("Queens number can only be a number between 1 and 100.")
            sys.exit()

        try:
            mutation_number = int(args.QM)
            if mutation_number != 0 and mutation_number != 1:
                print("N-queens mutation number can only be 0 or 1.")
                sys.exit()
        except ValueError:
            print("N-queens mutation number can only be 0 or 1.")
            sys.exit()

        try:
            crossover_number = int(args.QC)
            if crossover_number != 0 and crossover_number != 1:
                print("N-queens crossover number can only be 0 or 1.")
                sys.exit()
        except ValueError:
            print("N-queens crossover number can only be 0 or 1.")
            sys.exit()

        try:
            fitness_function = int(args.SF)
            if fitness_function != 0 and fitness_function != 1:
                print("String search fitness function can only be 0 or 1.")
                sys.exit()
        except ValueError:
            print("String search fitness function can only be 0 or 1.")
            sys.exit()

        try:
            crossover_type = int(args.SC)
            if crossover_type != 0 and crossover_type != 1:
                print("String search crossover type can only be 0 or 1.")
                sys.exit()
        except ValueError:
            print("String search crossover type can only be 0 or 1.")
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
            population_size = int(args.PS)
            if population_size < 10:
                print("Population size can only be 10 or larger.")
                sys.exit()
        except ValueError:
            print("Population size can only be a number, 10 or larger.")
            sys.exit()

        # Initialize general values
        self.algorithm = Algorithm(algorithm_type)
        self.genetic_problem = GeneticProblem(genetic_prob)
        self.selection = Selection(selection_type)
        self.original_selection = self.selection
        self.ga_popsize = population_size

        # Initialize string-search values
        self.string_search_crossover = StringSearchCrossOver(crossover_type)
        self.string_search_fitness = StringSearchFitness(fitness_function)

        # Initialize n-queens values
        self.queens_num = queens_number
        self.queens_mutation = NQueensMutation(mutation_number)
        self.queens_crossover = NQueensCrossover(crossover_number)

        # Initialize knapsack values:
        self.knapsack_problem = knapsack_problem - 1
