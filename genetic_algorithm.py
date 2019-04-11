import time
import random
from genetic_problem import GeneticProblem
from string_search_problem import StringSearchProblem
from nqueens_problem import NQueensProblem
from knapsack_problem import KnapsackProblem
from selection import Selection
from citizen import Citizen


class GeneticAlgorithm:
    def __init__(self, data):
        self.data = data
        if self.data.genetic_problem == GeneticProblem.STRING_SEARCH:
            self.problem = StringSearchProblem(data)
        elif self.data.genetic_problem == GeneticProblem.NQUEENS:
            self.problem = NQueensProblem(data)
        else:
            self.problem = KnapsackProblem(data)

    # The actual algorithm run
    def run(self):
        overall_time = time.clock()
        population = []
        buffer = []
        self.init_population(population, buffer)
        iter_num = 0
        for i in range(self.data.ga_maxiter):
            iter_num = i
            start_time = time.clock()
            self.problem.calc_fitness(population)
            population.sort()
            self.problem.print_best(population, i)
            if self.problem.is_done(population[0]):
                self.print_data(start_time)
                break
            self.mate(population, buffer)
            buffer, population = population, buffer
            self.print_data(start_time)

        overall_time = time.clock() - overall_time
        overall_clock_ticks = overall_time * self.data.clocks_per_second
        with open("output.txt", 'a') as file:
            file.write("Overall clock ticks: " + str(overall_clock_ticks) + "\n")
            file.write("Overall time: " + str(overall_time) + "\n")
            file.write("Overall iterations: " + str(iter_num) + "\n")

    # Initialize the population
    def init_population(self, population, buffer):
        for i in range(self.data.ga_popsize):
            population.append(self.problem.init_citizen())
        for i in range(self.data.ga_popsize):
            buffer.append(Citizen())

    # Print the data from the best fitness citizen
    def print_data(self, start_time):
        run_time = time.clock() - start_time
        clock_ticks = run_time * self.data.clocks_per_second
        with open("output.txt", 'a') as file:
            file.write("    Clock ticks elapsed: " + str(round(clock_ticks, 3)) + "\n")
            file.write("    Time elapsed: " + str(round(run_time, 3)) + "\n")

    # Mate the citizens to create a new generation
    def mate(self, population, buffer):
        esize = int(self.data.ga_popsize * self.data.ga_elitrate)
        self.elitism(population, buffer, esize)
        # Since aging uses the regular RANDOM from top half selection until we reach a point where we have aged citizens
        # we had to initialize another selection field called original_selection and use it when necessary
        if self.data.original_selection == Selection.AGING:
            self.initialize_aging_data(population)
            self.data.chosen_pairs.clear()
            self.data.selection = Selection.AGING
        for i in range(esize, self.data.ga_popsize):
            i1, i2 = self.select_parents(population)
            buffer[i] = self.problem.crossover(population[i1], population[i2])
            if random.randint(0, 32767) < self.data.ga_mutation:
                self.problem.mutate(buffer[i])

    # Move the percentage of elite to the next generation without changing them
    def elitism(self, population, buffer, esize):
        for i in range(esize):
            buffer[i].str = population[i].str
            buffer[i].fitness = population[i].fitness
            buffer[i].board = population[i].board
            buffer[i].knapsack = population[i].knapsack
            buffer[i].capacity = population[i].capacity
            buffer[i].age = population[i].age + 1
        return esize

    # When using aging, upon each generation, find which citizens are older than the GA_age
    def initialize_aging_data(self, population):
        self.data.aged_citizens.clear()
        for i, citizen in enumerate(population):
            if citizen.age > self.data.ga_age:
                self.data.aged_citizens.append(i)

    # Select the parents entries
    def select_parents(self, population):
        if self.data.original_selection == Selection.AGING and self.data.selection == Selection.AGING:
            if len(self.data.aged_citizens) < 2:
                self.data.selection = Selection.RANDOM
                return self.select_parents(population)
            else:
                for i in range(5):
                    i1 = random.choice(self.data.aged_citizens)
                    i2 = random.choice(self.data.aged_citizens)
                    if i2 > i1:
                        i1, i2 = i2, i1
                    if (i1, i2) not in self.data.chosen_pairs:
                        self.data.chosen_pairs.append((i1, i2))
                        return i1, i2
                self.data.selection = Selection.RANDOM
                return self.select_parents(population)
        elif self.data.selection == Selection.RANDOM:
            if self.data.original_selection == Selection.AGING:
                self.data.selection = Selection.AGING
            i1, i2 = self.random_selection()
            return i1, i2
        elif self.data.selection == Selection.RWS:
            max_weight = population[self.data.ga_popsize - 1].fitness
            weights = []
            for citizen in population:
                weights.append(max_weight - citizen.fitness)
            overall_weights = sum(weights)
            i1 = self.rws_selection(weights, overall_weights)
            i2 = self.rws_selection(weights, overall_weights)
            return i1, i2
        elif self.data.selection == Selection.TOURNAMENT:
            i1 = self.tournament_selection(population)
            i2 = self.tournament_selection(population)
            return i1, i2
        return 0, 1

    # Random selection from better half of population
    def random_selection(self):
        i1 = int(random.randint(0, 32767) % (self.data.ga_popsize / 2))
        i2 = int(random.randint(0, 32767) % (self.data.ga_popsize / 2))
        return i1, i2

    # Roulette selection, given weights for each citizen's fitness
    def rws_selection(self, weights, overall_weights):
        f = random.randint(0, overall_weights)
        for i, weight in enumerate(weights):
            f = f - weight
            if f <= 0:
                return i
        return 0

    # Tournament selection, choosing k citizens and returning the one with the best fitness
    def tournament_selection(self, population):
        best = 0
        for i in range(self.data.ga_k):
            entry = int(random.randint(0, 32767) % self.data.ga_popsize)
            if i == 0 or population[entry].fitness < population[best].fitness:
                best = entry
        return best
