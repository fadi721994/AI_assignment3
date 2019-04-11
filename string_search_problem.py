from citizen import Citizen
from string_search_fitness import StringSearchFitness
from string_search_crossover import StringSearchCrossOver
import random
import utils


class StringSearchProblem:
    def __init__(self, data):
        self.data = data

    def is_done(self, best):
        if best.fitness == 0:
            return True
        return False

    def init_citizen(self):
        citizen = Citizen()
        tsize = len(self.data.ga_target)
        for j in range(tsize):
            citizen.str = citizen.str + str(chr((random.randint(0, 32767) % 90) + 32))
        return citizen

    def calc_fitness(self, population):
        if self.data.string_search_fitness == StringSearchFitness.DISTANCE:
            self.calc_distance_fitness(population)
        else:
            self.calc_bulls_n_cows_fitness(population)

    def calc_distance_fitness(self, population):
        target = self.data.ga_target
        tsize = len(target)
        for i in range(self.data.ga_popsize):
            fitness = 0
            for j in range(tsize):
                fitness = fitness + abs(int(ord(population[i].str[j]) - ord(target[j])))
            population[i].fitness = fitness

    def calc_bulls_n_cows_fitness(self, population):
        target = self.data.ga_target
        tsize = len(target)
        not_found_penalty = 50
        incorrect_place_penalty = 20
        for i in range(self.data.ga_popsize):
            fitness = 0
            for j in range(tsize):
                if population[i].str[j] != target[j]:
                    if population[i].str[j] in target:
                        fitness = fitness + incorrect_place_penalty
                    else:
                        fitness = fitness + not_found_penalty
            population[i].fitness = fitness

    def print_best(self, gav, iter_num):
        print("Best: " + gav[0].str + " (" + str(gav[0].fitness) + ")")
        with open("output.txt", 'a') as file:
            file.write("Best: " + gav[0].str + " (" + str(gav[0].fitness) + ")\n")
            file.write("    Iteration number: " + str(iter_num) + "\n")
            file.write("    Fitness average: " + str(round(utils.average(gav, self.data.ga_popsize), 3)) + "\n")
            file.write("    Fitness deviation: " + str(round(utils.deviation(gav, self.data.ga_popsize), 3)) + "\n")

    def mutate(self, citizen):
        tsize = len(self.data.ga_target)
        ipos = int(random.randint(0, 32767) % tsize)
        delta = int((random.randint(0, 32767) % 90) + 32)
        string_list = list(citizen.str)
        string_list[ipos] = str(chr(((ord(string_list[ipos]) + delta) % 122)))
        citizen.str = ''.join(string_list)

    def crossover(self, first_parent, second_parent):
        tsize = len(self.data.ga_target)
        if self.data.string_search_crossover == StringSearchCrossOver.ONE_POINT:
            spos = int(random.randint(0, 32767) % tsize)
            return Citizen(first_parent.str[0:spos] + second_parent.str[spos:tsize])
        elif self.data.string_search_crossover == StringSearchCrossOver.TWO_POINT:
            spos1 = int(random.randint(0, 32767) % tsize)
            spos2 = int(random.randint(0, 32767) % tsize)
            if spos1 > spos2:
                spos1, spos2 = spos2, spos1
            return Citizen(first_parent.str[0:spos1] + second_parent.str[spos1:spos2] +
                           first_parent.str[spos2:tsize])
