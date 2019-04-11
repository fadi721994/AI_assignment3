from citizen import Citizen
import random
import utils
import math


class KnapsackProblem:
    def __init__(self, data):
        self.data = data
        self.problems_capacities = [165, 26, 190, 50, 104, 170, 750, 6404180]
        self.problems_weights = [
            [23, 31, 29, 44, 53, 38, 63, 85, 89, 82],
            [12, 7, 11, 8, 9],
            [56, 59, 80, 64, 75, 17],
            [31, 10, 20, 19, 4, 3, 6],
            [25, 35, 45, 5, 25, 3, 2, 2],
            [41, 50, 49, 59, 55, 57, 60],
            [70, 73, 77, 80, 82, 87, 90, 94, 98, 106, 110, 113, 115, 118, 120],
            [382745, 799601, 909247, 729069, 467902, 44328, 34610, 698150, 823460, 903959, 853665, 551830, 610856,
             670702, 488960, 951111, 323046, 446298, 931161, 31385, 496951, 264724, 224916, 169684]
        ]
        self.problems_prices = [
            [92, 57, 49, 68, 60, 43, 67, 84, 87, 72],
            [24, 13, 23, 15, 16],
            [50, 50, 64, 46, 50, 5],
            [70, 20, 39, 37, 7, 5, 10],
            [350, 400, 450, 20, 70, 8, 5, 5],
            [442, 525, 511, 593, 546, 564, 617],
            [135, 139, 149, 150, 156, 163, 173, 184, 192, 201, 210, 214, 221, 229, 240],
            [825594, 1677009, 1676628, 1523970, 943972, 97426, 69666, 1296457, 1679693, 1902996, 1844992, 1049289,
             1252836, 1319836, 953277, 2067538, 675367, 853655, 1826027, 65731, 901489, 577243, 466257, 369261]
        ]
        self.problems_optimal_solutions = [
            [1, 1, 1, 1, 0, 1, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [1, 1, 0, 0, 1, 0],
            [1, 0, 0, 1, 0, 0, 0],
            [1, 0, 1, 1, 1, 0, 1, 1],
            [0, 1, 0, 1, 0, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1],
            [1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1]
        ]
        self.capacity = self.problems_capacities[self.data.knapsack_problem]
        self.weights = self.problems_weights[self.data.knapsack_problem]
        self.prices = self.problems_prices[self.data.knapsack_problem]
        self.optimal_solution = self.problems_optimal_solutions[self.data.knapsack_problem]
        self.items = len(self.weights)

    def is_done(self, best):
        for i in range(self.items):
            if best.knapsack[i] != self.optimal_solution[i]:
                return False
        return True

    def init_citizen(self):
        citizen = Citizen()
        for j in range(self.items):
            choose = (random.randint(0, 32767) % 2)
            if choose == 1:
                citizen.knapsack.append(1)
            else:
                citizen.knapsack.append(0)
        self.validate_and_update_sack(citizen)
        return citizen

    def validate_and_update_sack(self, citizen):
        overall_weight = 0
        for i in range(self.items):
            if citizen.knapsack[i] == 1:
                overall_weight = overall_weight + self.weights[i]
        while overall_weight > self.capacity:
            chosen_items = []
            for i in range(self.items):
                if citizen.knapsack[i] == 1:
                    chosen_items.append(i)
            item = random.choice(chosen_items)
            overall_weight = overall_weight - self.weights[item]
            citizen.knapsack[item] = 0
        citizen.capacity = overall_weight

    def calc_fitness(self, population):
        overall_value = sum(self.prices)
        for citizen in population:
            citizen.fitness = overall_value - self.get_sack_value(citizen)

    def get_sack_value(self, citizen):
        sack_value = 0
        for i in range(self.items):
            if citizen.knapsack[i] == 1:
                sack_value = sack_value + self.prices[i]
        return sack_value

    def print_best(self, gav, iter_num):
        print("Iteration number " + str(iter_num))
        with open("output.txt", 'a') as file:
            file.write("Iteration number: " + str(iter_num) + "\n")
            file.write("    Best fitness: " + str(gav[0].fitness) + "\n")
            overall_value = 0
            overall_weight = 0
            for i in range(self.items):
                if gav[0].knapsack[i] == 1:
                    overall_value = overall_value + self.prices[i]
                    overall_weight = overall_weight + self.weights[i]
                    print("Item " + str(i) + " with value " + str(self.prices[i]) + " and weight "
                          + str(self.weights[i]))
                    file.write("    Item " + str(i) + " with value " + str(self.prices[i]) + " and weight " +
                               str(self.weights[i]) + "\n")
            print("Overall value is " + str(overall_value) + "/" + str(sum(self.prices)))
            print("Overall weight is " + str(overall_weight) + "/" + str(self.capacity))
            print()
            file.write("    Overall value is " + str(overall_value) + "\n")
            file.write("    Overall weight is " + str(overall_weight) + "\n")
            file.write("    Fitness average: " + str(round(utils.average(gav, self.data.ga_popsize), 3)) + "\n")
            file.write("    Fitness deviation: " + str(round(utils.deviation(gav, self.data.ga_popsize), 3)) + "\n")

    def mutate(self, citizen):
        item_1 = int(random.randint(0, 32767) % self.items)
        item_2 = int(random.randint(0, 32767) % self.items)
        while item_1 == item_2:
            item_2 = int(random.randint(0, 32767) % self.items)
        for_range = int(abs(item_1 - item_2) / 2) + 1
        if item_1 > item_2:
            item_1, item_2 = item_2, item_1
        for i in range(for_range):
            citizen.knapsack[item_1+i], citizen.knapsack[item_2-i] = \
                citizen.knapsack[item_2-i], citizen.knapsack[item_1+i]
        self.validate_and_update_sack(citizen)

    def crossover(self, first_parent, second_parent):
        citizen = Citizen()
        overall_weights = 0
        chosen_items = []
        not_chosen_items = []
        for i in range(self.items):
            if first_parent.knapsack[i] == 1 or second_parent.knapsack[i] == 1:
                citizen.knapsack.append(1)
                chosen_items.append(i)
            else:
                citizen.knapsack.append(0)
                not_chosen_items.append(i)
        if chosen_items:
            citizen.knapsack[random.choice(chosen_items)] = 0
        if not_chosen_items:
            citizen.knapsack[random.choice(not_chosen_items)] = 1
            citizen.knapsack[random.choice(not_chosen_items)] = 1
        for i in range(self.items):
            if citizen.knapsack[i] == 1:
                overall_weights = overall_weights + self.weights[i]
        citizen.capacity = overall_weights
        self.validate_and_update_sack(citizen)
        return citizen
