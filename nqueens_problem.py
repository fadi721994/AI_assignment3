from citizen import Citizen
from nqueens_mutation import NQueensMutation
from nqueens_crossover import NQueensCrossover
import utils
import random


class NQueensProblem:
    def __init__(self, data):
        self.data = data

    # Check if the fitness is zero, meaning, if the run is over
    def is_done(self, best):
        if best.fitness == 0:
            return True
        return False

    # Randomly initialize rows for columns for each citizen
    def init_citizen(self):
        citizen = Citizen()
        for i in range(self.data.queens_num):
            citizen.board.append(i)
        # Randomize entries
        for col in range(self.data.queens_num):
            row = (random.randint(0, 32767) % self.data.queens_num)
            citizen.board[col] = row
        return citizen

    # Calculate the fitness by calculating the overall conflicts
    def calc_fitness(self, population):
        for citizen in population:
            citizen.fitness = utils.calc_overall_conflicts(citizen.board, self.data.queens_num)

    # Print the best citizen at the end of each generation
    def print_best(self, gav, iter_num):
        print("Iteration number: " + str(iter_num))
        print("Best: ")
        self.print_board(gav[0].board)
        print("Fitness: " + str(gav[0].fitness))
        print()
        with open("output.txt", 'a') as file:
            file.write("Best fitness: " + str(gav[0].fitness) + "\n")
            file.write("    Iteration number: " + str(iter_num) + "\n")
            file.write("    Fitness average: " + str(round(utils.average(gav, self.data.ga_popsize), 3)) + "\n")
            file.write("    Fitness deviation: " + str(round(utils.deviation(gav, self.data.ga_popsize), 3)) + "\n")

    # Pretty print the board
    def print_board(self, board):
        for row in range(self.data.queens_num):
            print_str = ''
            for col in range(self.data.queens_num):
                if board[col] == row:
                    print_str = print_str + 'Q '
                else:
                    print_str = print_str + '. '
            print(print_str)

    # Choose the mutation method and run it
    def mutate(self, citizen):
        if self.data.queens_mutation == NQueensMutation.EXCHANGE:
            self.exchange_mutation(citizen)
        else:
            self.simple_inversion_mutation(citizen)

    # Exchange mutation, switch the places of 2 queens
    def exchange_mutation(self, citizen):
        col_1 = int(random.randint(0, 32767) % self.data.queens_num)
        col_2 = int(random.randint(0, 32767) % self.data.queens_num)
        while col_1 == col_2:
            col_2 = int(random.randint(0, 32767) % self.data.queens_num)
        citizen.board[col_1], citizen.board[col_2] = citizen.board[col_2], citizen.board[col_1]

    # Simple inversion mutation, reverse a section of the board
    def simple_inversion_mutation(self, citizen):
        col_1 = int(random.randint(0, 32767) % self.data.queens_num)
        col_2 = int(random.randint(0, 32767) % self.data.queens_num)
        while col_1 == col_2:
            col_2 = int(random.randint(0, 32767) % self.data.queens_num)
        for_range = int(abs(col_2 - col_1) / 2) + 1
        if col_1 > col_2:
            col_1, col_2 = col_2, col_1
        for i in range(for_range):
            citizen.board[col_1+i], citizen.board[col_2-i] = citizen.board[col_2-i], citizen.board[col_1+i]

    # Choose the crossover method and run it
    def crossover(self, first_parent, second_parent):
        if self.data.queens_crossover == NQueensCrossover.PMX:
            return self.pmx_crossover(first_parent, second_parent)
        else:
            return self.ox_crossover(first_parent, second_parent)

    # Perform PMX crossover. We run it 3 times for each descendant
    def pmx_crossover(self, first_parent, second_parent):
        citizen = Citizen()
        for i in range(3):
            entry = int(random.randint(0, 32767) % self.data.queens_num)
            for i in range(self.data.queens_num):
                if first_parent.board[i] == second_parent.board[entry]:
                    citizen.board.append(first_parent.board[entry])
                else:
                    citizen.board.append(first_parent.board[i])
            citizen.board[entry] = second_parent.board[entry]
        return citizen

    # OX crossover. Like described in the Permutation cross-over file
    def ox_crossover(self, first_parent, second_parent):
        chosen_entries = 0
        citizen = Citizen()
        first_parent_entries = []
        first_parent_chosen_values = []
        for i in range(self.data.queens_num):
            citizen.board.append(-1)
        while chosen_entries != self.data.queens_num / 2:
            entry = int(random.randint(0, 32767) % self.data.queens_num)
            if entry not in first_parent_entries:
                first_parent_entries.append(entry)
                first_parent_chosen_values.append(first_parent.board[entry])
                chosen_entries = chosen_entries + 1
                citizen.board[entry] = first_parent.board[entry]
        second_parent_chosen_values = []
        for i in range(self.data.queens_num):
            if citizen.board[i] == -1:
                for j in range(self.data.queens_num):
                    second_parent_value = second_parent.board[j]
                    if second_parent_value not in first_parent_chosen_values:
                        if second_parent_value not in second_parent_chosen_values:
                            citizen.board[i] = second_parent_value
                            second_parent_chosen_values.append(second_parent_value)
                            break
        for i in range(self.data.queens_num):
            if citizen.board[i] == -1:
                citizen.board[i] = int(random.randint(0, 32767) % self.data.queens_num)
        return citizen
