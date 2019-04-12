import time
import random
import utils
import copy
import math


class CSPAlgorithm:
    def __init__(self, data):
        self.data = data
        print("Running CSP algorithm")

    # The actual algorithm run
    def run(self):
        overall_time = time.clock()
        board = []
        iter_num = 0
        for i in range(self.data.queens_num):
            row = (random.randint(0, 32767) % self.data.queens_num)
            board.append(row)
        while not self.is_solved(board):
            print("Iteration number " + str(iter_num))
            utils.print_board(board, self.data.queens_num)
            print()
            random_col = (random.randint(0, 32767) % self.data.queens_num)
            board[random_col] = self.find_best_row(board, random_col)
            iter_num = iter_num + 1
        overall_time = time.clock() - overall_time
        overall_clock_ticks = overall_time * self.data.clocks_per_second
        with open("output.txt", 'a') as file:
            file.write("Overall clock ticks: " + str(overall_clock_ticks) + "\n")
            file.write("Overall time: " + str(overall_time) + "\n")
            file.write("Overall iterations: " + str(iter_num) + "\n")

    # Check if the n-queens board is solved
    def is_solved(self, board):
        fitness = 0
        for i in range(self.data.queens_num):
            fitness = fitness + utils.calc_conflicts(i, board, self.data.queens_num)
        if fitness == 0:
            return True
        return False

    # Find the best row to move the queen to
    def find_best_row(self, board, random_col):
        # Find the minimum value for conflicts
        new_board = copy.deepcopy(board)
        min_conflicts = math.inf
        for i in range(self.data.queens_num):
            new_board[random_col] = i
            conflicts = utils.calc_overall_conflicts(new_board, self.data.queens_num)
            if conflicts < min_conflicts:
                min_conflicts = conflicts
        # Get a list of rows to move the queen to, which provide the minimum value
        best_rows = []
        for i in range(self.data.queens_num):
            new_board[random_col] = i
            conflicts = utils.calc_overall_conflicts(new_board, self.data.queens_num)
            if conflicts == min_conflicts:
                best_rows.append(i)
        # Choose randomly among the minimum value rows to move to
        best_row = random.choice(best_rows)
        return best_row
