import os
import math


# Delete the output.txt file at the beginning of each run
def delete_files():
    if os.path.isfile("./output.txt"):
        os.remove("./output.txt")


# Calculate the overall fitness of a population
def fitness_sum(population):
    overall = 0
    for citizen in population:
        overall = overall + citizen.fitness
    return overall


# Calculate the average of a population
def average(population, pop_size):
    return fitness_sum(population) / pop_size


# Calculate the deviation of a population
def deviation(population, pop_size):
    avg = average(population, pop_size)
    overall = 0
    for i in range(pop_size):
        overall = overall + ((avg - population[i].fitness) ** 2)
    overall = math.sqrt(overall / pop_size)
    return overall


# Calculate the number of conflicts for a queen, given a board and the queen column
def calc_conflicts(checked_queen_col, board, queens_num):
    conflicts_num = 0
    checked_queen_row = board[checked_queen_col]
    for current_queen_col in range(queens_num):
        # If same queen, don't calculate conflicts
        if current_queen_col == checked_queen_col:
            continue
        current_queen_row = board[current_queen_col]
        if current_queen_row == checked_queen_row or\
                abs(checked_queen_row-current_queen_row) == abs(checked_queen_col-current_queen_col):
            conflicts_num = conflicts_num + 1
    return conflicts_num


# Calculate the overall conflicts
def calc_overall_conflicts(board, queens_num):
    fitness = 0
    for i in range(queens_num):
        fitness = fitness + calc_conflicts(i, board, queens_num)
    return fitness


# Print the n-queens board
def print_board(board, queens_num):
    for row in range(queens_num):
        print_str = ''
        for col in range(queens_num):
            if board[col] == row:
                print_str = print_str + 'Q '
            else:
                print_str = print_str + '. '
        print(print_str)
    print()
