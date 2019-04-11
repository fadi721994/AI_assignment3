from algorithm import Algorithm
from genetic_algorithm import GeneticAlgorithm
from csp_algorithm import CSPAlgorithm
import utils
from data import Data
import os
import cProfile, pstats, io


def main():
    utils.delete_files()
    data = Data()
    if data.algorithm == Algorithm.GENETIC:
        algorithm = GeneticAlgorithm(data)
    else:
        algorithm = CSPAlgorithm(data)
    algorithm.run()


# pr = cProfile.Profile()
# pr.enable()
main()
# pr.disable()
# s = io.StringIO()
# sortby = 'cumulative'
# ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
# ps.print_stats()
# print(s.getvalue())
