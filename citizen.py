class Citizen:
    def __init__(self, str='', fitness=0, age=0, board=None, knapsack=None, cap=0):
        self.str = str
        if board is None:
            self.board = []
        else:
            self.board = board
        if knapsack is None:
            self.knapsack = []
        else:
            self.knapsack = knapsack
        self.capacity = cap
        self.fitness = fitness
        self.age = age

    def __lt__(self, other):
        return self.fitness < other.fitness
