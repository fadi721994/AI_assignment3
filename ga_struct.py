class GaStruct:
    def __init__(self, str, fitness):
        self.str = str
        self.fitness = fitness

    def __lt__(self, other):
        return self.fitness < other.fitness
