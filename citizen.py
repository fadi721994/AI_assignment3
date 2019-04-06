class Citizen:
    def __init__(self, str='', fitness=0, age=0):
        self.str = str
        self.fitness = fitness
        self.age = age

    def __lt__(self, other):
        return self.fitness < other.fitness
