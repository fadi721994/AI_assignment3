import random
from ga_struct import GaStruct

GA_POPSIZE = 2048
GA_MAXITER = 16384
GA_ELITRATE = 0.1
GA_MUTATIONRATE = 0.25
GA_MUTATION = random.randint(0, 32767) * GA_MUTATIONRATE
GA_TARGET = "Hello World!"


def init_population(population, buffer):
    tsize = len(GA_TARGET)
    for i in range(GA_POPSIZE):
        citizen = GaStruct('', 0)
        for j in range(tsize):
            citizen.str = citizen.str + str(chr((random.randint(0, 32767) % 90) + 32))
        population.append(citizen)
    for i in range(GA_POPSIZE):
        citizen = GaStruct('', 0)
        buffer.append(citizen)


def calc_fitness(population):
    target = GA_TARGET
    tsize = len(target)
    for i in range(GA_POPSIZE):
        fitness = 0
        for j in range(tsize):
            population_string = population[i].str
            if j < len(population_string):
                fitness = fitness + abs(int(ord(population_string[j]) - ord(target[j])))
        population[i].fitness = fitness


def sort_by_fitness(population):
    population = population.sort()


def elitism(population, buffer, esize):
    for i in range(esize):
        buffer[i].str = population[i].str
        buffer[i].fitness = population[i].fitness


def mutate(member):
    tsize = len(GA_TARGET)
    ipos = int(random.randint(0, 32767) % tsize)
    delta = int((random.randint(0, 32767) % 90) + 32)
    string_list = list(member.str)
    string_list[ipos] = str(chr(((ord(string_list[ipos]) + delta) % 122)))
    member.str = ''.join(string_list)


def mate(population, buffer):
    esize = int(GA_POPSIZE * GA_ELITRATE)
    tsize = len(GA_TARGET)

    elitism(population, buffer, esize)
    for i in range(esize, GA_POPSIZE):
        i1 = int(random.randint(0, 32767) % (GA_POPSIZE / 2))
        i2 = int(random.randint(0, 32767) % (GA_POPSIZE / 2))
        spos = int(random.randint(0, 32767) % tsize)
        buffer[i] = GaStruct(population[i1].str[0:spos] + population[i2].str[spos:tsize], 0)
        if random.randint(0, 32767) < GA_MUTATION:
            mutate(buffer[i])


def print_best(gav):
    print("Best: " + gav[0].str + " (" + str(gav[0].fitness) + ")")


def swap(population, buffer):
    temp = population
    population = buffer
    buffer = temp
    return population, buffer


def main():
    pop_alpha = []
    pop_beta = []
    init_population(pop_alpha, pop_beta)
    population = pop_alpha
    buffer = pop_beta

    for i in range(GA_MAXITER):
        calc_fitness(population)
        sort_by_fitness(population)
        print_best(population)
        if population[0].fitness == 0:
            break
        mate(population, buffer)
        buffer, population = population, buffer


main()
