import numpy as np


class ES(object):
    def __init__(self,
                 popul_size,
                 fitness_vec_size,
                 l_bound,
                 u_bound):
        self.l_bound = l_bound
        self.u_bound = u_bound
        self.population_size = popul_size
        self.fitness_vector_size = fitness_vec_size
        self.population = None
        self.offspring = None

    def generate_population(self):
        rand_vals = (self.u_bound - self.l_bound) *\
            np.random.random((self.population_size, self.fitness_vector_size)) + self.l_bound
        self.population = [Individual(p) for p in rand_vals]

    def mutate(self):
        best = np.array(sorted(self.population, key=lambda x: x.fitness))[:mu]
        rnd = np.random.normal(loc=mn, scale=sigma_square, size=(len(best), ff_vec_size))
        new_popul = np.array(([x.phenotypes for x in best]))
        new_popul += rnd
        if isplus:
            new_popul = np.vstack((new_popul, np.array([x.phenotypes for x in best])))
        self.population = [Individual(x) for x in new_popul]


class Individual(object):
    def __init__(self, phenotypes):
        self.phenotypes = phenotypes  # phenotype
        self.fitness = fitness_func(self.phenotypes)  # value of the fitness function

    def __str__(self):
        return '{0} = {1}'.format(self.phenotypes, self.fitness)

def fitness_func(arg_vec):
    # arg_vec = real(arg_vec)
    # Sphere model (DeJong1)
    # return np.sum([x ** 2 for x in arg_vec])
    # Rosenbrock's saddle (DeJong2)
    # return sum([(100 * (xj - xi ** 2) ** 2 + (xi - 1) ** 2) for xi, xj in zip(arg_vec[:-1], arg_vec[1:])])
    # Rastrigin's function
    # return 10 * len(arg_vec) + np.sum([x ** 2 - 10 * np.cos(2 * np.pi * x) for x in arg_vec])
    # Ackley's Function
    # s1 = -0.2 * np.sqrt(np.sum([x ** 2 for x in arg_vec]) / len(arg_vec))
    # s2 = np.sum([np.cos(2 * np.pi * x) for x in arg_vec]) / len(arg_vec)
    # return 20 + np.e - 20 * np.exp(s1) - np.exp(s2)
    # Snytyuk function
    s1 = (sum(arg_vec) - sum(x*x for x in arg_vec)) * sum(np.cos(x) for x in arg_vec)
    s2 = 4 / (np.sqrt(np.abs(np.tan(sum(arg_vec))))) + int(sum(x*x for x in arg_vec))
    return s1 / s2

# Initial values
#
# fitness_func      - fitness_func to be evaluated
# interval          - fitness function interval
# p_c               - crossover probability
# p_b               - mutation probability
# population_size   - the size of desirable population
# ff_vec_size       - number of arguments that is given to fitness_func
# max_epochs        - number of iteration to be evaluated

interval = (-1.048, 1.048)
eps = 1E-3
mu = 2  # (parents)
lam = 18  # (offspring)
sigma_square = 0.6
mn = 0
isplus = True
ff_vec_size = 5
max_epochs = 1000

population_size = lam * mu
step = (sigma_square - 0.1) / max_epochs


def main():
    global sigma_square
    es = ES(population_size,
            ff_vec_size,
            *interval)

    es.generate_population()
    print('Initial population')
    for ind in sorted(es.population, key=lambda x: x.fitness):
        print(ind)
    for i in range(max_epochs):
        es.mutate()
        print('{0}/{1} Current population:'.format(i + 1, max_epochs))
        print(sorted(es.population, key=lambda x: x.fitness)[0])
        sigma_square -= step

if __name__ == '__main__':
    main()
