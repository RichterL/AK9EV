from constants import Settings
import errors
from benchmarks import Benchmark
from typing import Type
import numpy as np


class Algorithm:
    NAME = "NONE"

    def __init__(
        self,
        benchmark: Type[Benchmark],
        dimensions,
    ):
        self.benchmark = benchmark(max_runs=Settings.COST_FUNCTION_EVALUATION_FACTOR * dimensions)
        self.lowerBound = self.benchmark.get_lower_bound()
        self.upperBound = self.benchmark.get_upper_bound()
        self.objective = self.benchmark.evaluate
        self.dimensions = dimensions
        if dimensions <= 2:
            self.populationSize = 10
        elif dimensions <= 10:
            self.populationSize = 20
        else:
            self.populationSize = 50

        self.population = None
        self.costs = None
        self.best_individual = None
        self.best_individual_cost = np.inf

    def restart(self):
        raise NotImplementedError

    def evolve(self):
        raise NotImplementedError

    def _create_population(self):
        return np.random.uniform(self.lowerBound, self.upperBound, size=(self.populationSize, self.dimensions))


class DifferentialEvolution(Algorithm):
    NAME = 'DE'

    def __init__(self, benchmark: Type[Benchmark], dimensions, F, CR, mutation_type):
        super().__init__(benchmark, dimensions)
        self.F = F
        self.CR = CR
        self.mutationType = mutation_type

    def restart(self):
        self.benchmark.restart()
        self.population = self._create_population()
        self.costs = [self.objective(el) for el in self.population]
        self.best_individual = self.population[np.argmin(self.costs)]
        self.best_individual_cost = np.min(self.costs)

    def evolve(self):
        children = np.array([self._get_child(active_element) for active_element in self.population])
        children_costs = np.array([self.objective(child) for child in children])
        mask = children_costs < self.costs
        self.costs = np.where(mask, children_costs, self.costs)
        self.population = np.where(mask[:, np.newaxis], children, self.population)
        self.best_individual = self.population[np.argmin(self.costs)]
        self.best_individual_cost = np.min(self.costs)

    def _create_mutant(self, active_element):
        if self.mutationType == "rand":
            return self._create_mutant_rand(active_element)
        if self.mutationType == "best":
            return self._create_mutant_best(active_element)

    def _create_mutant_rand(self, active_element):
        available_elements = len(self.population) - np.sum(self.population == active_element)
        if available_elements < 3:
            raise errors.StagnantPopulationError("Not enough unique elements in the population")
        while True:
            parents = self.population[np.random.choice(self.population.shape[0], size=3, replace=False)]
            if active_element not in parents:
                break
        mutant = parents[0] + self.F * (parents[1] - parents[2])
        mutant = reflect_bounds(mutant, self.lowerBound, self.upperBound)

        return mutant

    def _create_mutant_best(self, active_element):
        available_elements = len(self.population) - np.sum(self.population == active_element)
        if available_elements < 2:
            raise errors.StagnantPopulationError("Not enough unique elements in the population")
        while True:
            parents = self.population[np.random.choice(self.population.shape[0], size=2, replace=False)]
            if active_element not in parents:
                break
        mutant = self.best_individual + self.F * (parents[0] - parents[1])
        mutant = reflect_bounds(mutant, self.lowerBound, self.upperBound)

        return mutant

    def _get_child(self, active_element):
        mutant = self._create_mutant(active_element)
        thresholds = np.random.random(self.dimensions)
        child = np.where(thresholds < self.CR, mutant, active_element)

        return child

    def _create_population(self):
        return np.random.uniform(self.lowerBound, self.upperBound, size=(self.populationSize, self.dimensions))

        # individuals = []
        # for s in range(self.populationSize):
        #     individual = [random.uniform(self.lowerBound, self.upperBound) for _ in range(self.dimensions)]
        #     individuals.append(individual)
        # return individuals


class ParticleSwarm(Algorithm):
    NAME = "PSO"

    def __init__(self, benchmark: Type[Benchmark], dimensions):
        super().__init__(benchmark, dimensions)
        self.c1 = 1.49618
        self.c2 = 1.49618
        self.w = 0.7298

    def restart(self):
        self.benchmark.restart()
        self.population = self._create_population()
        self.velocity = np.zeros((self.populationSize, self.dimensions))
        self.costs = [self.objective(el) for el in self.population]
        self.best_individual = self.population[np.argmin(self.costs)]
        self.personal_best = self.population.copy()
        self.personal_best_costs = self.costs.copy()
        self.global_best_cost = np.argmin(self.costs)
        self.best_individual_cost = self.global_best_cost
        self.global_best = self.population[np.argmin(self.costs)]

    def evolve(self):
        for i in range(self.populationSize):
            current_position = self.population[i]
            current_velocity = self.velocity[i]
            current_cost = self.costs[i]
            personal_best = self.personal_best[i]
            personal_best_cost = self.personal_best_costs[i]
            r1 = np.random.rand()
            r2 = np.random.rand()
            term_1 = self.c1 * r1 * (personal_best - current_position)
            term_2 = self.c2 * r2 * (self.best_individual - current_position)
            new_velocity = self.w * current_velocity + term_1 + term_2
            new_position = current_position + new_velocity
            reflect_bounds(new_position, self.lowerBound, self.upperBound)
            new_cost = self.objective(new_position)
            if new_cost < personal_best_cost:
                self.personal_best[i] = new_position
                self.personal_best_costs[i] = new_cost
            if new_cost < self.global_best_cost:
                self.global_best = new_position
                self.global_best_cost = new_cost
                self.best_individual_cost = new_cost
            self.population[i] = new_position
            self.costs[i] = new_cost
            self.velocity[i] = new_velocity
        self.best_individual = self.global_best


class SelfOrganizingMigrationAO(Algorithm):
    NAME = "SOMA-AO"

    def __init__(self, benchmark: Type[Benchmark], dimensions):
        super().__init__(benchmark, dimensions)
        self.pathLength = 3
        self.stepSize = 0.11
        self.prt = 0.7

    def restart(self):
        self.benchmark.restart()
        self.best_individual = None
        self.best_individual_cost = np.inf
        self.population = self._create_population()
        self.costs = [self.objective(el) for el in self.population]
        self.best_individual_cost = np.min(self.costs)
        self.best_individual = self.population[np.argmin(self.costs)]

    def evolve(self):
        t = self.stepSize
        leader = self.best_individual
        next_costs = self.costs.copy()
        next_positions = self.population.copy()

        while t <= self.pathLength:
            random_vector = np.random.random(size=(self.populationSize, self.dimensions))
            prt_vector = np.where(random_vector < self.prt, 1, 0)
            new_position = self.population + (leader - self.population) * t * prt_vector
            new_position = reflect_bounds(new_position, self.lowerBound, self.upperBound)
            new_costs = [self.objective(el) for el in new_position]

            for i in range(self.populationSize):
                if new_costs[i] < next_costs[i]:
                    next_costs[i] = new_costs[i]
                    next_positions[i] = new_position[i]

            t += self.stepSize

        self.costs = next_costs
        self.population = next_positions
        self.best_individual_cost = np.min(self.costs)
        self.best_individual = self.population[np.argmin(self.costs)]


class SelfOrganizingMigrationAA(Algorithm):
    NAME = "SOMA-AA"

    def __init__(self, benchmark: Type[Benchmark], dimensions):
        super().__init__(benchmark, dimensions)
        self.pathLength = 3
        self.stepSize = 0.11
        self.prt = 0.7

    def restart(self):
        self.benchmark.restart()
        self.best_individual = None
        self.best_individual_cost = np.inf
        self.population = self._create_population()
        self.costs = [self.objective(el) for el in self.population]
        self.best_individual_cost = np.min(self.costs)
        self.best_individual = self.population[np.argmin(self.costs)]

    def evolve(self):
        next_costs = self.costs.copy()
        next_positions = self.population.copy()

        for j in range(self.populationSize):
            for i in range(self.populationSize):
                leader = self.population[i]
                if i == j:
                    # interaction with oneself
                    continue
                t = self.stepSize
                while t < self.pathLength:
                    random_vector = np.random.random(size=self.dimensions)
                    prt_vector = np.where(random_vector < self.prt, 1, 0)
                    new_position = self.population[j] + (leader - self.population[j]) * t * prt_vector
                    new_position = reflect_bounds(new_position, self.lowerBound, self.upperBound)
                    new_cost = self.objective(new_position)

                    if new_cost < next_costs[j]:
                        next_costs[j] = new_cost
                        next_positions[j] = new_position

                    t += self.stepSize

        self.costs = next_costs
        self.population = next_positions
        self.best_individual_cost = np.min(self.costs)
        self.best_individual = self.population[np.argmin(self.costs)]


def reflect_bounds(array, lower_bound, upper_bound):
    low_mask = array < lower_bound
    high_mask = array > upper_bound
    while np.any(low_mask) or np.any(high_mask):
        array[low_mask] = (
                2 * lower_bound - array[low_mask]
        )
        array[high_mask] = (
                2 * upper_bound - array[high_mask]
        )
        low_mask = array < lower_bound
        high_mask = array > upper_bound

    return array
