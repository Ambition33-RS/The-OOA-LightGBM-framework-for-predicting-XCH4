import numpy as np
import random
import copy

class OspreyOptimizer:
    """
    Osprey Optimization Algorithm
    """
    def __init__(self, pop_size, dim, lb, ub, max_iter, objective_function, param_types=None, param_names=None):
        self.pop_size = pop_size
        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.max_iter = max_iter
        self.obj_func = objective_function
        self.param_types = param_types if param_types else {}
        self.param_names = param_names if param_names else [f'param_{i}' for i in range(dim)]

    def _initialize_population(self):
        """
        Initialize population positions, uniformly distributed within the boundaries.
        """
        population = np.random.rand(self.pop_size, self.dim) * (self.ub - self.lb) + self.lb
        return population

    def _convert_params(self, params):
        """
        Convert parameters to the specified type based on param_types.
        """
        params_converted = params.copy()
        for idx, t in self.param_types.items():
            params_converted[idx] = t(round(params_converted[idx]))
        return params_converted

    def _calculate_fitness(self, population):
        """
        Calculate the fitness of the entire population, where fitness is the objective function value.
        """
        fitness = []
        for individual in population:
            indiv_converted = self._convert_params(individual)
            fit = self.obj_func(indiv_converted)
            fitness.append(fit)
        return np.array(fitness).reshape(-1, 1)

    def _clip_population(self, population):
        """
        Ensure all parameters remain within the specified limits.
        """
        return np.clip(population, self.lb, self.ub)

    def _sort_by_fitness(self, population, fitness):
        """
        Sort the population by fitness (ascending order)
        """
        sorted_indices = np.argsort(fitness[:, 0])
        return population[sorted_indices], fitness[sorted_indices]

    def optimize(self):
        """
        Execution Optimization Process
        :return: Optimal fitness, optimal parameters, convergence curve (changes in optimal fitness during iteration)
        """
        population = self._initialize_population()
        fitness = self._calculate_fitness(population)
        population, fitness = self._sort_by_fitness(population, fitness)

        gbest_score = copy.deepcopy(fitness[0])
        gbest_position = copy.deepcopy(population[0:1])
        curve = np.zeros((self.max_iter, 1))

        for t in range(self.max_iter):
            for i in range(self.pop_size):
                # Select prey location
                if i < 2 or random.random() < 0.5:
                    selected_fish = copy.deepcopy(gbest_position[0])
                else:
                    k = random.randint(0, i - 1)
                    selected_fish = copy.deepcopy(population[k])

                # Phase 1: Dive Fishing
                I = random.randint(1, 2)
                r1 = random.random()
                new_position = population[i] + r1 * (selected_fish - I * population[i])
                new_position = np.clip(new_position, self.lb, self.ub)

                new_position_converted = self._convert_params(new_position)
                fit_new = self.obj_func(new_position_converted)
                if fit_new < fitness[i]:
                    fitness[i] = fit_new
                    population[i] = new_position

                # Phase 2: Local search
                # Adaptive Random Search
                r2 = random.random()
                search_range = (self.ub - self.lb) * np.exp(-2 * t / self.max_iter)  # Search scope decreasing with each iteration
                random_direction = 2 * np.random.random(self.dim) - 1  # Random direction within the range [-1, 1]
                new_position = population[i] + r2 * search_range * random_direction
                new_position = np.clip(new_position, self.lb, self.ub)

                new_position_converted = self._convert_params(new_position)
                fit_new = self.obj_func(new_position_converted)
                if fit_new < fitness[i]:
                    fitness[i] = fit_new
                    population[i] = new_position

            population = self._clip_population(population)
            fitness = self._calculate_fitness(population)
            population, fitness = self._sort_by_fitness(population, fitness)

            # Update Global Optimal
            if fitness[0] <= gbest_score:
                gbest_score = copy.deepcopy(fitness[0])
                gbest_position = copy.deepcopy(population[0:1])

            # Print
            params_print = []
            for i, name in enumerate(self.param_names):
                val = gbest_position[0, i]
                if i in self.param_types:
                    val_print = self.param_types[i](round(val))
                else:
                    val_print = round(val, 5)
                params_print.append(f"{name}={val_print}")

            print(f"第 {t + 1} 次迭代：最佳参数 = {', '.join(params_print)}，最佳适应度 = {gbest_score[0]}")

            curve[t] = gbest_score

        best_params = self._convert_params(gbest_position[0])
        return gbest_score, best_params, curve