import pygad
import numpy

def fitness_function(x, idx):
    if abs(x[0]) - abs(x[1]) < 1:
        return -1000
    fitness = 1.0 / numpy.sum(numpy.abs(x*x + 4*x - 5))
    return fitness

ga_instance = pygad.GA(num_generations=10000,
                       num_parents_mating=10,
                       fitness_func=fitness_function,
                       sol_per_pop=200,
                       num_genes=2,
                       init_range_low=-10,
                       init_range_high=10,
                       parent_selection_type="rank",
                       keep_parents=1,
                       crossover_type="uniform",
                       mutation_type="random",
                       mutation_percent_genes=100,
                       mutation_probability=0.1)

ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
# print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

# prediction = numpy.sum(numpy.array(function_inputs)*solution)
prediction = fitness_function(solution, 0)
print("Predicted output based on the best solution : {prediction}".format(prediction=prediction))