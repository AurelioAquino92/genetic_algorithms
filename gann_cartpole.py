import numpy
import pygad
from pygad.nn import predict
import pygad.gann
import gym

def fitness_func(solution, sol_idx):
    global GANN_instance, env

    solution_fitness = 0
    for _ in range(20):
        obs = env.reset()
        done = False
        acc_cart = 0
        acc_ang = 0
        while not done:
            data_input = numpy.array([numpy.append(obs, [acc_cart, acc_ang])])
            prediction = predict(last_layer=GANN_instance.population_networks[sol_idx],
                                   data_inputs=data_input, problem_type="regression")
            action = numpy.argmax(prediction)
            vel_cart = obs[1].copy()
            vel_ang = obs[3].copy()
            obs, reward, done, _ = env.step(action)
            env.render()
            acc_cart = vel_cart - obs[1]
            acc_ang = vel_ang - obs[3]
            solution_fitness += reward - numpy.abs(obs[0]) / 100 - numpy.abs(obs[2])
        
    return solution_fitness

def callback_generation(ga_instance):
    global GANN_instance, last_fitness

    population_matrices = pygad.gann.population_as_matrices(population_networks=GANN_instance.population_networks,
                                                            population_vectors=ga_instance.population)

    GANN_instance.update_population_trained_weights(population_trained_weights=population_matrices)

    fitness = ga_instance.best_solution()[1].copy()
    if fitness > last_fitness:
        print("Generation = {generation}".format(generation=ga_instance.generations_completed))
        print("Fitness    = {fitness}".format(fitness=fitness))
        print("Change     = {change}".format(change=fitness - last_fitness))
        ga_instance.save('ganncartpole')
        # obs = env.reset()
        # done = False
        # while not done:
        #     prediction = predict(last_layer=GANN_instance.population_networks[0],
        #                            data_inputs=numpy.array([obs]), problem_type="regression")
        #     obs, reward, done, info = env.step(numpy.argmax(prediction))
        #     env.render()
    else:
        print("No improvement. Generation = {generation} Fitness = {fitness}".format(generation=ga_instance.generations_completed, fitness=fitness))

    last_fitness = fitness

env = gym.make('CartPole-v1')
last_fitness = 0

num_solutions = 50 # A solution or a network can be used interchangeably.
GANN_instance = pygad.gann.GANN(num_solutions=num_solutions,
                                num_neurons_input=6,
                                num_neurons_hidden_layers=[10, 5],
                                num_neurons_output=2,
                                hidden_activations=["relu", "relu"],
                                output_activation="relu")

if __name__ == "__main__":
    population_vectors = pygad.gann.population_as_vectors(population_networks=GANN_instance.population_networks)

    initial_population = population_vectors.copy()

    num_parents_mating = 40 # Number of solutions to be selected as parents in the mating pool.
    num_generations = 500 # Number of generations.
    mutation_percent_genes = 25 # Percentage of genes to mutate. This parameter has no action if the parameter mutation_num_genes exists.
    parent_selection_type = "sss" # Type of parent selection.
    crossover_type = "single_point" # Type of the crossover operator.
    mutation_type = "random" # Type of the mutation operator.
    keep_parents = 2 # Number of parents to keep in the next population. -1 means keep all parents and 0 means keep nothing.

    init_range_low = -1
    init_range_high = 1

    ga_instance = pygad.GA(num_generations=num_generations,
                        num_parents_mating=num_parents_mating,
                        initial_population=initial_population,
                        fitness_func=fitness_func,
                        mutation_percent_genes=mutation_percent_genes,
                        init_range_low=init_range_low,
                        init_range_high=init_range_high,
                        parent_selection_type=parent_selection_type,
                        crossover_type=crossover_type,
                        mutation_type=mutation_type,
                        keep_parents=keep_parents,
                        on_generation=callback_generation,
                        parallel_processing=['process', 1])

    # ga_instance = pygad.load('C:\\Users\\mnsaaqui\\Desktop\\python\\IA\\genetic_algorithms\\ganncartpole')
    # ga_instance.num_generations = 10
    ga_instance.run()
    ga_instance.plot_fitness()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
    print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))
    # print(numpy.argmax(numpy.array([[0, 0, 0, 0]])*solution))

    ga_instance.save('ganncartpole')