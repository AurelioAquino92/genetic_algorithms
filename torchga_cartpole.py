import numpy
import pygad
from torch import nn, tensor
from pygad.torchga import predict, TorchGA
import gym
from time import time

def fitness_func(solution, sol_idx):
    global torch_ga, model

    solution_fitness = 0
    for _ in range(1):
        obs, _ = env.reset()
        inputs = numpy.append(obs, [obs, obs])
        done = False
        while not done:
            data_input = tensor(numpy.array([inputs]))
            prediction = predict(model=model,
                                solution=solution,
                                data=data_input)
            action = numpy.argmax(prediction.detach().numpy())
            obs, reward, done, _, _ = env.step(action)
            inputs = numpy.append(obs, inputs[:-4])
            # env.render()
            solution_fitness += reward - numpy.abs(obs[0]) / 100 - numpy.abs(obs[2])
        
    return solution_fitness

def callback_generation(ga_instance):
    global last_fitness, tempo

    solution, fitness, _ = ga_instance.best_solution()
    if fitness > last_fitness:
        print("Generation = {generation}".format(generation=ga_instance.generations_completed))
        print("Fitness    = {fitness}".format(fitness=fitness))
        print("Change     = {change}".format(change=fitness - last_fitness))
        ga_instance.save('torchgacartpole')
        numpy.save('torchmodel.npy', solution)
        # obs = env.reset()
        # done = False
        # while not done:
        #     prediction = predict(last_layer=GANN_instance.population_networks[0],
        #                            data_inputs=numpy.array([obs]), problem_type="regression")
        #     obs, reward, done, info = env.step(numpy.argmax(prediction))
        #     env.render()
    else:
        print("No improvement. Generation = {generation} Fitness = {fitness}".format(generation=ga_instance.generations_completed, fitness=fitness))
    print('Tempo: ', time()-tempo)
    tempo = time()
    last_fitness = fitness

env = gym.make('CartPole-v1')
last_fitness = 0

model = nn.Sequential(  nn.Linear(12, 13),
                        nn.ReLU(),
                        nn.Linear(13, 2))
torch_ga = TorchGA(model=model, num_solutions=50)

if __name__ == "__main__":
    initial_population = torch_ga.population_weights

    ga_instance = pygad.GA( num_generations=500,
                            num_parents_mating=40,
                            initial_population=initial_population,
                            fitness_func=fitness_func,
                            mutation_percent_genes=25,
                            parent_selection_type="sss",
                            crossover_type="single_point",
                            mutation_type="random",
                            keep_parents=2,
                            on_generation=callback_generation,
                            parallel_processing=['process', 1])

    # ga_instance = pygad.load('torchgacartpole')
    # ga_instance.num_generations = 10
    tempo = time()
    ga_instance.run()
    ga_instance.plot_fitness()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
    print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))
    
    ga_instance.save('torchgacartpole')