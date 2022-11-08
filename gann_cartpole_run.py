from numpy import argmax, array, append
from pygad import load
from gym import make
from pygad.nn import predict
from pygad.gann import GANN, population_as_matrices
from gann_cartpole import fitness_func, callback_generation

ga_instance_run = load('C:\\Users\\aurel\\Documents\\Projetos\\genetic_algorithms\\ganncartpole')
ga_instance_run.parallel_processing = None
env_run = make('CartPole-v1', render_mode="human")
GANN_instance2 = GANN(num_solutions=50,
                    num_neurons_input=6,
                    num_neurons_hidden_layers=[10, 5],
                    num_neurons_output=2,
                    hidden_activations=["relu", "relu"],
                    output_activation="relu")
population_matrices = population_as_matrices(GANN_instance2.population_networks, ga_instance_run.population)
GANN_instance2.update_population_trained_weights(population_matrices)
nn = GANN_instance2.population_networks[0]#ga_instance_run.best_solution()[2]]

for episode in range(5):
    obs, _ = env_run.reset()
    done = False
    acc_cart = 0
    acc_ang = 0
    total_reward = 0
    while not done:
        prediction = predict(nn, array([append(obs, [acc_cart, acc_ang])]), "regression")
        action = argmax(prediction)
        vel_cart = obs[1].copy()
        vel_ang = obs[3].copy()
        obs, rewards, done, info, _ = env_run.step(action)
        acc_cart = vel_cart - obs[1]
        acc_ang = vel_ang - obs[3]
        env_run.render()
        total_reward += rewards
    print("Run: {run}   Reward: {rew}".format(run=episode+1, rew=total_reward))