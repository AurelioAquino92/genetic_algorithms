from numpy import argmax, append, array
from pygad import load
from gym import make
from torch import nn, tensor
from pygad.torchga import predict
from torchga_cartpole import fitness_func, callback_generation

torchga_instance_run = load('C:\\Users\\mnsaaqui\\Desktop\\python\\IA\\genetic_algorithms\\torchgacartpole')
torchga_instance_run.parallel_processing = None
env_run = make('CartPole-v1')
model = nn.Sequential(  nn.Linear(12, 20),
                        nn.ReLU(),
                        nn.Linear(20, 9),
                        nn.ReLU(),
                        nn.Linear(9, 2))
solution = torchga_instance_run.best_solution()[0]

for episode in range(5):
    obs = env_run.reset()
    inputs = append(obs, [obs, obs])
    done = False
    total_reward = 0
    while not done:
        prediction = predict(model, solution, tensor(array(inputs)))
        action = argmax(prediction.detach().numpy()) 
        obs, rewards, done, info = env_run.step(action)
        inputs = append(obs, inputs[4:])
        env_run.render()
        total_reward += rewards
    print("Run: {run}   Reward: {rew}".format(run=episode+1, rew=total_reward))