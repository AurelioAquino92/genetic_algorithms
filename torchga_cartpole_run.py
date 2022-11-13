from numpy import argmax, append, array, load
from gym import make
from torch import nn, tensor
from pygad.torchga import predict

env_run = make('CartPole-v1', render_mode='human')
model = nn.Sequential(  nn.Linear(12, 13),
                        nn.ReLU(),
                        nn.Linear(13, 2))
solution = load('torchmodel.npy')

for episode in range(5):
    obs, _ = env_run.reset()
    inputs = append(obs, [obs, obs])
    done = False
    total_reward = 0
    while not done:
        prediction = predict(model, solution, tensor(array(inputs)))
        action = argmax(prediction.detach().numpy()) 
        print(action)
        obs, rewards, done, info, _ = env_run.step(action)
        inputs = append(obs, inputs[:-4])
        env_run.render()
        total_reward += rewards
    print("Run: {run}   Reward: {rew}".format(run=episode+1, rew=total_reward))