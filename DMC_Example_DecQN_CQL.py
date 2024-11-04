expert_score = 781.19
random_score = 121.37

# Imports
import sys
import gymnasium as gym
import torch
import numpy as np
import random
import pickle
import time

from Utils import MainUtils
from Algorithms import DecQN_CQL

# Load environment and dataset
env = MainUtils.DMSuiteWrapper(domain_name="quadruped", task_name="walk")
open_file = open("YourPathHere", "rb")
dataset = pickle.load(open_file)
open_file.close()

# Network and hyperparameters
device = "cuda:0"
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = env.action_space.high[0]
sub_action_dim = 3
num_critics = 2
alpha = 0.25
sub_actions = np.linspace(start=-max_action, stop=max_action, num=sub_action_dim)
memory_size = len(dataset)

print("Creating replay buffer...")
replay_buffer = MainUtils.ReplayBuffer(state_dim, action_dim, memory_size, device)
for k in range(memory_size):
    replay_buffer.add(dataset[k][0], dataset[k][1], dataset[k][2], dataset[k][3], dataset[k][4])
print("...replay buffer created!")
# Normalise states
mean = np.mean(replay_buffer.states, 0)
std = np.std(replay_buffer.states, 0) + 1e-3
replay_buffer.states = (replay_buffer.states - mean) / std
replay_buffer.next_states = (replay_buffer.next_states - mean) / std
dataset = [] # To save RAM

agent = DecQN_CQL.Agent(state_dim, action_dim, sub_action_dim, num_critics, device=device)

epochs = 200
iterations = 5000
grad_steps = 0
evals = 10
training_time = 0

for epoch in range(epochs):
    start_time = time.time()
    agent.train(replay_buffer, iterations, alpha)
    training_time += (time.time() - start_time)
    grad_steps += iterations

    # Policy evaluation
    scores = []
    scores_norm = []
    for e in range(evals):
        last_step = False
        state = env.reset()[0]
        score = 0
        while not last_step:
            with torch.no_grad():
                state = (state - mean) / std
                action = agent.choose_action(state)
                action_env = np.take(sub_actions, action)
                state, reward, done, last_step, info = env.step(action_env)
                score += reward
        score_norm = 100 * (score - random_score) / (expert_score - random_score)
        scores.append(score)
        scores_norm.append(score_norm)

    print("Grad steps", grad_steps,
          "Average Score Offline %.2f" % np.mean(scores),
          "Average Score Norm Offline %.2f" % np.mean(scores_norm))
