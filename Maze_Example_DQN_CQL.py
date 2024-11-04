expert_score = 99.22
random_score = -1.76

# Imports
import sys
import gymnasium as gym
import torch
import numpy as np
import random
import pickle
import time
from itertools import *

from Utils import MainUtils
from Algorithms import DQN_CQL
from lifelong_changing_actions_master.Environments.ToyMaze import Gridworld_CL

# Load environment
n_actions = 3
env = Gridworld_CL.Gridworld_CL(debug=True, n_actions=n_actions, change_interval=-1, change_count=1, max_episodes=100)

# Network and hyperparameters
device = "cuda:0"
state_dim = 2
action_dim = np.power(2, n_actions)
num_critics = 2
alpha = 0.25

# Load replay buffer
open_file = open("YourPathHere", "rb")
replay_buffer_factorised = pickle.load(open_file)
open_file.close()

fac_actions = np.array(list(product([0, 1], repeat=n_actions)))
print("Creating atomic replay buffer...")
replay_buffer = MainUtils.ReplayBuffer_Atomic(state_dim, replay_buffer_factorised.size)
for i in range(replay_buffer_factorised.size):
    replay_buffer.add(replay_buffer_factorised.states[i],
                      int(np.where(np.sum(replay_buffer_factorised.actions[i] == fac_actions, -1) == n_actions)[0]),
                      replay_buffer_factorised.rewards[i],
                      replay_buffer_factorised.next_states[i],
                      replay_buffer_factorised.dones[i])
    if i % 1000 == 0:
        print(i, "converted out of", replay_buffer_factorised.size)
print("... atomic replay buffer created!")
# Normalise states
mean = np.mean(replay_buffer.states, 0)
std = np.std(replay_buffer.states, 0) + 1e-3
replay_buffer.states = (replay_buffer.states - mean) / std
replay_buffer.next_states = (replay_buffer.next_states - mean) / std
# Put on device
replay_buffer.device = device

agent = DQN_CQL.Agent(state_dim, action_dim, num_critics, device=device)

epochs = 20
iterations = 5000
grad_steps = 0
evals = 100
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
        done = False
        state, valid_actions, flag = env.reset()
        score_eval = 0
        while not done:
            state = np.array(state)
            state = (state - mean) / std
            action = agent.choose_action(state)
            state, reward, done, _ = env.step(action)
            score_eval += reward
        score_norm = 100 * (score_eval - random_score) / (expert_score - random_score)
        scores.append(score_eval)
        scores_norm.append(score_norm)

    print("Grad steps", grad_steps,
          "Average Score Offline %.2f" % np.mean(scores),
          "Average Score Norm Offline %.2f" % np.mean(scores_norm))
