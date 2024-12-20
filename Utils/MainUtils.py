# Based on https://github.com/sfujim/TD3/blob/master/utils.py

import numpy as np
import torch

from itertools import product

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=1e6, device="cpu"):
        max_size = int(max_size)
        self.states = np.zeros(shape=(max_size, state_dim))
        self.actions = np.zeros(shape=(max_size, action_dim))
        self.rewards = np.zeros(shape=max_size)
        self.next_states = np.zeros(shape=(max_size, state_dim))
        self.dones = np.zeros(shape=max_size)

        self.entry = 0
        self.size = 0
        self.max_size = max_size
        self.device = device

    def add(self, state, action, reward, next_state, done):
        self.states[self.entry] = state
        self.actions[self.entry] = action
        self.rewards[self.entry] = reward
        self.next_states[self.entry] = next_state
        self.dones[self.entry] = done

        self.entry = (self.entry + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.states[indices]).to(self.device),
            torch.FloatTensor(self.actions[indices]).to(self.device),
            torch.FloatTensor(self.rewards[indices]).to(self.device),
            torch.FloatTensor(self.next_states[indices]).to(self.device),
            torch.FloatTensor(self.dones[indices]).to(self.device)
        )


class ReplayBuffer_Atomic(object):
    def __init__(self, state_dim, max_size=1e6, device="cpu"):
        max_size = int(max_size)
        self.states = np.zeros(shape=(max_size, state_dim))
        self.actions = np.zeros(shape=max_size)
        self.rewards = np.zeros(shape=max_size)
        self.next_states = np.zeros(shape=(max_size, state_dim))
        self.dones = np.zeros(shape=max_size)

        self.entry = 0
        self.size = 0
        self.max_size = max_size
        self.device = device

    def add(self, state, action, reward, next_state, done):
        self.states[self.entry] = state
        self.actions[self.entry] = action
        self.rewards[self.entry] = reward
        self.next_states[self.entry] = next_state
        self.dones[self.entry] = done

        self.entry = (self.entry + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.states[indices]).to(self.device),
            torch.FloatTensor(self.actions[indices]).reshape(-1, 1).long().to(self.device),
            torch.FloatTensor(self.rewards[indices]).to(self.device),
            torch.FloatTensor(self.next_states[indices]).to(self.device),
            torch.FloatTensor(self.dones[indices]).to(self.device)
        )


class FactoredToDiscreteMapping:
    def __init__(self, env, bin_size=3):
        """
        Given a DMSuiteWrapper environment, use the `get_atomic_action` on a factored action of the same environment to get the
        atomic action mapping. Note that the factored action must come from the CompositeDiscreteWrapper but this class only
        accepts the DMSuiteWrapper instance of an environment.

        :param env: DMSuiteWrapper
        :param bin_size: int
        """
        self.num_subaction_spaces = env.action_space.shape[0]
        self.bin_size = bin_size
        lows = env.action_space.low
        highs = env.action_space.high
        self.factored_action_lookup = {}  # dict which maps factored action to continuous action
        for a, l, h in zip(range(self.num_subaction_spaces), lows, highs):
            self.factored_action_lookup[a] = {}
            bins = np.linspace(l, h, bin_size)
            for count, b in enumerate(bins):
                self.factored_action_lookup[a][count] = b

        self.discrete_action_lookups = {}
        bins = []
        for low, high in zip(lows, highs):
            bins.append(np.linspace(low, high, bin_size).tolist())
        for count, action in enumerate(product(*bins)):
            self.discrete_action_lookups[tuple(action)] = count

    def get_continuous_action(self, action):
        continuous_action = []
        for action_id, a in enumerate(action):
            continuous_action.append(self.factored_action_lookup[action_id][a])
        return continuous_action

    def get_atomic_action(self, factored_action):
        continuous_action = self.get_continuous_action(factored_action)
        return self.discrete_action_lookups[tuple(continuous_action)]
