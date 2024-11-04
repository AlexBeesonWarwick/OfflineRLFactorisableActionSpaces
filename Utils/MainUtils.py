# Based on https://github.com/sfujim/TD3/blob/master/utils.py

import numpy as np
import torch

import dm_control.suite as suite
import gymnasium as gym
from gymnasium import spaces
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

class DMSuiteWrapper(gym.Env):
    def __init__(self, domain_name, task_name, episode_len=None, seed=None):
        if seed is not None:
            self.env = suite.load(domain_name, task_name, task_kwargs={'random': seed})
        else:
            self.env = suite.load(domain_name, task_name)
        num_actions = self.env.action_spec().shape[0]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(num_actions,))
        # Calculate the size of the state space
        time_step = self.env.reset()
        state_size = np.concatenate([v.flatten() for v in time_step.observation.values()]).shape[0]
        obs_high = np.array([np.inf for _ in range(state_size)], dtype=np.float32)
        obs_low = -obs_high
        self.observation_space = spaces.Box(obs_low, obs_high)
        self.episode_len = episode_len or self.env._step_limit
        self._time_step = None

    def reset(self, seed=None, options=None):
        self._time_step = self.env.reset()
        return np.concatenate([v.flatten() for v in self._time_step.observation.values()]), {}

    def step(self, action):
        self._time_step = self.env.step(action)
        observation, reward, termination, info = (
            np.concatenate([v.flatten() for v in self._time_step.observation.values()]),
            self._time_step.reward,
            self._time_step.last(),
            {}
        )
        if self._time_step.last():
            info['truncated'] = not self._time_step.step_type.last()
        return observation, reward, False, self._time_step.last(), info

    def render(self, mode='human'):
        pass

    def close(self):
        pass

class AtomicDiscreteWrapper(gym.Wrapper):
    def __init__(self, env, bin_size=3):
        super(AtomicDiscreteWrapper, self).__init__(env)
        lows = self.env.action_space.low
        highs = self.env.action_space.high
        self.action_lookups = {}
        bins = []
        for low, high in zip(lows, highs):
            bins.append(np.linspace(low, high, bin_size).tolist())
        for count, action in enumerate(product(*bins)):
            self.action_lookups[count] = list(action)
        self.num_actions = len(self.action_lookups)
        self.action_space = spaces.Discrete(self.num_actions)

    def step(self, action):
        action = self.get_continuous_action(action)
        return super().step(action)

    def get_continuous_action(self, action):
        continuous_action = self.action_lookups[action]
        return continuous_action


class CompositeDiscreteWrapper(gym.Wrapper):
    def __init__(self, env, bin_size=3):
        super(CompositeDiscreteWrapper, self).__init__(env)
        self.num_subaction_spaces = self.env.action_space.shape[0]
        self.bin_size = bin_size
        lows = self.env.action_space.low
        highs = self.env.action_space.high
        self.action_lookups = {}
        for a, l, h in zip(range(self.num_subaction_spaces), lows, highs):
            self.action_lookups[a] = {}
            bins = np.linspace(l, h, bin_size)
            for count, b in enumerate(bins):
                self.action_lookups[a][count] = b
        self.action_space = spaces.MultiDiscrete([self.bin_size] * self.num_subaction_spaces)

    def step(self, action):
        action = self.get_continuous_action(action)
        return super().step(action)

    def get_continuous_action(self, action):
        continuous_action = []
        for action_id, a in enumerate(action):
            continuous_action.append(self.action_lookups[action_id][a])
        return continuous_action


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
