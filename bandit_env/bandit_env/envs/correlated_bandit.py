import numpy as np
import random
import gym
from gym import error, spaces, utils
from gym.utils import seeding


class CorrelatedBanditEnv(gym.Env):
    def __init__(self, prob = None, info={}):
        self.n_arms = 2
        #bandits prob distribution #correlated bandits so for arm 1 'p' and for arm 2 '1-p'
        if(prob == None):
            self.arms_prob = np.random.rand(1)
            self.arms_prob = np.append(self.arms_prob, 1 - self.arms_prob)
        elif(type(prob)==float and prob >= 0.0 and prob <= 1.0):
            self.arms_prob = np.array([prob, 1-prob])
        else:
            raise Exception('Incorrect value for the probability for the arm 1')

        self.info = {'arms_prob': self.arms_prob, 'optimal_arm':np.argmax(self.arms_prob)}
        self.action_space = spaces.Discrete(self.n_arms)
        self.observation_space = spaces.box.Box(-1.0, 1.0, (1,), dtype=np.float32)

    def step(self, action):
        assert self.action_space.contains(action)
        if random.random() < self.arms_prob[action]:
            reward =  1.0
        else:
            reward = 0.0
        done = False

        return (np.array([0]), reward, done, self.info)

    def reset(self):
        return
    def render(self, mode='human', close=False):
        return