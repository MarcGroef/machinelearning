import numpy as np
from MLP2 import MLP
import random

class Cacla():
  a_max = None
  a_min = None
  a_delta = None

  def __init__(self, a_dim, a_min, a_max, state_size, random_chance = 0.1, discount = 0.99):
    self.a_max = a_max
    self.a_min = a_min
    self.action_size = a_dim
    self.state_size = state_size
    self.action_mlp = MLP(self.state_size, 1, [100], 1)
    self.value_mlp = MLP(self.state_size, 1, [100], 1)
    self.discount = discount
    self.random_chance = random_chance
    self.sd = 1
    self.sigma = 0.5

  #Draw a value from a univariate normal dist
  def getExplorationAction(self, state):
    rand_sample = np.random.normal(loc=0.0, scale=self.sd, size=self.action_size)
    action = self.getAction(state) + self.sigma * rand_sample
    return action

  def getBrain(self):
    return mlp.getBrain()

  def setBrain(self, brain):
    mlp.setBrain(brain)
 
  def getAction(self, state):
    action = self.action_mlp.process(state)
    return action
 
  def getQ(self, state):
    q = self.value_mlp.process(state)
    return q

  def updateQ(self, inp, target):
    self.value_mlp.train(inp, target, 0.01)

  def updateActor(self, inp, target):
    self.action_mlp.train(inp, target, 0.01)

  def epsilonGreedy(self):
    action = []

    for idx in range(self.action_size):
      action.append(random.uniform(self.a_min[idx], self.a_max[idx]))

    return action

  def getClampedBestAction(self, s):
    action = self.getAction(s)

    for idx in range(self.action_size):
      action[idx] = max(min(self.a_max[idx], action[idx]), self.a_min[idx])

    return action

  def getClampedGaussianAction(self, s):
    action = self.getExplorationAction(s)

    for idx in range(self.action_size):
      action[idx] = max(min(self.a_max[idx], action[idx]), self.a_min[idx])

    return action

  def adjustSigma(self):
    self.sigma = self.sigma * 0.98

  def epsilonStrategy(self, s):
    if (random.random() < self.random_chance):
      action = self.epsilonGreedy()
    else:
      action = self.getClampedBestAction(s)
    return action
 
  def chooseAction(self, s):
    # generate action = best action + random sample of normal distribution

    #action = self.epsilonStrategy(s)
    action = self.getClampedGaussianAction(s)

    return action

  def update(self, old_state, old_action, new_state, reward, goal):

    old_Q = self.getQ(old_state) 

    if goal:
	value = reward
    else:
        value = reward + self.discount * self.getQ(new_state)

    self.updateQ(old_state, value)

    if value > old_Q:
        self.updateActor(old_state, old_action)
    


