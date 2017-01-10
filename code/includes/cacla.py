import numpy as np
from MLP2 import MLP
import random

class Cacla():
  a_max = None
  a_min = None
  a_delta = None

  def __init__(self, a_dim, a_min, a_max, state_size, random_chance = 0.1, learningRate = 0.01, discount = 0.99):
    self.a_max = np.asarray(a_max)
    self.a_min = np.asarray(a_min)
    self.action_size = a_dim
    self.state_size = state_size
    self.action_mlp = MLP(self.state_size, 1, [20], 1)
    self.value_mlp = MLP(self.state_size, 1, [20], 1)
    self.learningRate = learningRate
    self.discount = discount
    self.random_chance = random_chance
    self.sigma = 1

  #Draw a value from a univariate normal dist
  def getExplorationAction(self, state, sigma):
    rand_sample = np.random.normal(loc=0.0, scale=self.sigma, size=1)
    action = self.getAction(state) + sigma * rand_sample
    return action

  def getBrain(self):
    return mlp.getBrain()

  def setBrain(self, brain):
    mlp.setBrain(brain)
 
  def getAction(self, state):
    action = self.action_mlp.process(state)
    #print action
    return action
 
  def getQ(self, state):
    q = self.value_mlp.process(state)
    return q

  def updateQ(self, inp, target):
    self.value_mlp.train(inp, target, 0.01)

  def updateActor(self, inp, target):
    self.action_mlp.train(inp, target, 0.01)

  def epsilonGreedy(self):
    action = random.random() * 2 - 1
    action = [action]
    return action

  def getClampedBestAction(self, s):
    action = self.getAction(s)
    action = max(min(1, action[0]), -1)
    action = [action]
    return action

  def getClampedGaussianAction(self, s, sigma):
    action = self.getExplorationAction(s, sigma)
    action = max(min(1, action[0]), -1)
    action = [action]
    return action
 
  def chooseAction(self, s):
    # generate action = best action + random sample of normal distribution

    #if (random.random() < self.random_chance):
      #action = self.epsilonGreedy()
    #else:
      #action = self.getClampedBestAction(s)

    action = self.getClampedGaussianAction(s, 0.1)
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
    


