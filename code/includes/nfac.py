import numpy as np
from MLP import MLP
import random

class NFAC():
  a_max = None
  a_min = None
  a_delta = None
 
  D = []

  def __init__(self, a_dim, a_min, a_max, state_size, random_chance = 0.01, learningRate = 0.001, discount = 0.1):
    self.a_max = np.asarray(a_max)
    self.a_min = np.asarray(a_min)
    self.action_size = a_dim
    self.state_size = state_size
    self.action_mlp = MLP(self.state_size, 20, 1)
    self.value_mlp = MLP(self.state_size, 20, 1)
    self.learningRate = learningRate
    self.discount = discount
    self.random_chance = random_chance
    self.sigma = 1

  def getAction(self, state):
    action = self.action_mlp.process(state)
    return action

  def getExplorationAction(self, state, sigma):
    action = self.getAction(state) + sigma * np.random.normal(loc=0.0, scale=self.sigma, size=1)
    return action
 
  def getQ(self, state):
    q = self.value_mlp.process(state)
    return q

  def updateQ(self, inp, des):
    self.value_mlp.train(inp, des, 0.01)

  def updateActor(self, state, action):
    self.action_mlp.train(state, action, 0.01)
 
  #Draw a value from a univariate normal dist or use epsilon greedy
  def chooseAction(self, s):
    action = self.getExplorationAction(s, 5)
    # clamp action value between -1 and 1
    action = max(min(1, action[0]), -1)
    action = [action]
    return action

  def clearCollection(self):
    self.D = []

  # should be updated to work with RPROP, batch prop
  def update(self):
    for o, b, e, r, n, g in self.D:
      old_Q = self.getQ(o) 

      if g:
	value = r
      else:
        value = r + self.discount * self.getQ(n)

      self.updateQ(o, value)

      if value > old_Q:
        self.updateActor(o, e)
      else:
	# update according to s (old_state) and u (best_action) instead of s and a (old_action)
        self.updateActor(o, b)

    self.clearCollection()
      

  def collect(self, old_state, old_action, reward, new_state, goal):
    # Store tuples of the form (s, u, a, r, s') as described in section 3. 
    # Note we also store a flag if new state is goal at the end

    best_action = self.getAction(old_state)
    self.D.append([old_state, best_action, old_action, reward, new_state, goal])





