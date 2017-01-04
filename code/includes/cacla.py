import numpy as np
from MLP import MLP

class Cacla():
  a_max = None
  a_min = None
  a_delta = None

  def __init__(self, a_dim, a_min, a_max, state_size, random_chance = 0.01, learningRate = 0.999, discount = 0.1):
    self.a_max = np.asarray(a_max)
    self.a_min = np.asarray(a_min)
    self.action_size = a_dim
    self.state_size = state_size
    self.action_mlp = MLP(self.state_size, 100, 1)
    self.value_mlp = MLP(self.state_size, 100, 1)
    self.max_iter = 10
    self.learningRate = learningRate
    self.discount = discount
    self.random_chance = random_chance
    self.sigma = 1

  def getAction(self, state):
    print "getAction: " + str(state)
    print "------------------------"
    return self.action_mlp.process(state)
 
  def getQ(self, state):
    print "getQ: " + str(state)
    print "------------------------"
    return self.value_mlp.process(state)

  def updateQ(self, inp, des):
    self.value_mlp.train(inp, des, 0.02)

  def updateActor(self, state, action):
    self.action_mlp.train(state, action, 0.02)
 
  #Draw a value from a univariate normal dist
  def chooseAction(self, s):
    print "getAction: " + str(self.getAction(s))
    print "------------------------"
    action = self.getAction(s) + np.random.normal(loc=0.0, scale=self.sigma, size=1)
    # clamp action value between -1 and 1
    action = max(min(1, action), -1)
    return action

  def update(self, old_state, old_action, new_state, reward):
    
    old_Q = self.getQ(old_state)
    value = reward + self.discount * self.getQ(new_state)
    self.updateQ(old_state, value)
    td = value - old_Q
    if td > 0:
        self.updateActor(old_state, old_action)
