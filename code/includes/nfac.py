import numpy as np
from MLP2 import MLP
import random

class NFAC():
  a_max = None
  a_min = None
  a_delta = None

  D = []

  def __init__(self, a_dim, a_min, a_max, state_size, random_chance = 0.1, discount = 0.99):
    self.a_max = a_max
    self.a_min = a_min
    self.action_size = a_dim
    self.state_size = state_size
    self.action_mlp = MLP(self.state_size, 1, [5], self.action_size)
    self.value_mlp = MLP(self.state_size, 1, [5], 1)
    self.discount = discount
    self.random_chance = random_chance
    self.sd = 1
    #choose a high sigma to be able to reach goal, 
    #due to mlp initialization it can be impossible otherwise
    self.sigma = 10
    self.learning_rate = 0.01

  #Draw a value from a univariate normal dist
  def getExplorationAction(self, state):
    rand_sample = np.random.normal(loc=0.0, scale=self.sd, size=self.action_size)
    #print "R: " + str( rand_sample)
    b_action = self.getAction(state)
    #print "B: " + str( b_action)
    action = b_action + self.sigma * rand_sample
    return action

  def getActorBrain(self):
    return self.action_mlp.getBrain()

  def setActorBrain(self, brain):
    self.action_mlp.setBrain(brain)

  def getCriticBrain(self):
    return self.value_mlp.getBrain()

  def setCriticBrain(self, brain):
    self.value_mlp.setBrain(brain)
 
  def getAction(self, state):
    action = self.action_mlp.process(state)
    return action
 
  def getQ(self, state):
    q = self.value_mlp.process(state)
    return q

  def updateQ(self, inp, target):
    self.value_mlp.train(inp, target, self.learning_rate)

  def updateActor(self, inp, target):
    self.action_mlp.train(inp, target, self.learning_rate)

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

    #print "A: " + str(action)

    return action

  def adjustSigma(self):
    self.sigma = self.sigma * 0.99

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
    #print action

    return action

  def clearCollection(self):
    self.D = []

  # should be updated to work with RPROP, batch prop
  def update(self):

    #old state, best state, expected state, reward, new state, goal
    for o, b, e, r, n, g in self.D:
      old_Q = self.getQ(o) 

      if g:
	value = r
      else:
        value = r + self.discount * self.getQ(n)

      # first update actor as described in paper since it is dependent on Q value. 
      if value > old_Q:
        self.updateActor(o, e)
      else:
	# update according to s (old_state) and u (best_action) instead of s and a (old_action)
        self.updateActor(o, b)

    # after modification of the actor, update critic similarly on (s, vk) learning base
    for o, b, e, r, n, g in self.D:

      if g:
	value = r
      else:
        value = r + self.discount * self.getQ(n)

      self.updateQ(o, value)
   
    # after modification of the actor, empty the learning base
    self.clearCollection()
      

  def collect(self, old_state, old_action, reward, new_state, goal):
    # Store tuples of the form (s, u, a, r, s') as described in section 3. 
    # Note we also store a flag if new state is goal at the end

    best_action = self.getClampedBestAction(old_state)
    self.D.append([old_state, best_action, old_action, reward, new_state, goal])








