import numpy as np
from MLP import MLP
##https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2014-175.pdf
class Sarsa():
  a_max = None
  a_min = None
  a_delta = None



  def __init__(self, a_min, a_max, a_delta, state_size):
    self.a_max = a_max
    self.a_min = a_min
    self.a_delta = a_delta
    self.action_size = length(a_max)
    self.state_size = state_size
    self.mlp = MLP(self.action_size + self.state_size, 100, 1)
    self.max_iter = 1000
    self.learningRate = 0.01
    self.discount = 0.1

  def action_dist(self, a1, a2):  
    return np.linalg.norm(a1, a2)
 
  def getQ(self, state, action):
    return self.mlp.process(np.concatenate(state, action))

  def updateQ(self, action, state, currentOut, targetOut):
    self.mlp.train(np.concatenate(state, action), targetOut, 0.01)
 
  def chooseAction(self, s):
    a_best = self.a_min

    Q_best = self.getQ(s, a_best)

    action_range = self.action_dist(self.a_max, self.a_min)
    action_space = np.mgrid(self.a_min:self.a_max:self.a_delta)#.reshape(self.action_size, action_range/self.action_size)
    #This assumes a_delta to be constant between action dimensions
    for d_a0 in np.nditer(action_space, flags=['external loop'], order='F'):
       a = self.a_min + d_a0

       a_prev = inf ##a_max + 10????
       
       for _ in range(self.max_iter):
         
         a = a - (self.mlp.d_network() / self.mlp.dd_network())
         a = np.maximum(a, a_max) #keep in range
         a = np.minimum(a, a_min)
         Q = self.getQ(s, a)
e
         if (Q > Q_best):
           a_best = a
           Q_best = Q
         if(self.action_dist(a - a_prev) < 0.001):
           break
         a_prev = a
    return a_best

  def update(self, old_state, old_action, new_state, action_performed, reward):
    old_Q = self.getQ(old_state, old_action)
    diff = self.learningRate * (reward + self.discount * self.getQ(new_state, action_performed) - old_Q)
    target = self.old_Q + diff
    self.updateQ(action_performed, old_state, target)
    
   
