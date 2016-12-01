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


  def action_dist(self, a1, a2):
    pass
 
  def chooseAction(self, s):
    a_best = self.a_min
    Q_best = self.mlp.process(np.concatenate(s, a_best))
    ## for all discretized actions: a0
       a = a0
       a_prev = inf ##a_max + 10????
       
       for( _ in range(self.max_iter)):
         Q = self.mlp.process(np.concatenate(s, a))
         a = a - (self.mlp.d_network() / self.mlp.dd_network()
         if(Q > Q_best)):
           a_best = a
           Q_best = Q
         if(self.action_dist(a - a_prev) < 0.001):
           break
         a_prev = a
     return a_best
