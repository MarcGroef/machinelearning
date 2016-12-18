import numpy as np
from MLP import MLP
##https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2014-175.pdf
class Sarsa():
  a_max = None
  a_min = None
  a_delta = None



  def __init__(self, a_min, a_max, a_delta, state_size, random_chance = 0.01, learningRate = 0.999, discount = 0.1):
    self.a_max = np.asarray([a_max])
    self.a_min = np.asarray([a_min])
    self.a_delta = np.asarray([a_delta])
    self.action_size = 1
    self.state_size = state_size
    self.mlp = MLP(self.action_size + self.state_size, 100, 1)
    self.max_iter = 10
    self.learningRate = learningRate
    self.discount = discount
    self.random_chance = random_chance

  def action_dist(self, a1, a2):  
    return np.linalg.norm(a1, a2)
 
  def getQ(self, state, action):
    #print np.asarray(state, dtype = np.float32)
    #print np.asarray([action], dtype = np.float32)
    mlpvec = np.concatenate([state, action])
    #print mlpvec
    return self.mlp.process(mlpvec)

  def updateQ(self, action, state, targetOut):
    self.mlp.train(np.concatenate([state, action]), targetOut, 0.02)
 
  def chooseAction(self, s):
    a_best = self.a_min

    Q_best = -1000000

    #action_range = self.action_dist(self.a_max, self.a_min)
    #action_space = np.mgrid[self.a_min:self.a_max:self.a_delta,self.a_min:self.a_max:self.a_delta]#.reshape(self.action_size, action_range/self.action_size)
    #action_space = np.mgrid[self.a_min:self.a_max:self.a_delta]
    #print action_space
    #This assumes a_delta to be constant between action dimensions
    #for a_x, a_y in np.nditer(action_space, flags=['external_loop'], order='F'):
    #for a in action_space:#np.nditer(action_space, flags=['external_loop'], order='F'):
    #for a in [-1, 1]
    if(np.random.rand(1) > (1 - self.random_chance)):
	rand_act = (np.random.rand(1) - 0.5) * 2
	#print "radom action: " +str(rand_act)
	return [np.asarray(rand_act), self.getQ(s, rand_act)]

    for a in range(-10, 10, 1):
       a /= 10.0
       a = np.asarray([a], dtype = np.float32)
       #print a
      ##Newtons method, to be added later..
       #print a
       #for _ in range(1):#range(self.max_iter):
#a += 0.1
#a = a - (self.mlp.d_network() / self.mlp.dd_network())
#a += np.random.rand(1) - 0.5
       a = np.minimum(a, self.a_max) #keep in range
       a = np.maximum(a, self.a_min)
       Q = self.getQ(s, a)
#print [Q_best, a_best]
       if (Q > Q_best):
         a_best = a
         Q_best = Q
	   
    return [a_best, Q_best]

  def update(self, old_state, old_action, new_state, action_performed, reward):
    old_Q = self.getQ(old_state, old_action)
    diff = self.learningRate * (reward + self.discount * self.getQ(new_state, action_performed) - old_Q)
    
    #print [old_action, old_Q, diff]
    target = old_Q + diff
    self.updateQ(action_performed, old_state, target)
    
   
