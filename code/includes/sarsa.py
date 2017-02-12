import numpy as np
from MLP2 import MLP
import sys
import matplotlib.pyplot as plt  ##sudo apt-get install python-matplotlib
#from sklearn.neural_network import MLPClassifier

##https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2014-175.pdf
class Sarsa():
  a_max = None
  a_min = None
  a_delta = None



  def __init__(self, a_dim, a_min, a_max, a_delta, state_size, random_chance = 1, learningRate = 0.01, discount = 0.8, discretize_state = False):
    self.a_max = np.asarray(a_max)
    self.a_min = np.asarray(a_min)
    self.a_delta = np.asarray(a_delta)
    self.action_size = a_dim
    self.action_space = self.define_action_space(a_dim, a_min, a_max, a_delta)

    self.state_size = state_size
    self.discretize_state = discretize_state
    self.nPositionBins = 10
    if(discretize_state):
      self.mlp = MLP(self.nPositionBins * self.state_size + self.action_size,1, [200, 20, 20,20,20], 1)
    else:
      self.mlp = MLP(self.action_size + self.state_size,1, [50, 10, 100,100,20], 1)
    self.max_iter = 10
    self.learningRate = learningRate
    self.discount = discount
    self.random_chance = random_chance
    np.random.seed()
    self.last_100 = []

  def discretizeState(self, state):

    discr = np.zeros(self.nPositionBins * self.state_size)
    
    for idx in range(self.state_size):
      tarIdx =int( (state[idx] + 1.5) / 3 * self.nPositionBins)
      #print [idx,tarIdx]
      try:
        discr[self.nPositionBins * idx + tarIdx] = 1
        if tarIdx + 1 < self.state_size:
          discr[self.nPositionBins * idx + tarIdx + 1] = 0.5
        if tarIdx > 0:
          discr[self.nPositionBins * idx + tarIdx - 1] = 0.5
      except IndexError:
        print "state idx out of bounds... + STATE[IDX] = " + str(state[idx])
    #print discr
    return discr
  
  def resetBrainBuffers(self):
    self.mlp.resetBuffers()

  def getBrain(self):
    return self.mlp.getBrain()

  def setBrain(self, brain):
    self.mlp.setBrain(brain)   

  def action_dist(self, a1, a2):  
    return np.linalg.norm(a1, a2)
 
  def getQ(self, state, action):
    
    
    if(self.discretize_state):
       state = self.discretizeState(state)
    mlpvec = np.concatenate([state, action])
    return self.mlp.process(mlpvec)

  def updateQ(self, action, state, targetOut):
    #self.mlp2.fit(np.concatenate([state, action]), targetOut)
    if(self.discretize_state):
       state = self.discretizeState(state)
    self.mlp.train(np.concatenate([state, action]), targetOut, self.learningRate, 0)

  def define_action_space(self, dim, min, max, delta):
    dim_lst = []

    for i in range(0, dim):
      dim_lst.append(np.arange(min[i], max[i] + delta[i], delta[i]))
    print dim_lst
    return np.array(np.meshgrid(*dim_lst))

  def printValueMap(self, action):
    posres = 10
    valMap = np.random.random((posres, posres))
    for pos in range(posres):
       for vel in range(posres):
          position = float(2 * pos) / posres - 1
          velocity = float(2 * vel) / posres - 1
          Q =  self.getQ(np.asarray([position, velocity]), np.asarray([action]))  #x, y = pos, vel
          #Q =  self.getQ(np.asarray([0.2, velocity]), np.asarray([position])) ##x, y = action, vel @ pos = -0.2
          #Q =  self.getQ(np.asarray([position, 0]), np.asarray([velocity]))  ##x, y = action, pos @ v = 0
          valMap[vel][pos] = Q

    
    plt.imshow(valMap, cmap = 'hot')#, interpolation='nearest')
    plt.show()
    plt.pause(0.001)

  def chooseAction(self, s):
    
    a_best = self.a_min
    Q_best = self.getQ(s, a_best)
    #print self.discretizeState(s)
    #    print(args)
    
    if(np.random.rand(1)[0] > (1 - self.random_chance)):  ##random action
	#rand_act = (np.random.rand(1) - 0.5) * 2
        rand_act = []
        for i in range(0,self.action_size):
            #rand_act.append(np.random.choice(np.arange(self.a_min[i],self.a_max[i],self.a_delta[i])))
            rand_act.append((np.random.rand(1)[0] - 0.5) * 2)
	#print "radom action: "# +str(rand_act)
        return [np.asarray(rand_act), self.getQ(s, rand_act)]

    loop_flags = []
    if self.action_size > 1:
        loop_flags = ['external_loop']

    for a in np.nditer(self.action_space, flags=loop_flags, order='F'):

        a_previous = self.a_max * 1000;

        if loop_flags == []:
            a = [a]
        a = np.asarray(a, dtype = np.float32)


        
        Q = self.getQ(s, a)
        if (Q > Q_best):
           a_best = a
           Q_best = Q 

        for _ in range(self.max_iter):
            
            delta = self.mlp.d_network()[2] #third element is action dim of mlp
            a = a + delta
            if not self.discretize_state:
               a = np.minimum(a, self.a_max) #keep in range
               a = np.maximum(a, self.a_min)
            else:
               a = np.minimum(a, np.ones(a.size()))
               a = np.maximum(a, np.ones(a.size()) * -1)
            
            Q = self.getQ(s, a)

            if (Q > Q_best):
                a_best = a
                Q_best = Q

            if np.linalg.norm(a - a_previous) < 0.0001:
                break
    #print a_best
    return [a_best, Q_best]

  def update(self, old_state, old_action, new_state, action_performed, 
	reward, isFinalState = False):
    learningRate = 1
    old_Q = self.getQ(old_state, old_action)
    new_Q = self.getQ(new_state, action_performed)
    if isFinalState:
       diff = learningRate * (reward - old_Q)
    else:
       diff = learningRate * (reward + self.discount * new_Q - old_Q)
    
    target = new_Q + diff
    self.updateQ(old_action, old_state, target)

    
   
