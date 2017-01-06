import numpy as np
from MLP import MLP
import sys
import matplotlib.pyplot as plt  ##sudo apt-get install python-matplotlib
from sklearn.neural_network import MLPClassifier

##https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2014-175.pdf
class Sarsa():
  a_max = None
  a_min = None
  a_delta = None



  def __init__(self, a_dim, a_min, a_max, a_delta, state_size, random_chance = 0.01, learningRate = 0.01, discount = 0.9):
    self.a_max = np.asarray(a_max)
    self.a_min = np.asarray(a_min)
    self.a_delta = np.asarray(a_delta)
    self.action_size = a_dim
    self.action_space = self.define_action_space(a_dim, a_min, a_max, a_delta)

    self.state_size = state_size
    self.mlp = MLP(self.action_size + self.state_size, 1000, 1)
    self.mlp2 = MLPClassifier(100, activation='relu')
    self.max_iter = 10
    self.learningRate = learningRate
    self.discount = discount
    self.random_chance = random_chance

  def action_dist(self, a1, a2):  
    return np.linalg.norm(a1, a2)
 
  def getQ(self, state, action):
    #print "state = " + str(state)
    #print "action = " + str(action)
    mlpvec = np.concatenate([state, action])
    #print mlpvec
    return self.mlp.predict(mlpvec)
    return self.mlp.process(mlpvec)

  def updateQ(self, action, state, targetOut):
    self.mlp2.fit(np.concatenate([state, action]), targetOut)
    #self.mlp.train(np.concatenate([state, action]), targetOut, 0.001, 0.1)

  def define_action_space(self, dim, min, max, delta):
    dim_lst = []

    for i in range(0, dim):
      dim_lst.append(np.arange(min[i], max[i] + delta[i], delta[i]))

    return np.array(np.meshgrid(*dim_lst))

  def printValueMap(self, action):



    posres = 200
    valMap = np.random.random((posres, posres))
    for pos in range(posres):
       for vel in range(posres):
          position = float(2 * pos) / posres - 1
          velocity = float(2 * vel) / posres - 1
          Q =  self.getQ(np.asarray([position, velocity]), np.asarray([action]))

          valMap[pos][vel] = Q

    
    plt.imshow(valMap, cmap = 'hot')#, interpolation='nearest')
    plt.show()
    plt.pause(0.001)

  def chooseAction(self, s):
    a_best = self.a_min

    Q_best = self.getQ(a_best, s)

    #    print(args)
    
    if(np.random.rand(1) > (1 - self.random_chance)):  ##random action
	#rand_act = (np.random.rand(1) - 0.5) * 2
        rand_act = []
        for i in range(0,self.action_size):
            rand_act.append(np.random.choice(np.arange(self.a_min[i],self.a_max[i],self.a_delta[i])))
	#print "radom action: " +str(rand_act)
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
        continue ##uncomment for sarsa-lamda
        ##Newtons method
        for _ in range(self.max_iter):
            
            deltaNewton = (self.mlp.d_network()[0][2] / self.mlp.dd_network()[0][2]) #third element is action dim of mlp
            a = a - deltaNewton

            a = np.minimum(a, self.a_max) #keep in range
            a = np.maximum(a, self.a_min)
            
            Q = self.getQ(s, a)

            if (Q > Q_best):
                a_best = a
                Q_best = Q

            if np.linalg.norm(a - a_previous) < 0.0001:
                break

    return [a_best, Q_best]

  def update(self, old_state, old_action, new_state, action_performed, reward, isFinalState = False):
    old_Q = self.getQ(old_state, old_action)
    if isFinalState:
       diff = self.learningRate * (reward - old_Q)
    else:
       diff = self.learningRate * (reward + self.discount * self.getQ(new_state, action_performed) - old_Q)
    
    #print [old_action, old_Q, diff]
    target = old_Q + diff
    self.updateQ(action_performed, old_state, target)
    
    
   
