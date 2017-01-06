from includes.MLP import MLP
from includes.tfmlp import tfMLP
from includes.sarsa import Sarsa
from includes.cacla import Cacla
from includes.nfac import NFAC
import matplotlib.pyplot as plt
import numpy as np
import gym

def xorTest():
  
  xorIn1 = np.array([-1,1])
  xorIn2 = np.array([1,-1])
  xorIn3 = np.array([1,1])
  xorIn4 = np.array([-1,-1])
  
  xorOut1 = np.array([1])
  xorOut2 = np.array([1])
  xorOut3 = np.array([-1])
  xorOut4 = np.array([-1])
  
  
  #nn = MLP(2, 10, 1)
  nn = tfMLP(2, 10, 1)
  for iter in range(1000):
    loss = 0
    loss += nn.train(xorIn1, xorOut1)#, 0.05, 0.04)
    loss += nn.train(xorIn4, xorOut4)#, 0.05, 0.04)
    loss += nn.train(xorIn2, xorOut2)#, 0.05, 0.04)
    loss += nn.train(xorIn3, xorOut3)#, 0.05, 0.04)
    
    print loss
  res = 50
  valMap = np.random.random((res, res))
  for x1 in range(res):
    for x2 in range(res):
      position = float(2 * x1) / res - 1
      velocity = float(2 * x2) / res - 1
      print [position, velocity]
      v = nn.process(np.asarray([x1, x2]))
      valMap[x1][x2] = v


  plt.imshow(valMap, cmap = 'hot')#, interpolation='nearest')
  plt.colorbar()
  plt.show()
  plt.pause(0.001)

def nfac_test():
	nfac = NFAC(1,[-1], [1], 2)
	env = gym.make('MountainCarContinuous-v0')
	state = env.reset()

	for x in range(2000):
		env.reset()
		tot_reward = 0
		tot_Q = 0
		finished = False
		for _ in range(2000):
			old_state = state
			#env.render()
			action = nfac.chooseAction(state)
			done = env.step(action)
			finished = done[2]
			reward = done[1]
			tot_reward += reward
			state = done[0]
			#collect for offline learning
			nfac.collect(old_state, action, reward, state, finished)
			if finished:
				break
		if finished:
			nfac.update()
		else:
			nfac.clearCollection()
		print tot_reward
  
def cacla_test():
	cacla = Cacla(1,[-1], [1], 2)
	env = gym.make('MountainCarContinuous-v0')
	state = env.reset()

        plt.ion() ## Note this correction
	fig=plt.figure()
	plt.axis([0,0,0,0])

	i=0
	x1=list()
	y1=list()
	for _ in range(2000):
		env.reset()
		tot_reward = 0
		tot_Q = 0
		finished = False
		while not finished:
			old_state = state
			#env.render()
			action = cacla.chooseAction(state)
			done = env.step(action)
			finished = done[2]
			reward = done[1]
			tot_reward += reward
			state = done[0]
			cacla.update(old_state, action, state, reward, finished)

		x1.append(i);
		y1.append(tot_reward);
		plt.scatter(i,tot_reward);
		i+=1;
		plt.show()
		plt.pause(0.0001) #Note this correction
		print tot_reward

def sarsa_test(render = False):
	#sarsa = Sarsa(1,[-1], [1], [0.1], 2)
        sarsa = Sarsa(1,[-1], [1], [2], 2)  ##discrete actions 1, -1 for sanity check
	env = gym.make('MountainCarContinuous-v0')
        
        plt.ion() ## Note this correction
	fig=plt.figure()
	#plt.axis([0,0,0,0])
	epochs = list()
        rewards = list()

	for epoch in range(2000):
                #print state
                sarsa.printValueMap(1)
		state = env.reset()
		tot_reward = 0
		tot_Q = 0
                action = sarsa.chooseAction(state)

                done = env.step(action[0])
		reward = done[1]
                finished = done[2]
		for iteration in range(10000):
                        if(render):
                            env.render()
                        old_state = state
  		        old_action = action
                        if( not finished):
			   
			   action = sarsa.chooseAction(state)
			   tot_Q += action[1]			
			
			
                           state = done[0]

			

			
			#print [action[0][0], reward, action[1][0]]
                        if finished:
			   sarsa.update(old_state, old_action[0], state, action[0], reward, True)
                           break
                        else:
		           sarsa.update(old_state, old_action[0], state, action[0], reward)

                        done = env.step(action[0])
                        reward = done[1]
                        tot_reward += reward
                        finished = done[2]

		print tot_reward
                epochs.append(epoch)
                rewards.append(tot_reward)
                #plt.scatter(epochs, rewards)
                #plt.show()
                #plt.pause(1)

if __name__ == "__main__":
	xorTest()
	#cacla_test()
	#sarsa_test()
	#nfac_test()

	
