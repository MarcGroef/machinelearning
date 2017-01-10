from includes.MLP2 import MLP
#from includes.tfmlp import tfMLP
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
  
  
  nn = MLP(2, 2, [10, 10, ], 1)
  #nn = tfMLP(2, 100, 1)
  for it in range(500):
    loss = 0
    loss += nn.train(xorIn1.astype(float), xorOut1.astype(float), 0.02, 0.2)
    loss += nn.train(xorIn4.astype(float), xorOut4.astype(float), 0.02, 0.2)
    loss += nn.train(xorIn2.astype(float), xorOut2.astype(float), 0.02, 0.2)
    loss += nn.train(xorIn3.astype(float), xorOut3.astype(float), 0.02, 0.2)
    #print "learning.. iter" + str(it)0
  
    print loss / 4
  res = 50
  valMap = np.random.random((res, res))
  for x1 in range(res):
    for x2 in range(res):
      position = float(2 * x1) / res - 1
      velocity = float(2 * x2) / res - 1
      
      v = nn.process(np.asarray([position, velocity ]).astype(float))#[0][0]
      #print v
      #print [position, velocity, v[0]]
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

	#env = gym.make('LunarLanderContinuous-v2')
        #stateSize = env.reset().size
        #actionSize = env.action_space.sample().size
        #cacla = Cacla(actionSize,np.ones(actionSize) * -1 , np.ones(actionSize), np.ones(actionSize), stateSize)
	#state = env.reset()

	#1 action, from -1 to 1, with 2 input states
	cacla = Cacla(1,[-1], [1], 2)
	env = gym.make('MountainCarContinuous-v0')
	state = env.reset()

        plt.ion() ## Note this correction
	fig=plt.figure()
	plt.axis([0,0,0,0])

	i=0
	x1=list()
	y1=list()
	for epoch in range(2000):
		env.reset()
		tot_reward = 0
		finished = False
		while not finished:

                #for iteration in range(4000):
                        #if(finished):
                          #break
			old_state = state
			#if epoch % 10 == 0:
			#	env.render()
			action = cacla.chooseAction(state)
			done = env.step(action)

			finished = done[2]
			reward = done[1]
                        #print reward
			tot_reward += reward
			state = done[0]
			cacla.update(old_state, action, state, reward, finished)

		cacla.adjustSigma()

		x1.append(i);
		y1.append(tot_reward);
		plt.scatter(i,tot_reward);
		i+=1;
		plt.show()
		plt.pause(0.0001) #Note this correction
		print tot_reward

def sarsa_test(render = False):
	
	#env = gym.make('MountainCarContinuous-v0')
        env = gym.make('LunarLanderContinuous-v2')
        stateSize = env.reset().size
        actionSize = env.action_space.sample().size

        #sarsa = Sarsa(1,[-1], [1], [2], 2)
        sarsa = Sarsa(actionSize,np.ones(actionSize) * -1 , np.ones(actionSize), np.ones(actionSize), stateSize)  ##discrete actions 1, -1 for sanity check
         
        #env = gym.make('Pendulum-v0')
        plt.ion() ## Note this correction
	fig=plt.figure()
	#plt.axis([0,0,0,0])
	epochs = list()
        rewards = list()
        nGameIterations = 20000
        nEpochs = 10000
        epochFailed = True
	for epoch in range(nEpochs):
                if(epoch == nEpochs - 100):
                   print "random action chance set to 0"
                   sarsa.random_chance = 0.1
                   sarsa.learningRate = 0
                #print state
                #sarsa.printValueMap(1)
		state = env.reset()
                
		tot_reward = 0
		tot_Q = 0
                action = sarsa.chooseAction(state)

                done = env.step(action[0])
		reward = done[1]
                finished = done[2]
                #if(epoch % 100 == 0 and epoch > 100):
                #    render = True
                #else:
                #    render = False
                render = True
                #ensure sarsa doesnt learn from killed epochs

                sarsa.resetBrainBuffers()
                epochFailed = True
                render = True

                 
		for iteration in range(nGameIterations):
                        
                            
                        #done = env.step(action[0])
                        #print done
                        #reward = done[1]
                        #tot_reward += reward
                        #finished = done[2]

                        #if(iteration % 10 != 0 and not finished):  ##only act once every 10 frames. Debugs and learns hell of a lot faster <edit: randomchance ==1 preformed qually well>
                        #    continue
                        #if(render and iteration % 5 == 0 and epoch > 100):
                        #env.render()
                        old_state = state
  		        old_action = action
                        if( not finished):
			   
			   action = sarsa.chooseAction(state)
			   tot_Q += action[1]			
			
			
                           state = done[0]

			done = env.step(action[0])
                        reward = done[1]
                        
                        tot_reward += reward
                        finished = done[2]
                        if(reward > 0 and finished):
                           print "***********************SUCCESS************************"
			
			#print [action[0][0], reward, action[1][0]]
                        if finished:
			   sarsa.update(old_state, old_action[0], state, action[0], reward, True)
                           epochFailed = False
                           break
                        else:
		           sarsa.update(old_state, old_action[0], state, action[0], reward)

                        #done = env.step(action[0])
                        ##reward = done[1]
                        #tot_reward += reward
                        #finished = done[2]

		print "rewards @ epoch " + str(epoch ) + ": " + str(tot_reward)
                epochs.append(epoch)
                rewards.append(tot_reward)
                #plt.scatter(epochs, rewards)
                #plt.show()
                #plt.pause(1)

if __name__ == "__main__":
	#xorTest()
	cacla_test()
	#sarsa_test()
	##nfac_test()

	
