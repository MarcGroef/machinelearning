from includes.MLP2 import MLP
#from includes.tfmlp import tfMLP
from includes.sarsa import Sarsa
from includes.cacla import Cacla
from includes.nfac import NFAC
import matplotlib.pyplot as plt
import pickle
import numpy as np
import gym
import os
from datetime import datetime
import sys

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

	#create logging folder
	dir_name = 'NFAC ' + ('%s' % datetime.now())
	dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../runs/nfac/' + dir_name)
	os.makedirs(dir_path)

	#env = gym.make('MountainCarContinuous-v0'
	env = gym.make('LunarLanderContinuous-v2')
	stateSize = env.reset().size
	actionSize = env.action_space.sample().size
	nfac = NFAC(actionSize, np.ones(actionSize) * -1, np.ones(actionSize), stateSize)
	state = env.reset()

	#create logging param file
	params_file = os.path.join(dir_path, 'parameters.txt')
	brain_init_file = os.path.join(dir_path, 'mlps_weights_init.txt')
	params = open(params_file, 'w')
	params.write('NFAC parameters\n')
	params.write('\nAction MLP:\n')
	params.write('Number of hidden layers: ' + str(nfac.action_mlp.nLayers) + '\n')
	params.write('Hidden layer sizes: ' + str(nfac.action_mlp.hiddenSizes) + '\n')
	params.write('\nValue MLP:\n')
	params.write('Number of hidden layers: ' + str(nfac.value_mlp.nLayers) + '\n')
	params.write('Hidden layer sizes: ' + str(nfac.value_mlp.hiddenSizes) + '\n')
	params.write('\nDiscount: ' + str(nfac.discount) + '\n')
	params.write('Random chance: ' + str(nfac.random_chance) + '\n')
	params.write('SD: ' + str(nfac.sd) + '\n')
	params.write('Sigma: ' + str(nfac.sigma) + '\n')
	params.close()

	#log mlp init weights
	brain_init = open(brain_init_file, 'w')
	brain_init.write('Action MLP:\n')
	action_brain = nfac.action_mlp.getBrain()
	brain_init.write('Initial hidden weights: ' + str(action_brain[0]) + '\n')
	brain_init.write('Initial hidden bias: ' + str(action_brain[1]) + '\n')
	brain_init.write('Initial output weights: ' + str(action_brain[2]) + '\n')
	brain_init.write('Initial output bias: ' + str(action_brain[3]) + '\n')
	brain_init.write('Value MLP:\n')
	value_brain = nfac.value_mlp.getBrain()
	brain_init.write('Initial hidden weights: ' + str(value_brain[0]) + '\n')
	brain_init.write('Initial hidden bias: ' + str(value_brain[1]) + '\n')
	brain_init.write('Initial output weights: ' + str(value_brain[2]) + '\n')
	brain_init.write('Initial output bias: ' + str(value_brain[3]) + '\n')
	brain_init.close()

    #create file for logging total reward per epoch
	rewards_file = os.path.join(dir_path, 'rewards.txt')
	rewards = open(rewards_file, 'w')
	rewards.write('epoch total_reward\n')
	rewards.close()

	#create lists for plotting total reward
	xl=[]
	yl=[]

	for x in range(20000):
		env.reset()
		tot_reward = 0
		tot_Q = 0
		finished = False
		while not finished:
			old_state = state
			#env.render()
			action = nfac.chooseAction(state)
			
			done = env.step(action)
			finished = done[2]
			reward = done[1]
			if(reward > 0 and finished):
				print "***********************SUCCESS************************"
			tot_reward += reward
			state = done[0]
			#collect for offline learning
			nfac.collect(old_state, action, reward, state, finished)
		nfac.adjustSigma()
		if finished:
			nfac.update()
			#periodically log mlp weights to file
			if x % 100 == 0:
				brain_file = os.path.join(dir_path, 'weights_mlps_epoch_' + str(x) + '.txt')
				brain = open(brain_file, 'w')
				action_brain = nfac.action_mlp.getBrain()
				brain.write('Action MLP:')
				brain.write('Hidden weights: ' + str(action_brain[0]))
				brain.write('Hidden bias: ' + str(action_brain[1]))
				brain.write('Output weights: ' + str(action_brain[2]))
				brain.write('Output bias: ' + str(action_brain[3]))
				brain.write('\nValue MLP:')
				value_brain = nfac.value_mlp.getBrain()
				brain.write('Hidden weights: ' + str(value_brain[0]))
				brain.write('Hidden bias: ' + str(value_brain[1]))
				brain.write('Output weights: ' + str(value_brain[2]))
				brain.write('Output bias: ' + str(value_brain[3]))
				brain.close()

		#else:
			#nfac.clearCollection()
		print tot_reward

		#update rewards log
		rewards = open(rewards_file, 'a')
		rewards.write(str(x) + ' ' + str(tot_reward) + '\n')
		rewards.close()

		#update rewards lists
		xl.append(x)
		yl.append(tot_reward)

	#make and store total rewards plot
	plt.plot(x, tot_reward)
	plt.savefig(os.path.join(dir_path, 'total_reward.png'))



    #self.a_max = a_max
  
def cacla_train(random_chance, discount, learningRate, sigma, sd, action_hidden_layers, value_hidden_layers):

	dir_name = 'CACLA ' + ('%s' % datetime.now())
	dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../runs/cacla/' + dir_name)
	os.makedirs(dir_path)

	env = gym.make('MountainCarContinuous-v0')
	#env = gym.make('LunarLanderContinuous-v2')
	stateSize = env.reset().size
	actionSize = env.action_space.sample().size
	cacla = Cacla(actionSize,np.ones(actionSize) * -1 , np.ones(actionSize), stateSize,random_chance,discount,learningRate,sigma,sd,action_hidden_layers, value_hidden_layers)
	state = env.reset()

	i=0
	xl = []
	yl = []
	n = 100
	for epoch in range(1500):
		env.reset()
		tot_reward = 0
		finished = False
		step = 0
		if (epoch % n == 0 and n != 0): 
		  output = open('actor_' + str(epoch) + '.pkl', 'wb')
		  pickle.dump(np.asarray(cacla.getActorBrain()), output)
		  output.close()
		  output = open('critic_' + str(epoch) + '.pkl', 'wb')
		  pickle.dump(np.asarray(cacla.getCriticBrain()), output)
		  output.close()
		while not finished:

                        if(step == 10000):
                          break
			
			old_state = state
			#if epoch % 10 == 0:
				#env.render()
			action = cacla.chooseAction(state)
			done = env.step(action)

			finished = done[2]
			reward = done[1]
                        if(reward > 0 and finished):
                         	print "***********************SUCCESS************************"
			tot_reward += reward
			state = done[0]
			cacla.update(old_state, action, state, reward, finished)
			step = step + 1


		cacla.adjustSigma()

		xl.append(i);
		yl.append(tot_reward);


		i+=1;
		plt.show()
		plt.pause(0.0001) #Note this correction
		print tot_reward
	plt.plot(xl, yl)
	savestr = str(random_chance) + '_' + str(discount) + '_' + str(learningRate) + '_' + str(sigma) + '_' + str(sd) + '_' + str(action_hidden_layers) + '_' + str(value_hidden_layers) + '_total_reward.png'
	os.path.join(dir_path, savestr)
	plt.savefig(os.path.join(dir_path, savestr))	


def cacla_test():

	#create logging folder
	dir_name = 'CACLA ' + ('%s' % datetime.now())
	dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../runs/cacla/' + dir_name)
	os.makedirs(dir_path)

	#env = gym.make('LunarLanderContinuous-v2')
        #stateSize = env.reset().size
        #actionSize = env.action_space.sample().size
        #cacla = Cacla(actionSize,np.ones(actionSize) * -1 , np.ones(actionSize), np.ones(actionSize), stateSize)
	#state = env.reset()

	#1 action, from -1 to 1, with 2 input states
	cacla = Cacla(1,[-1], [1], 2)
	env = gym.make('MountainCarContinuous-v0')
	state = env.reset()

	actor_file = open('actor_1900.pkl', 'rb')
	actor = pickle.load(actor_file)

 	critic_file = open('critic_1900.pkl', 'rb')
	critic = pickle.load(critic_file)

	cacla.setActorBrain(actor)
	cacla.setCriticBrain(critic) #moet dit niet critic zijn?

	# create logging param file
	params_file = os.path.join(dir_path, 'parameters.txt')
	params = open(params_file, 'w')
	params.write('CACLA parameters\n')
	params.write('\nAction MLP:\n')
	params.write('Number of hidden layers: ' + str(cacla.action_mlp.nLayers) + '\n')
	params.write('Hidden layer sizes: ' + str(cacla.action_mlp.hiddenSizes) + '\n')
	params.write('\nValue MLP:\n')
	params.write('Number of hidden layers: ' + str(cacla.value_mlp.nLayers) + '\n')
	params.write('Hidden layer sizes: ' + str(cacla.value_mlp.hiddenSizes) + '\n')
	params.write('\nDiscount: ' + str(cacla.discount) + '\n')
	params.write('Random chance: ' + str(cacla.random_chance) + '\n')
	params.write('SD: ' + str(cacla.sd) + '\n')
	params.write('Sigma: ' + str(cacla.sigma) + '\n')
	params.close()

	#log initialisation mlp
	brain_init_file = os.path.join(dir_path, 'mlps_weights_init.txt')
	brain_init = open(brain_init_file, 'w')

	brain_init.write('Action MLP:\n')
	action_brain = cacla.action_mlp.getBrain()
	brain_init.write('Initial hidden weights: ' + str(action_brain[0]) + '\n')
	brain_init.write('Initial hidden bias: ' + str(action_brain[1]) + '\n')
	brain_init.write('Initial output weights: ' + str(action_brain[2]) + '\n')
	brain_init.write('Initial output bias: ' + str(action_brain[3]) + '\n')

	brain_init.write('Value MLP:\n')
	value_brain = cacla.value_mlp.getBrain()
	brain_init.write('Initial hidden weights: ' + str(value_brain[0]) + '\n')
	brain_init.write('Initial hidden bias: ' + str(value_brain[1]) + '\n')
	brain_init.write('Initial output weights: ' + str(value_brain[2]) + '\n')
	brain_init.write('Initial output bias: ' + str(value_brain[3]) + '\n')
	brain_init.close()

	# create file for logging total reward per epoch
	rewards_file = os.path.join(dir_path, 'rewards.txt')
	rewards = open(rewards_file, 'w')
	rewards.write('epoch total_reward\n')
	rewards.close()

	# create lists for plotting total reward
	xl = []
	yl = []

	n = 100
	for epoch in range(2000):
		env.reset()
		tot_reward = 0
		finished = False
		step = 0
		while not finished:

                        if(step == 100000):
                          break
			
			old_state = state
			#if epoch % 10 == 0:
				#env.render()
			action = cacla.chooseAction(state)
			done = env.step(action)

			finished = done[2]
			reward = done[1]
			tot_reward += reward
			state = done[0]
			#cacla.update(old_state, action, state, reward, finished)
			step = step + 1
		# periodically log mlp weights to file
		if (epoch + 1) % 100 == 0:
			brain_file = os.path.join(dir_path, 'weights_mlps_epoch_' + str(epoch) + '.txt')
			brain = open(brain_file, 'w')
			action_brain = cacla.action_mlp.getBrain()
			brain.write('Action MLP:')
			brain.write('Hidden weights: ' + str(action_brain[0]))
			brain.write('Hidden bias: ' + str(action_brain[1]))
			brain.write('Output weights: ' + str(action_brain[2]))
			brain.write('Output bias: ' + str(action_brain[3]))
			brain.write('\nValue MLP:')
			value_brain = cacla.value_mlp.getBrain()
			brain.write('Hidden weights: ' + str(value_brain[0]))
			brain.write('Hidden bias: ' + str(value_brain[1]))
			brain.write('Output weights: ' + str(value_brain[2]))
			brain.write('Output bias: ' + str(value_brain[3]))
			brain.close()

		#cacla.adjustSigma()
		print tot_reward

		# update rewards log
		rewards = open(rewards_file, 'a')
		rewards.write(str(epoch) + ' ' + str(tot_reward) + '\n')
		rewards.close()

		# update rewards lists
		xl.append(epoch)
		yl.append(tot_reward)

	# make and store total rewards plot
	plt.plot(xl, yl)
	plt.savefig(os.path.join(dir_path, 'total_reward.png'))


def sarsa_test(render = False):

	# create logging folder
	dir_name = 'SARSA ' + ('%s' % datetime.now())
	dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../runs/sarsa/' + dir_name)
	os.makedirs(dir_path)

	#env = gym.make('MountainCarContinuous-v0')
	env = gym.make('LunarLanderContinuous-v2')
	stateSize = env.reset().size
	actionSize = env.action_space.sample().size

	#sarsa = Sarsa(1,[-1], [1], [2], 2)
	sarsa = Sarsa(actionSize,np.ones(actionSize) * -1 , np.ones(actionSize), np.ones(actionSize), stateSize)  ##discrete actions 1, -1 for sanity check

	# create logging param file
	params_file = os.path.join(dir_path, 'parameters.txt')
	params = open(params_file, 'w')
	params.write('SARSA parameters\n')
	params.write('\nMLP:\n')
	params.write('Number of hidden layers: ' + str(sarsa.mlp.nLayers) + '\n')
	params.write('Hidden layer sizes: ' + str(sarsa.mlp.hiddenSizes) + '\n')
	params.write('\nDiscount: ' + str(sarsa.discount) + '\n')
	params.write('Random chance: ' + str(sarsa.random_chance) + '\n')
	params.write('Learning rate: ' + str(sarsa.learningRate) + '\n')
	params.close()

	# log initialisation mlp
	brain_init_file = os.path.join(dir_path, 'mlps_weights_init.txt')
	brain_init = open(brain_init_file, 'w')

	brain_init.write('MLP:\n')
	brain_state = sarsa.mlp.getBrain()
	brain_init.write('Initial hidden weights: ' + str(brain_state[0]) + '\n')
	brain_init.write('Initial hidden bias: ' + str(brain_state[1]) + '\n')
	brain_init.write('Initial output weights: ' + str(brain_state[2]) + '\n')
	brain_init.write('Initial output bias: ' + str(brain_state[3]) + '\n')

	# create file for logging total reward per epoch
	rewards_file = os.path.join(dir_path, 'rewards.txt')
	rewardsf = open(rewards_file, 'w')
	rewardsf.write('epoch total_reward\n')
	rewardsf.close()

	# create lists for plotting total reward
	xl = []
	yl = []
         
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

		# periodically log mlp weights to file
		if (epoch + 1) % 100 == 0:
			brain_file = os.path.join(dir_path, 'weights_mlps_epoch_' + str(epoch) + '.txt')
			brain = open(brain_file, 'w')
			brain_state = sarsa.mlp.getBrain()
			brain.write('Hidden weights: ' + str(brain_state[0]) + '\n')
			brain.write('Initial hidden bias: ' + str(brain_state[1]) + '\n')
			brain.write('Initial output weights: ' + str(brain_state[2]) + '\n')
			brain.write('Initial output bias: ' + str(brain_state[3]) + '\n')
			brain.close()

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
			if(not finished):
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

		# update rewards log
		rewardsf = open(rewards_file, 'a')
		rewardsf.write(str(epoch) + ' ' + str(tot_reward) + '\n')
		rewardsf.close()

		# update rewards lists
		xl.append(epoch)
		yl.append(tot_reward)

	# make and store total rewards plot
	plt.plot(xl, yl)
	plt.savefig(os.path.join(dir_path, 'total_reward.png'))

if __name__ == "__main__":
	#xorTest()
	#random_chance, discount, learningRate, sigma, sd  action_hidden_layers, value_hidden_layers):
	"""

	random_chance_vals	= [0.5,0.1,0.05,0.01,0.001]
	discount_vals 		= [0.999, 0.99, 0.90, 0.80,0.50]
	learning_rate_vals	= [0.4,0.2,0,1,0.05,0.01]
	sigma_vals			= [20,10,5,1,0.1]
	sd_vals				= [5,2,1,0.5,0.1]
	action_hidden_layers= [500,200,100,50,10]
	value_hidden_layers	= [500,200,100,50,10]

	"""
	print sys.argv
	print "test"
	#random_chance_vals = sys.argv[2]

	random_chance_vals	= float(sys.argv[1])
	discount_vals 		= float(sys.argv[2])
	learning_rate_vals	= float(sys.argv[3])
	sigma_vals			= float(sys.argv[4])
	sd_vals				= float(sys.argv[5])
	action_hidden_layers= int(sys.argv[6])
	value_hidden_layers	= int(sys.argv[7])

	print random_chance_vals
	print discount_vals
	print learning_rate_vals
	print sigma_vals
	print sd_vals
	print action_hidden_layers
	print value_hidden_layers



	"""for rc_val in random_chance_vals:
		for disc_val in discount_vals:
			for lr_val in learning_rate_vals:
				for sig_val in sigma_vals:
					for sd_val in sd_vals:
						for ah_val in action_hidden_layers:
							for vh_val in value_hidden_layers:"""
	cacla_train(random_chance_vals, discount_vals, learning_rate_vals, sigma_vals, sd_vals, action_hidden_layers, value_hidden_layers)
	#cacla_train(0.1, 0.99, 0.01, 10, 1, 200, 200)

	#cacla_test()
	#sarsa_test()
	#nfac_test()

	
