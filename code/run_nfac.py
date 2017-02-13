from includes.MLP2 import MLP
from includes.nfac import NFAC
import matplotlib.pyplot as plt
import pickle
import numpy as np
import gym
import os
import random
from datetime import datetime
from scipy import signal
import sys

def nfac_test(environment, a_hu, v_hu, lr, discount, sigma=10, sig_k = 0.99, sig_min = 0):

	#create logging folder
	dir_name = 'NFAC ' + ('%s' % datetime.now())
	dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../runs/' + environment + '/nfac/' + dir_name)
	os.makedirs(dir_path)

	#initialize environment and algorithm class	
	env = gym.make(environment)	
	stateSize = env.reset().size
	actionSize = env.action_space.sample().size
	nfac = NFAC(actionSize, np.ones(actionSize) * -1, np.ones(actionSize), stateSize, a_hu, v_hu, lr, sigma, sig_k, sig_min, discount=discount)
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
	params.write('Learning rate: ' + str(nfac.learning_rate) + '\n')
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
	
	#create file for logging total reward per epoch
	success_file = os.path.join(dir_path, 'success.txt')
	successes = open(success_file, 'w')
	successes.close()

	#for 2000 epochs
	for x in range(2000):
		env.reset()
		tot_reward = 0
		tot_Q = 0
		finished = False
		step = 0

		while not finished: #one run of the algorithm
			old_state = state
			#get action from actor
			action = nfac.chooseAction(state)
			#check whether goal state is reached
			done = env.step(action)
			finished = done[2]
			reward = done[1]

			if step > 4000: #maximal number of iterations in one run
				break

			#update reward and state			
			tot_reward += reward
			state = done[0]

			#collect for offline learning
			nfac.collect(old_state, action, reward, state, finished)
			step = step + 1
		#learn offline
		nfac.adjustSigma()
		nfac.update()
		if x % 100 == 0: #periodically store mlp weights
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
		#success is defined as finishing with a positive reward
		if finished and tot_reward>0:
			success = True
		else:
			success = False

		#update rewards log
		rewards = open(rewards_file, 'a')
		rewards.write(str(tot_reward) + '\n')
		rewards.close()

		#update successes log
		successes = open(success_file, 'a')
		successes.write(str(success) + '\n')
		successes.close()


if __name__ == "__main__":
	mountain_car = True
        lunar_lander = True	
	if mountain_car:
		nfac_test('MountainCarContinuous-v0', 200, 200, 0.01, 0.99)
	if lunar_lander:
		nfac_test('LunarLanderContinuous-v2', 50, 50, 0.05, 0.99)
