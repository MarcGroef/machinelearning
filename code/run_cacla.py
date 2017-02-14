from includes.MLP2 import MLP
from includes.cacla import Cacla
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import gym
import os
import sys

def cacla(algorithm, random_chance, discount, learningRate, sigma, sd, action_hidden_layers, value_hidden_layers):

	dir_name = 'CACLA ' + ('%s' % datetime.now())
	dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../runs/CACLA/' + dir_name)
	os.makedirs(dir_path)

	if algorithm == 1: 
		env = gym.make('MountainCarContinuous-v0')
	if algorithm == 0:
		env = gym.make('LunarLanderContinuous-v2')
	
	stateSize = env.reset().size
	actionSize = env.action_space.sample().size
	cacla = Cacla(actionSize,np.ones(actionSize) * -1 , np.ones(actionSize), stateSize,random_chance,discount,learningRate,sigma,sd,action_hidden_layers, value_hidden_layers)
	state = env.reset()

	#create logging file
	rewards_file = os.path.join(dir_path, 'rewards.txt')
	rewards = open(rewards_file, 'w')
	rewards.write('epoch total_reward\n')
	rewards.close()

	i=0
	xl = []
	yl = []
	n = 100
	av_reward = 0
	nrsuccess = 0
	allrewards = []
	for epoch in range(2000):
		env.reset()
		tot_reward = 0
		finished = False
		step = 0
		while not finished:
			if(step == 10000):
				break
			
			old_state = state
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
			if epoch > 1899:
				if (finished or step == 10000):
					av_reward += (tot_reward / 100)
				if finished: 
					nrsuccess += 1
		cacla.adjustSigma()

		xl.append(i);
		yl.append(tot_reward);

		rewards = open(rewards_file, 'a')
		rewards.write(str(epoch) + ' ' + str(tot_reward) + '\n')
		rewards.close()

		i+=1;
		print epoch, tot_reward


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
	params.write('Learning rate: ' + str(cacla.learningRate) + '\n')
	params.write('Average Reward final 100 epochs: ' + str(av_reward) + '\n')	
	params.write('Number of successes in final 100 epochs: ' + str(nrsuccess) + '\n')
	params.close()

	success_file = os.path.join(dir_path, 'successes.txt')
	success = open(success_file, 'w')
	success.write('epoch total_reward\n')
	success.close()

	plt.plot(xl, yl)
	savestr = str(random_chance) + '_' + str(discount) + '_' + str(learningRate) + '_' + str(sigma) + '_' + str(sd) + '_' + str(action_hidden_layers) + '_' + str(value_hidden_layers) + '_total_reward.png'
	os.path.join(dir_path, savestr)
	plt.savefig(os.path.join(dir_path, savestr))	


if __name__ == "__main__":

	algorithm = int(sys.argv[1])

	print algorithm

	#LunarLander:
	if algorithm == 0:
		random_chance_vals = 0.1
		discount_vals = 0.999
		learning_rate_vals = 0.001
		sigma_vals = 10
		sd_vals = 1
		action_hidden_layers = 200
		value_hidden_layers = 200
	#MountainCar:
	if algorithm == 1:
		random_chance_vals = 0.1
		discount_vals = 0.999
		learning_rate_vals = 0.01
		sigma_vals = 10
		sd_vals = 1
		action_hidden_layers = 200
		value_hidden_layers = 200

	cacla(algorithm, random_chance_vals, discount_vals, learning_rate_vals, sigma_vals, sd_vals, action_hidden_layers, value_hidden_layers)