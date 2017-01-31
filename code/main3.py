from includes.MLP2 import MLP
from includes.nfac2 import NFAC2
import matplotlib.pyplot as plt
import pickle
import numpy as np
import gym
import os
import random
from datetime import datetime
from scipy import signal

def heuristic(env, s):
    # Heuristic for:
    # 1. Testing. 
    # 2. Demonstration rollout.
    angle_targ = s[0]*0.5 + s[2]*1.0         # angle should point towards center (s[0] is horizontal coordinate, s[2] hor speed)
    if angle_targ >  0.4: angle_targ =  0.4  # more than 0.4 radians (22 degrees) is bad
    if angle_targ < -0.4: angle_targ = -0.4
    hover_targ = 0.55*np.abs(s[0])           # target y should be proporional to horizontal offset

    # PID controller: s[4] angle, s[5] angularSpeed
    angle_todo = (angle_targ - s[4])*0.5 - (s[5])*1.0

    # PID controller: s[1] vertical coordinate s[3] vertical speed
    hover_todo = (hover_targ - s[1])*0.5 - (s[3])*0.5

    if s[6] or s[7]: # legs have contact
        angle_todo = 0
        hover_todo = -(s[3])*0.5  # override to reduce fall speed, that's all we need after contact

    if env.continuous:
        a = np.array( [hover_todo*20 - 1, -angle_todo*20] )
        a = np.clip(a, -1, +1)
    else:
        a = 0
        if hover_todo > np.abs(angle_todo) and hover_todo > 0.05: a = 2
        elif angle_todo < -0.05: a = 3
        elif angle_todo > +0.05: a = 1
    return a

def nfac_test():

	#create logging folder
	dir_name = 'NFAC ' + ('%s' % datetime.now())
	dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../runs/nfac/' + dir_name)
	os.makedirs(dir_path)

	#env = gym.make('MountainCarContinuous-v0')
	env = gym.make('LunarLanderContinuous-v2')
	stateSize = env.reset().size
	actionSize = env.action_space.sample().size

	nfac = NFAC2(actionSize, np.ones(actionSize) * -1, np.ones(actionSize), stateSize)
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

	#create lists for plotting total reward
	xl=[]
	yl=[]

	for x in range(20000):
		env.reset()
		tot_reward = 0
		tot_Q = 0
		finished = False
		step = 0

		while not finished:
			old_state = state
			if x > 200:
				env.render()

			bigstateinput1 = [0] * (10 * len(old_state))
			for i in range(len(old_state)):
				try: 
					bigstateinput1[i * 10 + int((state[i] + 1) / 0.2)] = 1 + (random.uniform(0, 0.2) - 0.1)
					#bigstateinput1[(i * 20 + int((state[i] + 1) / 0.1)) - 1] = 0.5
					#bigstateinput1[(i * 20 + int((state[i] + 1) / 0.1)) + 1] = 0.5
				except IndexError:
					pass

			gauss = signal.gaussian(5, 1)

			bigstateinput1 = np.convolve(bigstateinput1, gauss, 'same')


			#print bigstateinput1

			act2 = nfac.chooseAction(np.array(bigstateinput1))

			action = act2
			#if (x < 2000):
				#action = heuristic(env, state)

			done = env.step(action)
			finished = done[2]
			reward = done[1]
			if step > 4000:
				break

			tot_reward += reward
			state = done[0]

			bigstateinput2 = [0] * (10 * len(state))

			for j in range(len(old_state)):
				try: 
					bigstateinput2[j * 10 + int((state[j] + 1) / 0.2)] = 1 + (random.uniform(0, 0.2) - 0.1)
					#bigstateinput2[(j * 20 + int((state[j] + 1) / 0.1)) - 1] = 0.5
					#bigstateinput2[(j * 20 + int((state[j] + 1) / 0.1)) + 1] = 0.5
				except IndexError:
					pass

			gauss = signal.gaussian(5, 1)

			bigstateinput2 = np.convolve(bigstateinput2, gauss, 'same')
			#collect for offline learning
			#if x < 2000: 
			nfac.collect(np.array(bigstateinput1), action, reward, np.array(bigstateinput2), finished)
			step = step + 1
		if finished:
			nfac.adjustSigma()
			#if x < 2000:
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

		print str(x) + " " + str(tot_reward)

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


if __name__ == "__main__":
	nfac_test()

	
