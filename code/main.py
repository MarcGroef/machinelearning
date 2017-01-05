from includes.MLP import MLP
from includes.sarsa import Sarsa
from includes.cacla import Cacla
import matplotlib.pyplot as plt
import numpy as np
import gym

def xorTest():
  
  xorIn1 = np.array([0,1])
  xorIn2 = np.array([1,0])
  xorIn3 = np.array([1,1])
  xorIn4 = np.array([0,0])
  
  xorOut1 = np.array([1])
  xorOut2 = np.array([1])
  xorOut3 = np.array([0])
  xorOut4 = np.array([0])
  
  
  nn = MLP(2, 5, 1)
  for iter in range(100):
    loss = 0
    loss += nn.train(xorIn1, xorOut1, 0.5)
    loss += nn.train(xorIn4, xorOut4, 0.5)
    loss += nn.train(xorIn2, xorOut2, 0.5)
    loss += nn.train(xorIn3, xorOut3, 0.5)
    
    print loss
  
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
	for x in range(2000):
		env.reset()
		tot_reward = 0
		tot_Q = 0
		n_iter = 0
		for _ in range(5000):
			n_iter += 1
			old_state = state
			action = env.action_space.sample()
			#env.render()
			old_action = action
			action = cacla.chooseAction(state)
			done = env.step(action)
			finished = done[2]
			reward = done[1]
			tot_reward += reward
			state = done[0]
			cacla.update(old_state, old_action, state, reward)
			if finished:
				break

		x1.append(i);
		y1.append(tot_reward);
		plt.scatter(i,tot_reward);
		i+=1;
		plt.show()
		plt.pause(0.0001) #Note this correction
		print tot_reward

def sarsa_test():
	sarsa = Sarsa(1,[-1], [1], [0.1], 2)
	env = gym.make('MountainCarContinuous-v0')
	state = env.reset()
	for _ in range(2000):
                print state
		env.reset();
		tot_reward = 0
		tot_Q = 0
		n_iter = 0
		for _ in range(10000):
			n_iter += 1
			old_state = state
			action = env.action_space.sample()
			env.render()
			old_action = action
			action = sarsa.chooseAction(state)
			tot_Q += action[1]
			#action = action[0]

			#print action
			done = env.step(action[0])
			finished = done[2]
			if finished:
				break
			#print done
			reward = done[1]
			tot_reward += reward
			state = done[0]
			#print [action[0][0], reward, action[1][0]]
			sarsa.update(old_state, old_action, state, action[0], reward)
		print tot_reward

if __name__ == "__main__":
	#xorTest()
	cacla_test()
	#sarsa_test()


