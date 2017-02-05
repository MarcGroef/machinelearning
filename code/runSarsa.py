from includes.MLP2 import MLP
from includes.sarsa import Sarsa
import matplotlib.pyplot as plt
import numpy as np
import gym
import os
from datetime import datetime


# create logging folder
dir_name = 'SARSA ' + ('%s' % datetime.now())
dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../runs/sarsa/' + dir_name)
os.makedirs(dir_path)

env = gym.make('MountainCarContinuous-v0')
#env = gym.make('LunarLanderContinuous-v2')
stateSize = env.reset().size
actionSize = env.action_space.sample().size

#sarsa = Sarsa(1,[-1], [1], [2], 2)
sarsa = sarsa = Sarsa(actionSize,np.ones(actionSize) * -1 , np.ones(actionSize), np.ones(actionSize) * 0.5, stateSize)   ##discrete actions 1, -1 for sanity check

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
  
  
def sarsa_test(render = False):


  nGameIterations = 10000
  nEpochs = 2000
  epochFailed = True
  for epoch in range(nEpochs):
    #if(epoch == nEpochs - 100):
  #   print "random action chance set to 0"
  #   sarsa.random_chance = 0.1
  #   sarsa.learningRate = 0
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
    sarsa.random_chance *= 0.99
    state = env.reset()

    tot_reward = 0
    tot_Q = 0
    action = sarsa.chooseAction(state)

    done = env.step(action[0])
    state = done[0]
    reward = done[1]
    finished = done[2]
    #if(epoch % 100 == 0 and epoch > 100):
    #    render = True
    #else:
    #    render = False
    render = True
    #ensure sarsa doesnt learn from killed epochs

    sarsa.resetBrainBuffers()
    render = True


    for iteration in range(nGameIterations):

      #if(render and iteration % 5 == 0 and epoch > 100):
      #env.render()
      
      old_state = state
      old_action = action


      action = sarsa.chooseAction(state)
      tot_Q += action[1]
      done = env.step(action[0])
      state = done[0]     
      reward = done[1]
      tot_reward += reward
      finished = done[2]

  
      if finished:
        if reward > 0:
          print "***********************SUCCESS************************"
        sarsa.update(old_state, old_action[0], state, action[0], reward, True)
        break
      else:
        sarsa.update(old_state, old_action[0], state, action[0], reward)
        
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
  sarsa_test()