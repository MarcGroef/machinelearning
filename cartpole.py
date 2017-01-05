import gym

#env = gym.make('CartPole-v0')
env = gym.make('MountainCarContinuous-v0')
env.reset()
action = env.action_space.sample()
for _ in range(1000):
	env.render()
        done = env.step(action)#env.action_space.sample()) 
	if done[2]:
		env.reset()

