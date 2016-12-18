import gym

env = gym.make('CartPole-v0')
env.reset()

for _ in range(1000):
	env.render()
        done = env.step(env.action_space.sample())

	if done[2]:
		env.reset()

