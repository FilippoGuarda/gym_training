import gym
import matplotlib as plt

env = gym.make('MountainCar-v0')

#observation and action space
obs_space = env.observation_space
action_space = env.action_space
print('The observation space: {}'.format(obs_space))
print("The action space: {}".format(action_space))
#reset the environment and see the initial observation
obs = env.reset()
print("The initial observation is {}".format(obs))
#sample a random action from the entire action space
random_action = env.action_space.sample()
#Take the action and get the new observation space
new_obs, reward, done, info = env.step(random_action)
print("The new observation is {}".format(new_obs))
env.render(mode = "human")
#input('')