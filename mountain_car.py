from xxlimited import foo
import gym
import numpy as np


def main():
    env = gym.make('MountainCar-v0')
    env.reset()

    episode_num = 10000
    max_steps = 200

    # define hyperparameters 
    epsilon = 1
    epsilon_min = 0.01
    epsilon_decay = 0.0001
    alpha = 0.1
    gamma = 0.9
    lambda_weight = 0.9

    state_num = (env.observation_space.high - env.observation_space.low) * np.array([10,100])
    state_num = np.round(state_num, 0).astype(int) + 1

    #check
    print(env.observation_space.high, env.observation_space.low)
    print(state_num)

    #initialize function approx tilings
    approximator = np.zeros((state_num[0], state_num[1], env.action_space.n, 10))

    #initialize qtable
    qtable = np.zeros((state_num[0], state_num[1], env.action_space.n, 10))

    #initialize elegibility
    elegibility = np.zeros_like(qtable)


    #define action choice function
    def choose_action(state_num, elegibility):

        action = 0
        e = elegibility[state_num[0], state_num[1]]
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
            e = 0
        else:
            action = np.argmax(np.transpose(qtable[state_num[0],state_num[1]])*approximator)
            e = e + 1
        return action

    #state discretization
    def discretize(state):

        state_adj = (state - env.observation_space.low)*np.array([10, 100])
        state_adj = np.round(state_adj, 0).astype(int)
        return state_adj

    #epsilon reduction
    def reduce_epsilon(episode_num):

        if epsilon > epsilon_min:
            epsilon = epsilon*np.exp(-epsilon_decay*episode_num)

    #adds 1 to all shifted tiles where the state is present
    def tiling_elegibility(state, decrease):

        for i in range(elegibility[:,:,:,1]):

            state_adj = (state - env.observation_space.low)*np.array([10, 100])+[0.1*i, 0.1*i]
            state_adj = np.round(state_adj, 0).astype(int)

            if not decrease:
                elegibility[state_adj[0], state_adj[1], :, i] += 1
            else:
                elegibility[state_adj[0], state_adj[1], :, i] += gamma*lambda_weight*elegibility[state_adj[0], state_adj[1], :, i]


    #update function
    def update(state, new_state, action, new_action, elegibility, reward):

        q = qtable[state[0], state[1],action]
        nextq = qtable[new_state[0], new_state[1], :]
        e = elegibility[state[0], state[1]]


        #prediction of future state-action value
        sigma = reward - np.transpose(qtable)*approximator


        #difference between target and predicted state-action value
        sigma = sigma - gamma*np.max(np.transpose(nextq)*approximator)

        approximator = approximator + alpha*sigma*e

        tiling_elegibility(state, decrease=True)


    #Q-learning function
    for episode in range(episode_num):

        done = False;

        #reset state and elegibility
        state = env.reset()
        state_adj = discretize(state)
        elegibility = np.zeros_like(qtable)

        while done != True:

            if episode >= (episode_num - 20):
                env.render()

            #add elegibility
            tiling_elegibility(state, decrease=False)
            #select action
            action = choose_action(state_adj, elegibility)
            new_state, reward, done, info = env.step(action)
            new_state_adj = discretize(new_state)
            new_action = choose_action(new_state_adj, elegibility)

            update(state_adj, new_state_adj, action, new_action, elegibility, reward)

            #update new state and action

            state = new_state
            action = new_action

        #decrease epsilon
        reduce_epsilon(episode)



main()