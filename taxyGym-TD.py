# taxi gym using a temporal difference algorithm

import numpy as np
import gym 
 

def main():
    #creating taxi environ
    env = gym.make('Taxi-v3')

    #initialize Q-table sizes
    state_size = env.observation_space.n #total number of states 
    action_size = env.action_space.n #total number of actions

    #initializes a qtable with 0's for all q-values 
    qtable = np.zeros((state_size, action_size))

    #hyperparameters to tune 
    epsilon = 1.0
    epsilon_min = 0.01
    decay_rate = 0.0001
    alpha = 0.1
    gamma = 0.9
    lambda_weight = 1

    #training variables
    num_episodes = 10000
    max_steps = 100# per episode

    def choose_action(state):
        action = 0
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(qtable[state,:])
        return action

    def update(state, new_state, action, new_action, reward):

        #prediction of future state-action value
        predict = qtable[state,action]
        #target state-action value
        target = reward + gamma*qtable[new_state, new_action]
        qtable[state, action] = qtable[state,action] + alpha*(target - predict)

        #difference between target and predicted state-action value
        # sigma = target - predict
        # eligibility[state, action] = eligibility[state,action] + 1

        # for n in range(state_size):
        #     for m in range(action_size):
        #         #Qlearning algorithm: qtable  element = qtable element + discounted sigma and eligibility
        #         qtable[n,m] = qtable[n,m] + alpha*sigma*eligibility[n,m]
        #         eligibility[n,m] = gamma*lambda_weight*eligibility[n,m]

    #training
    for episode in range(num_episodes):


        #create eligibility trace instance
        eligibility = np.zeros((state_size, action_size))
        #reset the environment 
        state = env.reset()
        action = choose_action(state)
        done = False

        for s in range(max_steps):

            #render step for debugging purposes
            #env.render()

            #take action and observe reward, get new_state
            new_state, reward, done, info = env.step(action)

            #choose action using policy derived from Q (epsilon-greedy)
            new_action = choose_action(new_state)

            update(state, new_state, action, new_action, reward)

            #update to our new state
            state = new_state
            action = new_action

            #if done, finish episode
            if done == True:
                break

        #decrease epsilon 
        # epsilon = np.exp(-alpha*episode)
        if epsilon > epsilon_min:
            epsilon = epsilon*np.exp(-decay_rate*episode)
        print(f"\r episode number: {episode}")

    print(f"Training completed over {num_episodes} episodes")
    input("Press Enter to watch trained agent")

    #watch trained agent 
    state = env.reset()
    done = False
    rewards = 0

    for s in range(max_steps):

        print(f"Trained agent")
        print("Step {}".format(s+1))

        action = np.argmax(qtable[state,:])
        new_state, rewards, done, info = env.step(action)
        env.render()
        print(f"score: {rewards}")
        state = new_state

        if done == True:
            break

        env.close()

if __name__ == "__main__": 
    main()
