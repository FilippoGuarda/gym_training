import numpy as np
import gym 
import random
 

def main():
    #creating taxi environ
    env = gym.make('Taxi-v3')

    #initialize Q-table sizes
    state_size = env.observation_space.n #total number of states 
    action_size = env.action_space.n #total number of actions
    #initializes a qtable with 0's for all q-values 
    qtable = np.zeros((state_size, action_size))

    #hyperparameters to tune 
    learning_rate = 0.8
    discount_rate = 0.6
    epsilon = 1.0 #probability our agent will explore
    decay_rate = 0.05 #of epsilon 

    #training variables
    num_episodes = 1000
    max_steps = 99 # per episode

    #training

    for episode in range(num_episodes):

        #reset the environment 
        state = env.reset()
        done = False

        for s in range(max_steps):
            #we include an exploration /  exploitation changing algo
            #exploration-exploitation tradeoff 
            if random.uniform(0,1) < epsilon:
                #explore
                action = env.action_space.sample()
            else: 
                #exploit
                action = np.argmax(qtable[state,:])

            #take action and observe reward
            new_state, reward, done, info = env.step(action)

            #Qlearning algorithm: Q(s,a) := Q(s,a) + learning_rate * (reward + discount_rate * max Q(s'a") - Q(s,a))
            qtable[state, action] += learning_rate * (reward + discount_rate * np.max(qtable[new_state, :]) - qtable[state, action])

            #update to our new state
            state = new_state

            #if done, finish episode
            if done == True:
                break

        #decrease epsilon 
        
        epsilon = np.exp(-decay_rate*episode)
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