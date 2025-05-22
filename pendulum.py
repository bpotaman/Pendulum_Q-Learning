import gymnasium as gym
import numpy as np
import pickle
import random
import threading

sample_frequency = 100

def learn(is_training, thread_number):
    env = gym.make("Pendulum-v1", render_mode="human" if not is_training else None)

    divide = 15
    epsilon = 1 if is_training else 0
    epsilon_decrease = 0.01
    epsilon_min = 0.05
    alpha = 10 ** (-thread_number)
    gamma = 0.9

    # discretize action and state space
    a = np.linspace(env.action_space.low[0], env.action_space.high[0], divide)
    
    x = np.linspace(env.observation_space.low[0], env.observation_space.high[0], divide)
    y = np.linspace(env.observation_space.low[1], env.observation_space.high[1], divide)
    w = np.linspace(env.observation_space.low[2], env.observation_space.high[2], divide)

    if is_training:
        # Q-table will have dimensions: divide x divide x divide x divide
        q_table = np.zeros((divide, divide, divide, divide))

    else:
        with open(rf"q_tables\q_table_{thread_number + 10}.pkl", "rb") as f:
            q_table = np.load(f, allow_pickle=True)
    
    episode_number = 0
    mean_rewards = []
    
    # learning loop
    while episode_number < 20000:
        episode_number += 1
        episode_number % 100 == 0 and print(f"Episode {episode_number}")
        rewards = []
        sd = random.randint(0, 100)
        sx, sy, sw = env.reset(seed=sd)[0]

        sx = np.digitize(sx, x)
        sy = np.digitize(sy, y)
        sw = np.digitize(sw, w)

        steps = 0
        # episode loop
        while steps < 1000 or is_training == False:
            # behavioral policy will be epsilon greedy
            if epsilon > random.uniform(0, 1):
                # take random action
                action_index = random.randint(0, divide - 1) 
            else:
                # take greedy action
                action_index = np.argmax(q_table[sx, sy, sw, :])
            
            
            action = a[action_index]
           
            state, reward, _, _, _ = env.step([action])

            rewards.append(reward)
            state[0] = min(divide - 1, np.digitize(state[0], x))
            state[1] = min(divide - 1, np.digitize(state[1], y))
            state[2] = min(divide - 1, np.digitize(state[2], w))
           
            # update Q-table
            q_table[sx, sy, sw, action_index] = q_table[sx, sy, sw, action_index] + \
            alpha*(reward + gamma*np.max(q_table[np.int64(state[0]), np.int64(state[1]), np.int64(state[2]), :]) - q_table[sx, sy, sw, action_index])

            sx = np.int64(state[0])
            sy = np.int64(state[1])
            sw = np.int64(state[2])

            steps += 1
        
        epsilon -= epsilon_decrease
        epsilon = max(epsilon, epsilon_min)

        episode_number % sample_frequency == 0 and mean_rewards.append(np.mean(rewards))
        episode_number % (sample_frequency * 10) == 0 and print(f"Mean reward: {np.mean(rewards)}")

    with open(rf"q_tables\q_table_{thread_number + 10}.pkl", "wb") as f:
        pickle.dump(q_table, f)

    graph = [range(0, episode_number + 1, sample_frequency), mean_rewards]
    
    with open(rf"pendulum_plots\graphs\graph_{thread_number + 10}.pkl", "wb") as f:
        pickle.dump(graph, f)
                

if __name__ == "__main__":
    threads = []

    # learn(False, 3)

    for i in range(1, 4):
        t = threading.Thread(target=learn, args=(True, i,))
        threads.append(t)
    
    for t in threads:
        t.start()
    
    for t in threads:
        t.join()