import gymnasium as gym
import numpy as np
import pickle
import random
import threading

sample_frequency = 100


# it can be used to learn or to animate fruits of our learning
def learn(is_training, thread_number):
    env = gym.make("Pendulum-v1", render_mode="human" if not is_training else None)

    divide = 15
    epsilon = 1 if is_training else 0
    epsilon_decrease = 0.01
    epsilon_min = 0.05
    alpha = 0.1 * thread_number
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
        with open(rf"q_tables\q_table_a_01_20k.pkl", "rb") as f:
            q_table = np.load(f, allow_pickle=True)
    
    episode_number = 0
    mean_rewards = []
    
    # learning loop
    while episode_number < 20000:
        episode_number += 1
        if episode_number % 100 == 0:
            print(f"Episode {episode_number}")
        rewards = []
        sd = random.randint(0, 100)
        obs = env.reset(seed=sd)[0]

        sx = np.digitize(obs[0], x)
        sy = np.digitize(obs[1], y)
        sw = np.digitize(obs[2], w)

        # Choose initial action using epsilon-greedy
        if epsilon > random.uniform(0, 1):
            action_index = random.randint(0, divide - 1)
        else:
            action_index = np.argmax(q_table[sx, sy, sw, :])

        steps = 0
        # episode loop
        while steps < 1000 or not is_training:
            action = [a[action_index]]
            next_obs, reward, _, _, _ = env.step(action)
            rewards.append(reward)

            # Discretize next state
            next_sx = min(divide - 1, np.digitize(next_obs[0], x))
            next_sy = min(divide - 1, np.digitize(next_obs[1], y))
            next_sw = min(divide - 1, np.digitize(next_obs[2], w))

            # Choose next action using epsilon-greedy (SARSA)
            if epsilon > random.uniform(0, 1):
                next_action_index = random.randint(0, divide - 1)
            else:
                next_action_index = np.argmax(q_table[next_sx, next_sy, next_sw, :])

            # SARSA update rule
            q_table[sx, sy, sw, action_index] += alpha * (
                reward + gamma * q_table[next_sx, next_sy, next_sw, next_action_index]
                - q_table[sx, sy, sw, action_index]
            )

            # Move to next state and action
            sx, sy, sw = next_sx, next_sy, next_sw
            action_index = next_action_index

            steps += 1
        
        epsilon -= epsilon_decrease
        epsilon = max(epsilon, epsilon_min)

        episode_number % sample_frequency == 0 and mean_rewards.append(np.mean(rewards))
        episode_number % (sample_frequency * 10) == 0 and print(f"Mean reward: {np.mean(rewards)}")

    with open(rf"q_tables\q_table_sarsa_{thread_number}.pkl", "wb") as f:
        pickle.dump(q_table, f)

    graph = [range(0, episode_number + 1, sample_frequency), mean_rewards]
    
    with open(rf"pendulum_plots\graphs\graph_{thread_number}.pkl", "wb") as f:
        pickle.dump(graph, f)
                

def threaded_learning(number_of_threads):
    threads = []
    for i in range(1, number_of_threads + 1):
        t = threading.Thread(target=learn, args=(True, i,))
        threads.append(t)
    for t in threads:
        t.start()
    for t in threads:
        t.join()

if __name__ == "__main__":
    threads = []

    # learn(False, 3)

    # learn(True, 1)

    threaded_learning(2)