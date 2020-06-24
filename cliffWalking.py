import numpy as np

rows = 5
cols = 12
iteration = 10

def create_table():
    q_table = np.zeros((4, cols * rows))

    return q_table

def epsilon_greedy_policy(state, q_table, epsilon):
    decide = np.random.random() # 0.0 ~ 1.0
    if (decide < epsilon):
        action = np.random.choice(4)  # UP = 0, LEFT = 1, RIGHT = 2, DOWN = 3
    else:
        action = np.argmax(q_table[:, state])

    return action

def move_agent(agent, action):
    (posX, posY) = agent
    if ((action == 0) and posX > 0): # UP
        posX -= 1
    if ((action == 1) and posY > 0): # LEFT
        posY -= 1
    if ((action == 2) and (posY < cols-1)): # RIGHT
        posY += 1
    if ((action) == 3 and (posX < rows-1)): # DOWN
        posX += 1
    agent = (posX, posY)

    return agent

def get_state(agent, q_table):
    (posX, posY) = agent
    state = cols * posX + posY
    maximum_state_value = np.amax(q_table[:, state])

    return state, maximum_state_value

def get_reward(state):
    game_end = False
    reward = -1
    if (state == cols*rows-1):
        game_end = True
        reward = 10
    if (state >= (rows-1)*cols+1 and state <= cols*rows-2):
        game_end = True
        reward = -100

    return reward, game_end

def visited_env(agent, env):
    (posY, posX) = agent
    env[posY][posX] = 1

    return env

def qlearning(num_episodes=iteration, gamma_discount=0.9, alpha=0.5, epsilon=0.1):
    reward_cache, arrival_check = list(), list()
    step_by_step = []
    q_table = create_table()
    for episode in range(0, num_episodes):
        agent = (rows-1, 0)
        game_end = False
        reward_cum = step_cum = 0
        env = np.zeros((rows, cols))
        env = visited_env(agent, env)
        while (not(game_end)):
            step_by_step.append([])
            state, _ = get_state(agent, q_table)
            action = epsilon_greedy_policy(state, q_table, epsilon)
            agent = move_agent(agent, action)
            step_by_step[episode].append(agent)
            env = visited_env(agent, env)
            step_cum += 1
            next_state, max_next_state_value = get_state(agent, q_table)
            reward, game_end = get_reward(next_state)
            reward_cum += reward
            q_table[action, state] = q_table[action, state] + 0.5 * (reward + (0.9 * max_next_state_value) - q_table[action, state])

        reward_cache.append(reward_cum)
        if(step_by_step[episode][-1] == (rows-1, cols-1)):
            arrival_check.append("True")
        else:
            arrival_check.append("False")
        if(episode > num_episodes-2):
            print("Q-learning")
            print(env)

    return q_table, reward_cache, step_by_step, arrival_check

def sarsa(num_episodes=iteration, gamma=0.9, alpha=0.5, epsilon=0.1):
    reward_cache,arrival_check = list(), list()
    step_by_step = []
    q_table = create_table()
    for episode in range(0, num_episodes):
        agent = (rows-1, 0)
        game_end = False
        reward_cum, step_cum = 0, 0
        env = np.zeros((rows, cols))
        env = visited_env(agent, env)
        state, _ = get_state(agent, q_table)
        action = epsilon_greedy_policy(state, q_table, epsilon)
        while (not(game_end)):
            step_by_step.append([])
            agent = move_agent(agent, action)
            step_by_step[episode].append(agent)
            env = visited_env(agent, env)
            step_cum += 1
            next_state, _ = get_state(agent, q_table)
            reward, game_end = get_reward(next_state)
            reward_cum += reward
            next_action = epsilon_greedy_policy(next_state, q_table, epsilon)
            next_state_value = q_table[next_action][next_state]
            q_table[action, state] = q_table[action, state] + alpha * (reward + (gamma * next_state_value) - q_table[action, state])
            state = next_state
            action = next_action
        reward_cache.append(reward_cum)
        if (step_by_step[episode][-1] == (3, 11)):
            arrival_check.append("True")
        else:
            arrival_check.append("False")
        if(episode > num_episodes-2):
            print("SARSA")
            print(env)

    return q_table, reward_cache, step_by_step, arrival_check

def retrieve_environment(q_table, action):
    env = q_table[action, :].reshape((rows, cols))

    for n in range(rows):
        for m in range(cols):
            print("%7.2f" %env[n][m], end = " ")
        print("")

def main():
    print("iteration :", iteration)
    q_table_SARSA, reward_cache_SARSA, step_by_step_SARSA, arrival_check_SARSA = sarsa()
    q_table_qlearning, reward_cache_qlearning , step_by_step_qlearning, arrival_check_qlearning = qlearning()

    if (0):
        if (0):
            print("\nQlearningStateReward       UP   LEFT    RIGHT   DOWN")
            for n in range(0, 4):
                retrieve_environment(q_table_qlearning, n)
                print("")
        for n in range(0, iteration):
            print("step cnt :", "%3d" %len(step_by_step_qlearning[n]), "cumulative reward : %4d" %reward_cache_qlearning[n], "Arrival : %5s  " %arrival_check_qlearning[n], step_by_step_qlearning[n])

    if (0):
        if (0):
            print("\nSarsaStateReward       UP   LEFT    RIGHT   DOWN")
            for n in range(0, 4):
                retrieve_environment(q_table_SARSA, n)
                print("")
        for n in range(0, iteration):
            print("step cnt :", "%3d" %len(step_by_step_SARSA[n]), "cumulative reward : %4d" %reward_cache_SARSA[n], "Arrival : %5s  " %arrival_check_SARSA[n], step_by_step_SARSA[n])

if __name__ == "__main__":
    main()