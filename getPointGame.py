
'''

게임 룰)
0. agent는 (0,0) 에서 게임시작함.
1. 컴파일할때마다 랜덤위치에 아이템(+n) m개를 만든다.
2. 에피소드별 다 먹을때까지 agent는 움직이고 그 경로를 저장해둔다.
+++ 2020 05 25
최단경로로 아이템을 먹되, 순찰한 경로는 -reward를 준다. 랜덤으로 횟수제한 제한사항 방문했던곳은 -값 주기.
벽 또한 랜덤으로 생성하기때문에 하려했으나 환경을 (agent가 움직일 수 없는) 제대로 설정안하면 무한루프에서 못 나옴

'''

import numpy as np
import random

iteration = 500
rows = 5
cols = 10
items = 3

def epsilon_greedy_policy(state, q_table, epsilon):
    decide_explore_exploit = np.random.random() # 0.0 ~ 1.0
    if (decide_explore_exploit < epsilon):
        action = np.random.choice(4)            # UP = 0, LEFT = 1, RIGHT = 2, DOWN = 3
    else:
        action = np.argmax(q_table[:, state])

    return action

def move_agent(agent, action, env):
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

def get_reward(state, env, item_count):
    game_end = False
    reward = -1
    posX = int(state/cols)
    posY = state % cols

    if (env[posX][posY] == 2):
        env[posX][posY] = 0
        item_count -= 1
        reward = 50

    if (item_count == 0):
        game_end = True

    return reward, game_end, item_count

def getRandomPosition(env, n):
    for i in range(n):
        env[random.randint(0, rows-1)][random.randint(0, cols-1)] = 2

def qlearning(num_episodes = iteration, gamma_discount = 0.9, alpha = 0.5, epsilon = 0.1):
    q_table = np.zeros((4, cols * rows))
    tmp, env = np.zeros((rows, cols)), np.zeros((rows, cols))
    getRandomPosition(tmp, items)
    reward_cache, step_by_step = [], []
    print(tmp)
    for episode in range(0, num_episodes):
        game_end = False
        item_count = items
        agent = (0, 0)
        env = tmp.copy()
        reward_cum = 0
        while (not(game_end)):
            step_by_step.append([])
            state, _ = get_state(agent, q_table)
            action = epsilon_greedy_policy(state, q_table, epsilon)
            agent = move_agent(agent, action, env)
            step_by_step[episode].append(agent)
            next_state, max_next_state_value = get_state(agent, q_table)
            reward, game_end, item_count = get_reward(next_state, env, item_count)
            reward_cum += reward
            q_table[action, state] = q_table[action, state] + alpha * (reward + (gamma_discount * max_next_state_value) - q_table[action, state])

        reward_cache.append(reward_cum)

    return q_table, reward_cache, step_by_step

def retrieve_environment(q_table, action):
    env = q_table[action, :].reshape((rows, cols))
    for n in range(rows):
        for m in range(cols):
            print("%7.2f" %env[n][m], end = " ")
        print("")

def main():
    print("iteration :", iteration)
    q_table_qlearning, reward_cache_qlearning , step_by_step_qlearning = qlearning()

    if(1):
        if(1):
            print("\nQlearningStateReward\nUP   LEFT    RIGHT   DOWN")
            for n in range(0, 4):
                retrieve_environment(q_table_qlearning, n)
                print("")

        for n in range(0, iteration):
            print("step cnt :", "%4d" %len(step_by_step_qlearning[n]), "cumulative reward : %4d" %reward_cache_qlearning[n], step_by_step_qlearning[n])

if __name__ == "__main__":
    main()