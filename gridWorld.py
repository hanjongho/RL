import random
import numpy as np

'''
중간에 벽을 쳤을때 (3,3)까지 가는데 걸리는 평균 횟수 구하기
몬테카를로, TD 방식
'''

class GridWorld():
    def __init__(self):
        self.x = 0
        self.y = 0

    def step(self, a):
        if a == 0:
            if (self.y < 3 and not(self.x == 1 and self.y+1 == 1) and not(self.x == 2 and self.y+1 == 1)):
                self.y += 1
        elif a == 1:
            if (self.y > 0 and not(self.x == 1 and self.y-1 == 1) and not(self.x == 2 and self.y-1 == 1)):
                self.y -= 1
        elif a == 2:
            if (self.x > 0 and not(self.x-1 == 1 and self.y == 1) and not(self.x-1 == 2 and self.y == 1)):
                self.x -= 1
        elif a == 3:
            if (self.x < 3 and not(self.x+1 == 1 and self.y == 1) and not(self.x+1 == 2 and self.y == 1)):
                self.x += 1

        reward = -1
        done = self.checkArrived()
        return (self.x, self.y), reward, done

    def checkArrived(self):
        if self.x == 3 and self.y == 3:
            return True
        return False

    def getState(self):
        return (self.x, self.y)

    def reset(self):
        self.x = 0
        self.y = 0
        return (self.x, self.y)

class Agent():
    def __init__(self):
        pass

    def selectDirection(self):
        coin = random.random()
        if coin < 0.25:
            action = 0
        elif coin < 0.5:
            action = 1
        elif coin < 0.75:
            action = 2
        elif coin < 1:
            action = 3
        return action

# MC
def MC():
    env = GridWorld()
    agent = Agent()
    data = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
    gamma = 1.0
    alpha = 0.01

    for k in range(10000):
        done = False
        history = []
        while not done:
            action = agent.selectDirection()
            (x, y), reward, done = env.step(action)
            history.append((x, y, reward))
        env.reset()

        cum_reward = 0
        for transition in history[::-1]:
            x, y, reward = transition
            data[x][y] = data[x][y] + alpha * (cum_reward - data[x][y])
            cum_reward = cum_reward + gamma * reward

    np.set_printoptions(precision=4)
    print(np.array(data))

# TD
def TD():
    env = GridWorld()
    agent = Agent()
    data = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
    gamma = 1.0
    alpha = 0.01

    for k in range(10000):
        done = False
        while not done:
            x, y = env.getState()
            action = agent.selectDirection()
            (x_prime, y_prime), reward, done = env.step(action)
            x_prime, y_prime = env.getState()

            data[x][y] = data[x][y] + alpha*(reward+gamma*data[x_prime][y_prime]-data[x][y])

        env.reset()

    np.set_printoptions(precision=4)
    print(np.array(data))

if __name__ == '__main__':
    MC()
    print('\n\n')
    TD()