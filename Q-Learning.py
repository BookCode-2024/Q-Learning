# -*- coding: utf-8 -*-
"""
Created on Tue May  7 14:50:59 2024

@author: zhaodf
"""
"""Q Learning to solve a simple world model
Simple deterministic MDP is made of 6 grids (states)
确定性环境中，起始位置位于（0,0），可在如下6格环境中移动，仅在移动到（0,2）位置时得分100，移动到（1,2）位置时得分-100，其余操作不得分。
-------------------------------------------------
|          |           |                      |
|   Start  |           |   Goal(score+100)    |
|  （0,0） |  （0,1）   |    （0,2）           |
|          |           |                      |
-------------------------------------------------
|           |          |                      |
|           |          |   Hole(score-100)    |
|  （1,0）  |  （1,1）  |    （1,2）           |
|           |          |                      |
-------------------------------------------------

 Q-Table（初始值为0）
 以表格形式列出处于不同位置时朝各个方向移动策略的Q值
------------------------------------------------------------------------------------------
|         action |     Move-Left    |    Move-Down    |    Move-Right     |    Move-Up   |
|  state         |                  |                 |                   |              |
------------------------------------------------------------------------------------------
|  （0,0）       |                  |                 |                   |              |
------------------------------------------------------------------------------------------
|  （0,1）       |                  |                 |                   |              |
------------------------------------------------------------------------------------------
|  （0,2）       |                  |                 |                   |              |
------------------------------------------------------------------------------------------
|  （1,0）       |                  |                 |                   |              |
------------------------------------------------------------------------------------------
|  （1,1）       |                  |                 |                   |              |
------------------------------------------------------------------------------------------
|  （1,2）       |                  |                 |                   |              |
------------------------------------------------------------------------------------------
"""
# import argparse
import os
import time
import numpy as np
from collections import deque
from termcolor import colored

class QWorld:
    def __init__(self):
        """Simulated deterministic world made of 6 states.
        Q-Learning by Bellman Equation. 
        """
        # 4 actions
        # 0 - Left, 1 - Down, 2 - Right, 3 - Up

        # 6 states
        self.row = 6
        self.col = 4

        # setup the environment
        self.q_table = np.zeros([self.row, self.col])
        self.init_transition_table()
        self.init_reward_table()

        # 90% exploration(探索), 10% exploitation（利用/开发）
        self.epsilon = 0.9
        # exploration decays by this factor every episode
        self.epsilon_decay = 0.9
        # in the long run, 10% exploration, 90% exploitation
        self.epsilon_min = 0.1

        # reset the environment
        self.reset()
        self.is_explore = True

    """状态编号
            -------------
            | 0 | 1 | 2 |
            -------------
            | 3 | 4 | 5 |
            -------------
    """
    def reset(self):
        """start of episode"""
        self.state = 0
        return self.state

    def is_in_win_state(self):
        """agent wins when the goal is reached"""
        return self.state == 2

    def init_reward_table(self):
        """
        0 - Left, 1 - Down, 2 - Right, 3 - Up
        ----------------
        | 0 | 0 | 100  |
        ----------------
        | 0 | 0 | -100 |
        ----------------
        """
        self.reward_table = np.zeros([self.row, self.col])
        self.reward_table[1, 2] = 100.
        self.reward_table[4, 2] = -100.

    def init_transition_table(self):
        """状态编号
        -------------
        | 0 | 1 | 2 |
        -------------
        | 3 | 4 | 5 |
        -------------
        """
        self.transition_table = np.zeros([self.row, self.col], dtype=int)

        self.transition_table[0, 0] = 0##各个动作引起的状态变化
        self.transition_table[0, 1] = 3
        self.transition_table[0, 2] = 1
        self.transition_table[0, 3] = 0

        self.transition_table[1, 0] = 0
        self.transition_table[1, 1] = 4
        self.transition_table[1, 2] = 2
        self.transition_table[1, 3] = 1

        # 达到目标
        self.transition_table[2, 0] = 2
        self.transition_table[2, 1] = 2
        self.transition_table[2, 2] = 2
        self.transition_table[2, 3] = 2

        self.transition_table[3, 0] = 3
        self.transition_table[3, 1] = 3
        self.transition_table[3, 2] = 4
        self.transition_table[3, 3] = 0

        self.transition_table[4, 0] = 3
        self.transition_table[4, 1] = 4
        self.transition_table[4, 2] = 5
        self.transition_table[4, 3] = 1

        # 进洞
        self.transition_table[5, 0] = 5
        self.transition_table[5, 1] = 5
        self.transition_table[5, 2] = 5
        self.transition_table[5, 3] = 5
        
    
    def step(self, action):
        """在设置的环境中执行动作
        变量:
            动作
        返回：
            下一步环境状态
            奖励
            是否到达终止状态
        """
        # 根据给定的状态和动作决定下一个状态
        next_state = self.transition_table[self.state, action]
        # 如果下一个状态到达目标或者进洞，done=true
        done = next_state == 2 or next_state == 5
        # 赋予奖励
        reward = self.reward_table[self.state, action]
        # 环境进入新状态
        self.state = next_state
        return next_state, reward, done


    def act(self):
        """决定下一个动作来自Q-table（利用/开发）或者随机（探索）
        """
        # action is from exploration
        if np.random.rand() <= self.epsilon:
            # explore - do random action
            self.is_explore = True
            return np.random.choice(4,1)[0]

        # 选择最大Q值的动作
        self.is_explore = False
        action = np.argmax(self.q_table[self.state])
        return action


    def update_q_table(self, state, action, reward, next_state):
        # Q(s, a) = reward + gamma * max_a' Q(s', a')
        q_value = 0.9 * np.amax(self.q_table[next_state])
        q_value += reward
        self.q_table[state, action] = q_value


    def print_q_table(self):
        """UI to dump Q Table contents"""
        print("Q-Table (Epsilon: %0.2f)" % self.epsilon)
        print(self.q_table)


    def update_epsilon(self):
        """update Exploration-Exploitation mix"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def print_cell(self, row=0):
        """UI to display agent moving on the grid"""
        print("")
        for i in range(13):
            j = i - 2
            if j in [0, 4, 8]: 
                if j == 8:
                    if self.state == 2 and row == 0:
                        marker = "\033[4mG\033[0m"
                    elif self.state == 5 and row == 1:
                        marker = "\033[4mH\033[0m"
                    else:
                        marker = 'G' if row == 0 else 'H'
                    color = self.state == 2 and row == 0
                    color = color or (self.state == 5 and row == 1)
                    color = 'red' if color else 'blue'
                    print(colored(marker, color), end='')
                elif self.state in [0, 1, 3, 4]:
                    cell = [(0, 0, 0), (1, 0, 4), (3, 1, 0), (4, 1, 4)]
                    marker = '_' if (self.state, row, j) in cell else ' '
                    print(colored(marker, 'red'), end='')
                else:
                    print(' ', end='')
            elif i % 4 == 0:
                    print('|', end='')
            else:
                print(' ', end='')
        print("")


    def print_world(self, action, step):
        """UI to display mode and action of agent"""
        actions = { 0: "(Left)", 1: "(Down)", 2: "(Right)", 3: "(Up)" }
        explore = "Explore" if self.is_explore else "Exploit"
        print("Step", step, ":", explore, actions[action])
        for _ in range(13):
            print('-', end='')
        self.print_cell()
        for _ in range(13):
            print('-', end='')
        self.print_cell(row=1)
        for _ in range(13):
            print('-', end='')
        print("")


def print_episode(episode, delay=1):
    """UI to display episode count
    Arguments:
        episode (int): episode number
        delay (int): sec delay

    """
    os.system('clear')
    for _ in range(13):
        print('=', end='')
    print("")
    print("Episode ", episode)
    for _ in range(13):
        print('=', end='')
    print("")
    time.sleep(delay)


def print_status(q_world, done, step, delay=1):
    """UI to display the world, 
        delay of 1 sec for ease of understanding
    """
    os.system('clear')
    q_world.print_world(action, step)
    q_world.print_q_table()
    if done:
        print("-------EPISODE DONE--------")
        delay *= 2
    time.sleep(delay)


# main loop of Q-Learning
if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # help_ = "Trains and show final Q Table"
    # parser.add_argument("-t",
    #                     "--train",
    #                     help=help_,
    #                     action='store_true')
    # args = parser.parse_args()
    #
    # if args.train:
    #     maxwins = 2000
    #     delay = 0
    # else:
    #     maxwins = 10
    #     delay = 1

    maxwins = 10
    delay = 1
    wins = 0
    episode_count = 10 * maxwins
    # scores (max number of steps bef goal) - good indicator of learning
    scores = deque(maxlen=maxwins)
    q_world = QWorld()
    step = 1

    # state, action, reward, next state iteration
    for episode in range(episode_count):
        state = q_world.reset()
        done = False
        print_episode(episode, delay=delay)
        while not done:
            action = q_world.act()
            next_state, reward, done = q_world.step(action)
            q_world.update_q_table(state, action, reward, next_state)
            print_status(q_world, done, step, delay=delay)
            state = next_state
            # if episode is done, perform housekeeping
            if done:
                if q_world.is_in_win_state():
                    wins += 1
                    scores.append(step)
                    if wins > maxwins:
                        print(scores)
                        exit(0)
                # Exploration-Exploitation is updated every episode
                q_world.update_epsilon()
                step = 1
            else:
                step += 1

    print(scores)
    q_world.print_q_table()