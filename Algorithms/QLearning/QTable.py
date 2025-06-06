"""
This part of code is the Q learning brain, which is a brain of the agent.
All decisions are made in here.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
import pandas as pd


class QLearningTable:
    # QLearningTable(actions)
    def __init__(self, actions, learning_rate=0.001, reward_decay=0.99, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        # q_table --> an instance of pandas.DataFrame class
        # columns: actions when creating instance --- QLearningTable(actions)
        # methods: 
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation, False)

        if np.random.uniform() < self.epsilon:
            state_action = self.q_table.loc[observation, :]
            state_action = state_action.reindex(np.random.permutation(state_action.index))  # some actions have same value
            action = state_action.idxmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_, print_):
        # s, s_ --- string of state
        # a --- action
        # r --- reward
        self.check_state_exist(s_, print_)

        q_predict = self.q_table.loc[s, a]

        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update

    def check_state_exist(self, state, print_=False):
        if state in self.q_table.index:
            print(f"State EXISTS - \tQ_TABLE with shape - {self.q_table.shape}")
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table._append(
                pd.Series([0]*len(self.actions), index=self.q_table.columns, name=state,)
            )
            if print_:
                print(f"\tQ_TABLE with shape - {self.q_table.shape} ")
                print("\tSTATE: ", state)
