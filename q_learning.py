import random
from collections import defaultdict

import numpy as np

from ground_truth import (
    get_x_vector, gen_W_hat, get_w_hat_t, get_W_hat_rows
)
from utils import scalar, sum_to_S

N_EPISODES = 1000
N = 8
K = 3


class Environment:

    def __init__(self, N, K):
        self.N = N
        self.K = K
        self.x = get_x_vector(self.N, self.K)
        self.W_hat = gen_W_hat(self.N)
        self.y = self.W_hat @ self.x

        W = np.array([get_w_hat_t(i, N) for i in range(N)])
        to_remove = get_W_hat_rows(W, self.W_hat)

        self.W = np.delete(W, to_remove, 0)
        self.n_solutions = len(self.get_solutions((self.W_hat, self.y)))
        self.action_space = list(range(len(self.W)))  # All indices of W rows

    def get_W(self):
        return self.W

    def get_state(self):
        return self.W_hat, self.y

    def step(self, _action, reward_func):
        """
        Update environment based on current action

        :param reward_func: reward function
        :param _action: index of current row to sample from W
        :return:
        """
        self.action_space.remove(_action)
        sampled_row = self.W[_action]
        self.W = np.delete(self.W, _action, 0)
        self.W_hat = np.append(self.W_hat, [sampled_row], 0)
        self.y = self.W_hat @ self.x
        next_state = np.array([self.W_hat, self.y])
        reward, done = reward_func(next_state)
        return next_state, reward, done

    def get_solutions(self, _state):
        W_hat, y = _state
        S = scalar(y)
        solutions = []
        for partition in sum_to_S(S, self.K):
            if len(set(partition)) == len(partition) and \
                    max(partition) < self.N:
                partition = sorted(partition)
                if partition not in solutions:
                    solutions.append(partition)
        x_vectors = []
        for sol in solutions:
            tmp = np.zeros(self.N)
            tmp[sol] = 1
            x_vectors.append(tmp)
        return x_vectors

    def linear_reward(self, _state):
        prev_solutions = self.n_solutions
        self.n_solutions = len(self.get_solutions(_state))
        done = False
        if self.n_solutions == 1:
            reward = 100
            done = True
        # If agent made no progress
        elif self.n_solutions == prev_solutions:
            reward = -100
        else:
            reward = -50
        return reward, done


class QLAgent:

    def __init__(self,
                 W,
                 learning_rate=0.01,
                 discount_factor=0.8,
                 epsilon=0.1
                 ):
        self.W = W  # Each row is a possible action
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

    def learn(self, _state, action, reward, next_state):
        """Update Q-function with sample <s, a, r, s'>"""
        q_t = self.q_table[_state][action]
        q_t1 = reward + self.discount_factor * max(self.q_table[next_state])
        self.q_table[_state][action] += self.learning_rate * (q_t1 - q_t)

    def get_action(self, _state):
        """Get action for state according to Q-table, agent picks action based
        on epsilon-greedy policy"""
        p = np.random.rand()
        if p < self.epsilon:
            _action = np.random.choice(len(self.W), 1, replace=False)
        else:
            state_action = self.q_table[_state]
            _action = self.arg_max(state_action)
        return _action

    @staticmethod
    def arg_max(state_action):
        max_index_list = []
        max_value = state_action[0]
        for index, value in enumerate(state_action):
            if value > max_value:
                max_index_list.clear()
                max_value = value
                max_index_list.append(index)
            elif value == max_value:
                max_index_list.append(index)
        return random.choice(max_index_list)


if __name__ == '__main__':
    env = Environment(N, K)
    agent = QLAgent(env.get_W())

    for episode in range(N_EPISODES):
        state = env.get_state()

        while True:
            action = agent.get_action(state)
            print(action)