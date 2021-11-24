import pprint
import random
from collections import defaultdict

import numpy as np
from colorama import Fore, Style

from ground_truth import (
    get_x_vector, gen_W_hat, get_w_hat_t, get_W_hat_rows
)
from utils import scalar, sum_to_S

N_EPISODES = 15
N = 8
K = 4
PP = pprint.PrettyPrinter(indent=4)

np.random.seed(1)


class Environment:

    def __init__(self, _N, _K):
        self.N = _N
        self.K = _K
        self.x = get_x_vector(self.N, self.K)
        self.W_hat = gen_W_hat(self.N)
        self.y = self.W_hat @ self.x

        W = np.array([get_w_hat_t(i, _N) for i in range(_N)])
        self.idx_map = {k: v for k, v in enumerate(W)}
        to_remove = get_W_hat_rows(W, self.W_hat)

        self.W = np.delete(W, to_remove, 0)
        self.action_space = list(set(range(self.N)) - set(to_remove))
        self.action_num = random.choice(self.action_space)
        self.n_solutions = len(self.get_solutions((self.action_num, self.W_hat, self.y)))
        print(f'Start with {self.n_solutions} solutions')

    def __str__(self):
        return f'{Fore.BLUE}{Style.BRIGHT}ENV: N = {self.N}, K = {self.K}, ' \
               f'x = {self.x}{Style.RESET_ALL}'

    def get_action_space(self):
        return self.action_space

    def get_state(self):
        return self.action_num, self.W_hat, self.y

    def reset(self):


    def step(self, _action, reward_func):
        """
        Update environment based on current action

        :param reward_func: reward function
        :param _action: index of current row to sample from W
        :return:
        """
        sampled_row = self.idx_map[_action]
        sampled_idx = np.where(np.all(self.W == sampled_row, axis=1))
        self.action_space.remove(_action)
        self.W_hat = np.append(self.W_hat, [sampled_row], 0)
        self.y = self.W_hat @ self.x
        _next_state = self.action_num, self.W_hat, self.y
        _reward, _done = reward_func(_next_state)
        self.action_num = _action
        self.W = np.delete(self.W, sampled_idx, 0)
        return _next_state, _reward, _done

    def get_solutions(self, _state):
        W_hat, y = _state[1:]
        print(f'W_hat rows: {len(W_hat)}')
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
        _done = False
        if self.n_solutions == 1:
            _reward = 1.0
            _done = True
        elif self.n_solutions == 0:
            _reward = -1.0
            _done = True
            print(f'{Fore.RED}Did not converge{Style.RESET_ALL}')
        # If agent made no progress
        elif self.n_solutions == prev_solutions:
            _reward = -0.05
        elif self.n_solutions < prev_solutions:
            _reward = -1.0
        else:
            _reward = 0
        print(f'{Fore.BLUE}Current # solutions: {self.n_solutions}{Style.RESET_ALL}')
        return _reward, _done


class QLAgent:

    def __init__(self,
                 W,
                 learning_rate=0.3,
                 discount_factor=0.9,
                 epsilon=0.5
                 ):
        self.W = W  # Each row is a possible action
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: [0.0] * int((N - np.log2(N))))
        self.W_map = {i: k for k, i in enumerate(W)}

    def learn(self, _state, _action, _reward, _next_state):
        """Update Q-function with sample <s, a, r, s'>"""
        q_t = self.q_table[_state][self.W_map[_action]]
        q_t1 = _reward + self.discount_factor * max(self.q_table[_next_state])
        PP.pprint(dict(self.q_table))
        self.q_table[_state][self.W_map[_action]] += self.learning_rate * (q_t1 - q_t)

    def get_action(self, _state, action_space):
        """Get action for state according to Q-table, agent picks action based
        on epsilon-greedy policy"""
        p = np.random.rand()
        if p < self.epsilon:
            print(f'Choosing random action from action space: {action_space}')
            _action = random.choice(action_space)
        else:
            print(f'Choosing locally optimal action from action space'
                  f' {action_space}:')
            state_action = self.q_table[_state]
            if len(action_space) > 1:
                idx = self.arg_max(state_action, list(map(self.W_map.get, action_space)))
                print(idx)
                _action = action_space[idx]
            else:
                _action = action_space[0]
        print(f'{Style.BRIGHT}Action chosen: {_action}{Style.RESET_ALL}')
        return _action

    @staticmethod
    def arg_max(state_action, action_space_idx):
        max_index_list = []
        max_value = state_action[0]
        for index, value in enumerate(state_action):
            if value > max_value:
                max_index_list.clear()
                max_value = value
                max_index_list.append(index)
            elif value == max_value:
                max_index_list.append(index)
        return random.choice([i for i in max_index_list if i in action_space_idx])


if __name__ == '__main__':
    env = Environment(N, K)
    print(f'\n{env}')
    agent = QLAgent(env.get_action_space())

    for episode in range(N_EPISODES):
        print(f'{Style.BRIGHT}{Fore.CYAN}EPISODE {episode}{Style.RESET_ALL}')
        state = env.get_state()

        while True:
            a_space = env.get_action_space()
            action = agent.get_action(str(state[0]), a_space)
            next_state, reward, done = env.step(action, env.linear_reward)
            agent.learn(str(action), action, reward, str(next_state[0]))
            state = next_state
            if done:
                break

"""
TODO:
Create fixed mapping for row indices so that we can account for changing
size of W_hat
"""