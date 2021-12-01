# %%
import datetime
import json
import random

from colorama import Fore, Style
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.advanced_activations import PReLU
import matplotlib.pyplot as plt
import numpy as np

from ground_truth import (
    check_x_set, gen_W_hat, get_w_hat_t, get_x_vector, get_W_hat_rows
)
from utils import scalar, sum_to_S

# Main obstacle with this problem: action space cannot shrink, making training
# MUCH longer --> agent will have to learn not to repeat
# For now, ignore repetitions and don't count them in episodes

np.random.seed(1)

N = 8
K = 2

epsilon = 1.0


class Environment:

    def __init__(self, _N, _K, _x):
        self.N = _N
        self.K = _K
        self.x = _x

        W = np.array([get_w_hat_t(i, N) for i in range(self.N)])
        self.row_map = {k: v for k, v in enumerate(W)}
        self.W_hat = gen_W_hat(self.N)

        to_remove = get_W_hat_rows(W, self.W_hat)
        self.W = np.delete(W, to_remove, 0)

        self.action_space = list(set(range(N)) - set(to_remove))
        self.solutions = self.get_solutions(self.W_hat @ self.x)
        self.n_solutions = len(self.solutions)

    def reset(self):
        self.min_reward = int(-0.5 * ((self.N - np.log2(self.N)) ** 2))
        self.total_reward = 0.0

        W = np.array([get_w_hat_t(i, N) for i in range(self.N)])
        self.row_map = {k: v for k, v in enumerate(W)}
        self.W_hat = gen_W_hat(self.N)

        to_remove = get_W_hat_rows(W, self.W_hat)
        self.W = np.delete(W, to_remove, 0)

        self.action_space = list(set(range(N)) - set(to_remove))
        self.solutions = self.get_solutions(self.W_hat @ self.x)
        self.n_solutions = len(self.solutions)
        self.state = self.W_hat, self.W_hat @ self.x

        self.picked = set()

    def update_state(self, action):
        """

        :param action: Index of any of the rows that can be sampled
        :return:
        """
        sampled_row = self.row_map[action]
        sampled_idx = np.where(np.all(self.W == sampled_row, axis=1))
        self.W_hat = np.append(self.W_hat, [sampled_row], 0)
        self.W = np.delete(self.W, sampled_idx, 0)
        self.state = self.W_hat, self.W_hat @ self.x

    def get_solutions(self, y):
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

    def reduce_solutions(self):
        valid_solutions = [list(map(int, _x)) for _x in check_x_set(
            self.solutions,
            self.W_hat,
            self.W_hat @ self.x
        )]
        self.solutions = valid_solutions
        return self.solutions

    def reward(self, action):
        prev_solutions = self.n_solutions
        # print(f'Previous amount of solutions: {prev_solutions}')
        n_solutions = len(self.reduce_solutions())
        self.n_solutions = n_solutions
        # print(f'New amount of solutions: {n_solutions}')
        if action in self.picked:
            # print(f'{action} already in {self.picked}')
            return self.min_reward - 1
        if n_solutions == prev_solutions:
            return -0.75
        if n_solutions < prev_solutions:
            return -0.5
        if n_solutions == 1:
            return 1.0

        # if n_solutions == 0:
        #     return self.min_reward - 1

    def act(self, action):
        self.update_state(action)
        reward = self.reward(action)
        self.total_reward += reward
        self.picked.add(action)
        status = self.status()
        env_state = self.observe()
        if len(env_state) > self.N ** 2:
            status = 'Did not converge'
        res = env_state, reward, status
        return res

    def observe(self):
        flattened_W_hat = self.state[0].reshape((1, -1))
        padding = np.array([-1] * (self.N ** 2 - flattened_W_hat.shape[1]))
        return np.array([np.append(flattened_W_hat[0], padding)])

    def status(self):
        if self.total_reward < self.min_reward or len(self.W_hat) > N:
            return 'Did not converge'
        if self.n_solutions == 1:
            return 'Converged'
        return 'Calculating...'

    def valid_actions(self):
        W = np.array([get_w_hat_t(i, N) for i in range(self.N)])
        W_hat_bits = [list(reversed(row)) for row in self.W_hat]
        W_idx = [np.where(np.all(W == row, axis=1)) for row in W_hat_bits]
        W_idx = [i[0][0] for i in W_idx if i[0].size > 0]
        valid_actions = set(range(len(W))) - set(list(W_idx))
        return list(valid_actions)


def run_sampling(_model, _env):
    _env.reset()
    env_state = _env.observe()

    while True:
        prev_env_state = env_state

        q = _model.predict(prev_env_state)
        action = np.argmax(q[0])

        env_state, reward, status = _env.act(action)

        if status == 'Converged':
            return True
        elif status == 'Did not converge':
            return False


class QLAgent:

    def __init__(self, _model, max_memory=100, discount=0.999):
        self.model = _model  # A neural network model
        self.max_memory = max_memory  # Maximal length of episodes to keep
        self.discount = discount
        self.memory = []
        self.n_actions = _model.output_shape[-1]

    def remember(self, episode):
        # episode = [env_state, action, reward, next_state, status]
        # env_state == 1D W_hat
        self.memory.append(episode)
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def predict(self, env_state):
        return self.model.predict(env_state)[0]

    def get_data(self, data_size=10):
        env_size = self.memory[0][0].shape[1]
        mem_size = len(self.memory)
        data_size = min(mem_size, data_size)
        inputs = np.zeros((data_size, env_size))
        targets = np.zeros((data_size, N))

        for i, j in enumerate(np.random.choice(
                range(mem_size), data_size, replace=False)
        ):
            env_state, action, reward, next_state, status = self.memory[j]
            inputs[i] = env_state
            if status:
                targets[i, action] = reward
                return inputs, targets
            targets[i] = self.predict(env_state)
            Q_sa = np.max(self.predict(next_state))
            targets[i, action] = reward + self.discount * Q_sa

        return inputs, targets


def train_ql(_model, _N, _K, _x, **config):
    global epsilon
    n_epochs = config.get('n_epochs', 15000)
    max_memory = config.get('max_memory', 1000)
    data_size = config.get('data_size', 50)
    weights_file = config.get('weights_file', '')
    name = config.get('name',
                      f'Q_model_N{_N}_K{_K}_x{"".join([str(b) for b in _x])}')
    eps_threshold = config.get('eps_threshold', 0.8)  # Win rate threshold to change epsilon
    learning_rate = config.get('learning_rate', 0.001)
    start_time = datetime.datetime.now()

    if weights_file:
        print('Loading weights from file: %s' % (weights_file,))
        _model.load_weights(weights_file)

    environment = Environment(_N, _K, _x)
    agent = QLAgent(_model, max_memory=max_memory)

    win_history = []
    history_window_size = (_N ** 2) // 2
    # history_window_size = _N * 2
    win_rate = 0.0

    win_rate_history = []
    loss_history = []
    epsilon_history = []

    for epoch in range(n_epochs):
        _loss = 0.0
        # sample_row_idx = random.choice(list(environment.valid_actions()))
        environment.reset()
        done = False

        env_state = environment.observe()

        n_episodes = 0

        while not done:
            valid_actions = environment.valid_actions()
            # print(f'EPOCH {epoch}: {valid_actions}')
            if not valid_actions:
                break
            prev_env_state = env_state

            # Epsilon-greedy strategy
            if np.random.rand() < epsilon:
                action = random.choice(valid_actions)
                # print(f'{action} randomly')
            else:
                action = np.argmax(agent.predict(prev_env_state))
                # print(f'{action} from prediction')

            # Apply action, get reward, new environment state
            env_state, reward, status = environment.act(action)
            n_episodes += 1

            if status == 'Converged':
                if n_episodes <= int(K * np.log2(N) / np.log2(K + 1)):
                    print(f'{Fore.GREEN}Converged in sufficient (<= {int(K * np.log2(N) / np.log2(K + 1))}) episodes'
                          f'{Style.RESET_ALL}')
                    win_history.append(1)
                else:
                    print(f'{Fore.MAGENTA}Converged in insufficient ({n_episodes}) episodes'
                          f'{Style.RESET_ALL}')
                    win_history.append(0)
                done = True
            elif status == 'Did not converge':
                print(f'{Fore.RED}Did not converge'
                      f'{Style.RESET_ALL}')
                win_history.append(0)
                done = True
            else:
                done = False

            # Store episode
            episode = [prev_env_state, action, reward, env_state, done]

            agent.remember(episode)

            inputs, targets = agent.get_data(data_size=data_size)
            h = _model.fit(
                inputs,
                targets,
                epochs=8,
                batch_size=16,
                verbose=0
            )
            _loss = _model.evaluate(inputs, targets, verbose=0)

        if len(win_history) > history_window_size:
            win_rate = sum(
                win_history[-history_window_size:]
            ) / history_window_size

        dt = datetime.datetime.now() - start_time
        t = format_time(dt.total_seconds())
        template = 'Epoch: {:03d}/{:d} | Loss: {:.4f} | Episodes: {:d} | ' \
                   'Win count: {:d} | Win rate: {:.3f} | Time: {}'
        print(template.format(epoch,
                              n_epochs - 1,
                              _loss,
                              n_episodes,
                              sum(win_history),
                              win_rate,
                              t))
        epsilon_history.append(epsilon)
        print(f'epsilon = {epsilon}\n')
        if win_rate > eps_threshold:
            epsilon -= learning_rate
            eps_threshold += 0.01
            if eps_threshold > 0.9:
                eps_threshold = 0.9
        win_rate_history.append(win_rate)
        loss_history.append(_loss)
        if sum(win_history[-history_window_size:]) == history_window_size:
            print('Reached 100%% win rate at epoch: %d' % (epoch,))
            break

    h5_file = 'models/' + name + '.h5'
    json_file = 'models/' + name + '.json'
    _model.save_weights(h5_file, overwrite=True)
    with open(json_file, 'w') as out_file:
        json.dump(_model.to_json(), out_file, indent=4)

    end_time = datetime.datetime.now()
    dt = end_time - start_time
    _seconds = dt.total_seconds()
    t = format_time(_seconds)
    print('Files: %s, %s' % (h5_file, json_file))
    print("# Epochs: %d, Max Memory: %d, time: %s" % (
        epoch, max_memory, t)
          )
    return _seconds, win_rate_history, loss_history, epsilon_history


def format_time(seconds):
    if seconds < 400:
        s = float(seconds)
        return '%.1f seconds' % (s,)
    elif seconds < 4000:
        m = seconds / 60.0
        return '%.2f minutes' % (m,)
    else:
        h = seconds / 3600.0
        return '%.2f hours' % (h,)


def build_model(_env, lr=0.001):
    _model = Sequential()
    _model.add(Dense(N * N, input_shape=(N * N,)))
    _model.add(PReLU())
    _model.add(Dense(N * N))
    _model.add(PReLU())
    _model.add(Dense(N))
    _model.compile(optimizer='adam', loss='mse')
    return _model


# %%
x = get_x_vector(N, K)
env = Environment(N, K, x)
model = build_model(env.W)
seconds, rates, loss, eps = train_ql(
    model, N, K, x, n_epochs=200, data_size=32, eps_threshold=0.55, learning_rate=0.01
)

#%%
plt.plot(eps, color='black')
plt.xlabel('Epochs')
plt.ylabel('Epsilon')
plt.title(f'Epsilon Decay Up to 200 Epochs: N = {N}, K = {K}')
plt.show()
