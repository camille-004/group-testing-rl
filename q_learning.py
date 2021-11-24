# %%
import datetime
import json
import random

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.advanced_activations import PReLU
import numpy as np

from ground_truth import (
    check_x_set, gen_W_hat, get_w_hat_t, get_x_vector, get_W_hat_rows
)
from utils import scalar, sum_to_S

# %%
np.random.seed(1)

N = 8
K = 4

epsilon = 0.1


# %%
class Environment:

    def __init__(self, _N, _K, _x):
        self.N = _N
        self.K = _K
        self.x = _x

        W = np.array([get_w_hat_t(i, N) for i in range(self.N)])
        self.row_map = {k: v for k, v in enumerate(W)}
        self.W_hat = gen_W_hat(N)

        to_remove = get_W_hat_rows(W, self.W_hat)
        self.W = np.delete(W, to_remove, 0)

        self.action_space = list(set(range(N)) - set(to_remove))
        self.solutions = self.get_solutions(self.W_hat @ self.x)
        self.n_solutions = len(self.solutions)

    def reset(self, start_idx):
        W = np.array([get_w_hat_t(i, N) for i in range(self.N)])
        self._W = np.copy(self.W)
        self.min_reward = -0.8 * (self.N - np.log2(self.N))
        self.total_reward = 0
        self.W_hat = gen_W_hat(N)

        sampled_row = W[start_idx]
        sampled_idx = np.where(np.all(self._W == sampled_row, axis=1))
        self._W = np.delete(self._W, sampled_idx, 0)
        self.W_hat = np.append(self.W_hat, [sampled_row], 0)

        self.state = self.W_hat, self.W_hat @ self.x

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

    def reward(self):
        prev_solutions = self.n_solutions
        print(f'Previous amount of solutions: {prev_solutions}')
        n_solutions = len(self.reduce_solutions())
        self.n_solutions = n_solutions
        print(f'New amount of solutions: {n_solutions}')
        if n_solutions == prev_solutions:
            return -0.75
        if n_solutions < prev_solutions:
            return -0.05
        if n_solutions == 1:
            return 1.0
        if n_solutions == 0:
            return self.min_reward - 1

    def act(self, action):
        self.update_state(action)
        reward = self.reward()
        self.total_reward += reward
        status = self.status()
        env_state = self.observe()
        return env_state, reward, status

    def observe(self):
        return self.state[0].reshape((1, -1))

    def status(self):
        if self.total_reward < self.min_reward:
            return 'Did not converge'
        if self.n_solutions == 1:
            return 'Converged'
        return 'Calculating...'

    def valid_actions(self):
        W = np.array([get_w_hat_t(i, N) for i in range(self.N)])
        W_hat_bits = [list(reversed(row)) for row in self.W_hat]
        W_idx = [np.where(np.all(W == row, axis=1))[0][0] for row in W_hat_bits]
        valid_actions = set(range(len(W))) - set(list(W_idx))
        return list(valid_actions)


# %%
def run_sampling(model, _env):
    _env.reset()
    env_state = _env.observe()

    while True:
        prev_env_state = env_state

        q = model.predict(prev_env_state)
        action = np.argmax(q[0])

        env_state, reward, status = _env.act(action)

        if status == 'Converged':
            return True
        elif status == 'Did not converge':
            return False


# %%
class QLAgent:

    def __init__(self, model, max_memory=100, discount=0.95):
        self.model = model  # A neural network model
        self.max_memory = max_memory  # Maximal length of episodes to keep
        self.discount = discount
        self.memory = []
        self.n_actions = model.output_shape[-1]

    def remember(self, episode):
        # episode = [env_state, action, reward, next_state, status]
        # env_state == 1D W_hat
        self.memory.append(episode)
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def predict(self, env_state):
        print(env_state)
        return self.model.predict(env_state)[0]

    def get_data(self, data_size=10):
        env_size = self.memory[0][0].shape[1]
        print(f'Env size: {env_size}')
        mem_size = len(self.memory)
        data_size = min(mem_size, data_size)
        inputs = np.zeros((data_size, env_size))
        targets = np.zeros((data_size, self.n_actions))

        for i, j in enumerate(np.random.choice(
                range(mem_size), data_size, replace=False)
        ):
            env_state, action, reward, next_state, status = self.memory[j]
            inputs[i] = env_state
            targets[i] = self.predict(env_state)
            Q_sa = np.max(self.predict(next_state))
            if status:
                targets[i, action] = reward
            else:
                targets[i, action] = reward + self.discount * Q_sa

        return inputs, targets


# %%

def train_ql(model, _N, _K, _x, **config):
    global epsilon
    n_epochs = config.get('n_epoch', 15000)
    max_memory = config.get('max_memory', 1000)
    data_size = config.get('data_size', 50)
    weights_file = config.get('weights_file', '')
    name = config.get('name', 'model')
    start_time = datetime.datetime.now()

    if weights_file:
        print('Loading weights from file: %s' % (weights_file,))
        model.load_weights(weights_file)

    environment = Environment(_N, _K, _x)
    agent = QLAgent(model, max_memory=max_memory)

    win_history = []
    history_window_size = _N // 2
    win_rate = 0.0

    for epoch in range(n_epochs):
        loss = 0.0
        sample_row_idx = random.choice(list(environment.valid_actions()))
        environment.reset(sample_row_idx)
        done = False

        env_state = environment.observe()

        n_episodes = 0

        while not done:
            valid_actions = environment.valid_actions()
            if not valid_actions:
                break
            prev_env_state = env_state

            # Epsilon-greedy strategy
            if np.random.rand() < epsilon:
                action = random.choice(valid_actions)
            else:
                action = np.argmax(agent.predict(prev_env_state))

            # Apply action, get reward, new environment state
            env_state, reward, status = environment.act(action)

            if status == 'Converged':
                win_history.append(1)
                done = True
            elif status == 'Did not converge':
                win_history.append(0)
                done = True
            else:
                done = False

            # Store episode
            episode = [prev_env_state, action, reward, env_state, done]
            agent.remember(episode)
            n_episodes += 1

            inputs, targets = agent.get_data(data_size=data_size)
            h = model.fit(
                inputs,
                targets,
                epochs=8,
                batch_size=16,
                verbose=0
            )
            loss = model.evaluate(inputs, targets, verbose=0)

        if len(win_history) > history_window_size:
            win_rate = sum(
                win_history[-history_window_size:]
            ) / history_window_size

        dt = datetime.datetime.now() - start_time
        t = format_time(dt.total_seconds())
        template = 'Epoch: {:03d}/{:d} | Loss: {:.4f} | Episodes: {:d} | ' \
                   'Win count: {:d} | Win rate: {:.3f} | time: {}'
        print(template.format(epoch,
                              n_epochs - 1,
                              loss,
                              n_episodes,
                              sum(win_history),
                              win_rate,
                              t))
        if win_rate > 0.9:
            epsilon = 0.05
        if sum(win_history[:-history_window_size]) == history_window_size:
            print('Reached 100%% win rate at epoch: %d' % (epoch,))
            break

        h5_file = name + '.h5'
        json_file = name + '.json'
        model.save_weights(h5_file, overwrite=True)
        with open(json_file, 'w') as out_file:
            json.dump(model.to_json(), out_file)

        end_time = datetime.datetime.now()
        dt = end_time - start_time
        seconds = dt.total_seconds()
        t = format_time(seconds)
        print('Files: %s, %s' % (h5_file, json_file))
        print("n_epochs: %d, max_memory: %d, data: %d, time: %s" % (
            epoch, max_memory, data_size, t)
        )
        return seconds


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


# %%
def build_model(_env, lr=0.001):
    _model = Sequential()
    _model.add(Dense(N * N, input_shape=(N * N,)))
    _model.add(PReLU())
    _model.add(Dense(N * N))
    _model.add(PReLU())
    _model.add(Dense(N - np.log2(N)))
    _model.compile(optimizer='adam', loss='mse')
    return _model


# %%
x = get_x_vector(N, K)
env = Environment(N, K, x)
model = build_model(env.W)
train_ql(model, N, K, x, max_memory=8 * N, data_size=32)

# TODO : Make input environment state fixed shape: ((1, N ** 2)), i.e., append -1's to rest of W_hat
