# %%
import numpy as np

from ground_truth import check_x_set, gen_W_hat, get_w_hat_t, get_x_vector, get_W_hat_rows
from utils import scalar, sum_to_S

# %%
np.random.seed(1)

N = 8
K = 4


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

    def reset(self):
        self._W = np.copy(self.W)
        self.min_reward = -0.8 * (self.N - np.log2(self.N))
        self.total_reward = 0
        self.W_hat = gen_W_hat(N)
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
        print(S)
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

    def act(self, action):
        self.update_state(action)
        reward = self.reward()
        self.total_reward += reward
        status = self.status()
        env_state = self.observe()
        return env_state, reward, status

    def observe(self):
        return self.state

    def status(self):
        if self.total_reward < self.min_reward:
            return 'Did not converge'
        if self.n_solutions == 1:
            return 'Converged'
        return 'Calculating...'

    def valid_actions(self):
        pass


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


# %%
x = get_x_vector(N, K)
env = Environment(N, K, x)
env.reset()
# env.reward()
print(env.act(6))
