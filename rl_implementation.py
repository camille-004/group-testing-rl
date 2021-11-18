import numpy as np

from ground_truth import (
    get_x_vector, gen_W_hat, get_w_hat_t, get_W_hat_rows
)


class Environment:

    def __init__(self, N, K):
        self.N = N
        self.K = K
        self.x = get_x_vector(self.N, self.K)
        self.W_hat = gen_W_hat(self.N)
        self.y = None

        W = np.array([get_w_hat_t(i, N) for i in range(N)])
        to_remove = get_W_hat_rows(W, self.W_hat)

        self.W = np.delete(W, to_remove, 0)
        self.action_space = list(range(len(self.W)))  # All indices of W rows

    def get_state(self):
        self.y = self.W_hat @ self.x
        return np.array([self.W_hat, self.y])

    def step(self, action):
        """
        Update environment based on current action

        :param action: index of current row to sample from W
        :return:
        """
        state = self.get_state()
        self.action_space.remove(action)


