import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

np.random.seed(0)
sns.set_style('whitegrid')

# TODO Transfer everything to Jupyter Notebook for easier testing


def to_binary(n, N):
    """
    Get binary representation of input n

    :param n: Integer of which to get binary representation
    :param N: Length of bitstring
    :return: str
    """
    return bin(n).replace('0b', '').zfill(N)


def get_w_hat_t_helper(row_idx, col_idx, N):
    """
    Helper function to get entry (t, col_idx) of W-H matrix

    :param row_idx: Row index of W-H matrix to find
    :param col_idx: Column index of W-H matrix to find
    :param N: Order of W-H matrix
    :return: int
    """
    row_bin = to_binary(row_idx, N)
    col_bin = to_binary(col_idx, N)

    row_list = [int(row_bit) for row_bit in list(row_bin[::-1])]
    col_list = [int(col_bit) for col_bit in list(col_bin[::-1])]

    result = 0
    for row_bit, col_bit in zip(row_list, col_list):
        result += row_bit * col_bit

# TODO Change back to -1 and 1
    result = int((1 + ((-1) ** result)) / 2)
    return (2 * result) - 1


class Oracle:
    """Class to generate a Walsh-Hadamard matrix based on Walsh function
    computation"""

    def __init__(self, K, N):
        # Use bitwise manipulations to check if N is a power of 2
        assert (N & (N - 1) == 0) and N != 0
        self._N = N
        self._K = K
        self.W = self.gen_walsh_hadamard()
        self._x = self.get_x_vector(self._N, self._K)

    @staticmethod
    def get_x_vector(N, K):
        """
        Return x from given order of WH matrix and K

        :param N: Order of WH matrix
        :param K: Number of ones
        :return: numpy.ndarray
        """
        x = np.zeros(N)
        random_pos = np.random.choice(
            np.arange(0, N), K, replace=False
        )
        x[random_pos] = 1
        return x

    def get_w_hat_t(self, t_idx: int) -> list:
        """
        Get t-th row of W-H matrix of order _N

        :param t_idx: Index of row in W-H matrix
        :return: list
        """
        row = []
        for i in range(self._N):
            row.append(get_w_hat_t_helper(t_idx, i, self._N))

        return row

    def get_y_t(self, t_idx, x):
        """
        Get observation vector from self._x and self.w_hat_t

        :return: numpy.ndarray
        """
        return np.multiply(self.get_w_hat_t(t_idx), x)

    def gen_walsh_hadamard(self):
        """
        Generate Walsh-Hadamard matrix of order _N

        :return: list
        """
        return [self.get_w_hat_t(i) for i in range(self._N)]

    def check_unique_sol(self, n_rows):
        """
        Attempt to arrive at a solution, x = WW^-1y, through sampling rows of
        WH without replacement

        :param n_rows: Number of rows to sample from the WH matrix to check for a
        unique solution
        :return: numpy.ndarray
        """
        # TODO Return the UNIQUE solution
        assert n_rows <= self._N

        row_pos = np.random.choice(
            np.arange(0, len(self.W)), n_rows, replace=False
        )

        print('---')
        print(self._x)
        print('---')

        W_hat = np.empty((0, self._N))
        diffs = []
        # for row_idx in range(len(row_pos)):
        for row_idx in range(self._N):
            # x is N x 1, W is row_idx x N, W_inv is N x row_idx,
            # y should be row_idx x 1
            W_hat = np.vstack([W_hat, self.W[row_idx]])
            # print(W_hat)
            W_hat_inv = np.linalg.pinv(W_hat) * self._N
            # print(np.matmul(W_hat, W_hat_inv))
            curr_y = self.get_y_t(row_idx, self._x)[:row_idx + 1]
            # print(curr_y)
            # print(W_hat_inv)
            x_tilde = np.round(W_hat_inv.dot(curr_y), 2)# * n_rows
            print(x_tilde)
            diffs.append(
                np.count_nonzero(
                    np.isclose(
                        x_tilde - self._x, np.zeros(self._N), atol=0.01)
                ) / self._N)

        plt.plot(diffs)
        plt.xlabel('Number of Rows to Generate x_tilde')
        plt.ylabel('Difference from True x Proportion')
        plt.axvline(int(self._K * np.log2(self._N) / np.log2(self._K)), c='r')
        plt.title(f'Closeness to True x: K = {self._K}, N = {self._N}')
        plt.show()


if __name__ == '__main__':
    N = 8
    K = 2
    orc = Oracle(K, N) # Number of rows needed: KlogN / logK <- Try this
    t = 2

    ideal_n_rows = int(K * np.log2(N) / np.log2(K))
    # print(f'Solution to W * x = y: {orc.check_unique_sol(int(K * np.log2(N) / np.log2(K)))}')
    # orc.check_unique_sol(N)

    print(orc._x)
    # print(orc.W)
    # print(np.linalg.pinv(orc.W))
    y = orc.get_y_t(8, orc._x)
    print(np.round(np.dot(np.linalg.pinv(orc.W), y), 2))
