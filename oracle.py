import numpy as np
from scipy.linalg import hadamard


def gen_walsh_hadamard(order):
    """
    Generate a Walsh-Hadamard matrix of given order

    :param order: order of W-H matrix
    :return: numpy.ndarray
    """
    mat = hadamard(order)
    mat[mat < 0] = 0
    return mat


class Oracle:
    """
    Oracle who knows x, a binary vector of length N = 2^n and has K entries of
    ones, and receives input w_hat_t in [0, 1]^N from the rows of W_hat_N,
    matrix to be designed. Outputs y, where y in {0, 1, ..., K}^m, representing
    m observations
    """

    def __init__(self, x):
        assert np.log2(len(x)).is_integer()
        self.x_ = x

    def get_y_t(self, w_hat_t):
        """
        :param w_hat_t: t-th row (in [0, 1]^N) of W_hat_N
        :return: y_t, the observation vector
        """
        return np.multiply(w_hat_t, self.x_)

# TODO Other Oracles that knows WH matrix and picks index of w_hat_t that is given in above Oracle, another that
#  determines if solution is unique
# Oracle that determines i-th row of Walsh-Hadamard matrix <- find out how to do this without the full matrix


if __name__ == '__main__':
    wh = gen_walsh_hadamard(4)
    w_t = wh[1]
    print(w_t)
    orc = Oracle([1, 1, 0, 0])
    print(orc.get_y_t(1))
