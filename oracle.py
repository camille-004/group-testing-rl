import numpy as np


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

    return int((1 + ((-1) ** result)) / 2)


class Oracle:
    """Class to generate a Walsh-Hadamard matrix based on Walsh function
    computation"""

    def __init__(self, x, N):
        self._x = x
        self._N = N
        self.W = self.gen_walsh_hadamard()

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

    def get_y_t(self, t_idx):
        """
        Get observation vector from self._x and self.w_hat_t

        :return: numpy.ndarray
        """
        return np.multiply(self.get_w_hat_t(t_idx), self._x)

    def gen_walsh_hadamard(self):
        """
        Generate Walsh-Hadamard matrix of order _N

        :return: list
        """
        return [self.get_w_hat_t(i) for i in range(self._N)]


if __name__ == '__main__':
    orc = Oracle([1, 1, 0, 0], 4)
    t = 3

    w_hat_t = orc.get_w_hat_t(t)  # [1, 0, 0, 1]
    print(w_hat_t)

    y_hat_t = orc.get_y_t(t)  # [1, 0, 0, 0]
    print(y_hat_t)

    wh = orc.gen_walsh_hadamard()
    '''
    [[1, 1, 1, 1], 
     [1, 0, 1, 0], 
     [1, 1, 0, 0], 
     [1, 0, 0, 1]]
    '''
    print(wh)
