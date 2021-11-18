import numpy as np

from utils import scalar, sum_to_S, to_binary


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


def gen_W_hat(N):
    """
    Generate W_(logN x N) with binary expansions of locations
    """
    W_hat = np.empty((int(np.log2(N)), 0))
    for i in range(N):
        binary_str = str(to_binary(i, int(np.log2(N))))[::-1]
        list_binary_str = ([int(bit) for bit in binary_str])
        W_hat = np.column_stack((W_hat, list_binary_str))
    return W_hat


def get_W_hat_rows(W, W_hat):
    """
    Find rows of W_hat within the Walsh-Hadamard matrix
    """
    W_hat_bits = [list(reversed(row)) for row in W_hat]
    W_idx = [np.where(np.all(W == row, axis=1))[0][0] for row in W_hat_bits]
    return W_idx


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

    result = int((1 + ((-1) ** result)) / 2)
    return result


def get_w_hat_t(t_idx, N):
    """
    Get t-th row of W-H matrix of order _N

    :param t_idx: Index of row in W-H matrix
    :return: list
    """
    row = []
    for i in range(N):
        row.append(get_w_hat_t_helper(t_idx, i, N))

    return row


def check_x_set(all_x, A, b):
    """
    Get all x's that are valid solutions to Ax = b
    """
    valid_x = set()
    for x_vector in all_x:
        if (A @ x_vector == b).all():
            valid_x.add(tuple(x_vector))  # Make x-vectors immutable

    return list(valid_x)


def get_solutions(N, K, W_hat, x):
    """
    Get valid indices of x that sum up to S
    """
    # Scalar form of y = W_hat * x
    S = scalar(W_hat @ x)
    # print(f'Scalar value = {S}')
    solutions = []
    for partition in sum_to_S(S, K):
        if len(set(partition)) == len(partition) and max(partition) < N:
            partition = sorted(partition)
            if partition not in solutions:
                solutions.append(partition)
    x_vectors = []
    for sol in solutions:
        tmp = np.zeros(N)
        tmp[sol] = 1
        x_vectors.append(tmp)
    return x_vectors


def sample_remaining(N, K, W, W_hat, x, n_rows=1, logging=True):
    """
    Sample remaining rows of W to get the correct x, after initial logN
    measurements
    """
    # Rows of W_hat, to remove from W
    to_remove = get_W_hat_rows(W, W_hat)
    W = np.delete(W, to_remove, 0)
    valid_solutions = get_solutions(N, K, W_hat, x)
    if logging:
        print(f'Initial possible solutions: {len(valid_solutions)}')
    y = W_hat @ x
    for i in range(n_rows - 1):
        if len(W) == 0:
            if logging:
                print('No solution found.')
            break
        if logging:
            print(f'ITERATION #{int(np.log2(N) + i + 1)}')
            print('---------------')
        random_idx = np.random.choice(len(W), replace=False)
        sampled_row = W[random_idx]
        W = np.delete(W, random_idx, 0)
        W_hat = np.append(W_hat, [sampled_row], 0)
        if logging:
            print(f'# measurements: {len(W_hat)}')
        y = W_hat @ x
        valid_solutions = [
            list(map(int, x)) for x in (check_x_set(valid_solutions, W_hat, y))
        ]
        if len(valid_solutions) != 1:
            if logging:
                print(f'{len(valid_solutions)} solutions left')
                for s in valid_solutions:
                    print(s)
                print('\n')
        else:
            if logging:
                print(f'CORRECT SOLUTION: {valid_solutions[0]}')
            return W_hat


def get_true_W_hat(N, K, x, logging=True):
    W_hat = gen_W_hat(N)
    W = np.array([get_w_hat_t(i, N) for i in range(N)])
    if logging:
        print(f'x = {x}')
    return sample_remaining(N, K, W, W_hat, x, N, False)
