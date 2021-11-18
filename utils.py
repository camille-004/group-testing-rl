import itertools


def scalar(y):
    """
    Compute scalar value: Y = \sum {2^i * y_i}
    """
    return int(sum([2 ** i * y_i for i, y_i in enumerate(y)]))


def sum_to_S(n, k):
    """
    Helper function to return all ways for k numbers to sum up to S
    """
    solutions = []
    for comb in itertools.combinations(range(n + k - 1), k - 1):
        s = [comb[0]]
        for i in range(1, k - 1):
            s.append(comb[i] - comb[i - 1] - 1)
        s.append(n + k - 2 - comb[k - 2])
        solutions.append(s)
    return solutions


def to_binary(n, N):
    """
    Get binary representation of input n

    :param n: Integer of which to get binary representation
    :param N: Length of bitstring
    :return: str
    """
    return bin(n).replace('0b', '').zfill(N)
