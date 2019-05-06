import sys
import numpy as np
from myDtw import *


def cal_dtw_distance(ts_a, ts_b):
    """Returns the DTW similarity distance between two 2-D
    timeseries numpy arrays.

    Arguments
    ---------
    ts_a, ts_b : array of shape [n_samples, n_timepoints]
        Two arrays containing n_samples of timeseries data
        whose DTW distance between each sample of A and B
        will be compared

    d : DistanceMetric object (default = abs(x-y))
        the distance measure used for A_i - B_j in the
        DTW dynamic programming function

    Returns
    -------
    DTW distance between A and B
    """
    d = lambda x, y: abs(x - y)
    max_warping_window = 10000

    # Create cost matrix via broadcasting with large int
    ts_a, ts_b = np.array(ts_a), np.array(ts_b)
    M, N = len(ts_a), len(ts_b)
    cost = sys.maxsize * np.ones((M, N))

    # Initialize the first row and column
    cost[0, 0] = d(ts_a[0], ts_b[0])
    for i in range(1, M):
        cost[i, 0] = cost[i - 1, 0] + d(ts_a[i], ts_b[0])

    for j in range(1, N):
        cost[0, j] = cost[0, j - 1] + d(ts_a[0], ts_b[j])

    # Populate rest of cost matrix within window
    for i in range(1, M):
        for j in range(max(1, i - max_warping_window),
                       min(N, i + max_warping_window)):
            choices = cost[i - 1, j - 1], cost[i, j - 1], cost[i - 1, j]
            cost[i, j] = min(choices) + d(ts_a[i], ts_b[j])

    # Return DTW distance given window
    return cost[-1, -1]


if __name__ == "__main__":
    # 案例：判断ts_a与ts_b和ts_c哪个更相似

    ts_a = [1, 5, 8, 10, 56, 21, 32, 8]
    ts_b = [1, 5, 8, 10,10]
    ts_c = [1, 5, 6, 10,8]

    # 调用cal_dtw_distance计算dtw相似度
    dtw_ab = cal_dtw_distance(ts_a, ts_b)
    dtw_ac = cal_dtw_distance(ts_a, ts_c)

    print("ts_a与ts_b的dtw相似度为 %2.f，\nts_a与ts_c的dtw相似度为 %2.f。" % (dtw_ab, dtw_ac))

    if dtw_ab < dtw_ac:
        print("ts_a与ts_b 更相似！")
    else:
        print("ts_a与ts_c 更相似！")

    euclidean_norm = lambda x, y: np.abs(x - y)
    d, cost_matrix, acc_cost_matrix, path = dtw(ts_a, ts_b, dist=euclidean_norm)
    print("d ,np.sum(acc_cost_matrix.shape) is {},{}".format(d, np.sum(acc_cost_matrix.shape)))
    notes_score = 60 - int(d * np.sum(acc_cost_matrix.shape))
    print("notes_score is {}".format(notes_score))

    d, cost_matrix, acc_cost_matrix, path = dtw(ts_a, ts_c, dist=euclidean_norm)
    print("d ,np.sum(acc_cost_matrix.shape) is {},{}".format(d, np.sum(acc_cost_matrix.shape)))
    notes_score = 60 - int(d * np.sum(acc_cost_matrix.shape))
    print("notes_score is {}".format(notes_score))