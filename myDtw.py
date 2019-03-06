import numpy as np

# We define two sequences x, y as numpy array
# where y is actually a sub-sequence from x
x = np.array([1,2,4,6])
y = np.array([1,2,4])

from dtw import dtw


def get_dtw_min(x,y,move_range):
    euclidean_norm = lambda x, y: np.abs(x - y)

    d, cost_matrix, acc_cost_matrix, path = dtw(x, y, dist=euclidean_norm)

    print(d)

    ds = []
    test1 = [1,4,6,19]
    print("test1 is {}".format(test1))
    min_d = 1000000
    min_index = 0
    best_y = y
    for i in np.arange(0, move_range):
        y2 = [j + i for j in y]
        print("y2 is {}".format(y2))
        d, cost_matrix, acc_cost_matrix, path = dtw(x, y2, dist=euclidean_norm)
        ds.append(d)
        if d < min_d:
            min_index = i
            min_d = d
            best_y = y2
        print("d is {}".format(d))
    print("best index is {}".format(min_index))
    return round(min_d, 2),best_y

# You can also visualise the accumulated cost and the shortest path
import matplotlib.pyplot as plt

# plt.imshow(acc_cost_matrix.T, origin='lower', cmap='gray', interpolation='nearest')
# plt.plot(path[0], path[1], 'w')
# plt.show()

min_d,best_y = get_dtw_min(x,y,5)
print("min_d,best_y is {},{}".format(min_d,best_y))