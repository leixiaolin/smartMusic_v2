import numpy as np

# We define two sequences x, y as numpy array
# where y is actually a sub-sequence from x
x = np.array([1,2,4,6])
y = np.array([1,2,4])

from dtw import dtw


def get_dtw_min(x,y,move_range,move=True):
    euclidean_norm = lambda x, y: np.abs(x - y)

    d, cost_matrix, acc_cost_matrix, path = dtw(x, y, dist=euclidean_norm)

    print(d)

    ds = []
    #test1 = [1,4,6,19]
    #print("test1 is {}".format(test1))
    min_d = 1000000
    min_index = 0
    best_y = y
    if x[-1] - x[0] < y[-1] - y[0] and x[-1] < y[-1] and move: # 需要移动实唱线的情况
        x = [m + y[-1] - x[-1] for m in x]
    for i in np.arange(0, move_range):
        y2 = [j + i for j in y]
        #print("y2 is {}".format(y2))
        d, cost_matrix, acc_cost_matrix, path = dtw(x, y2, dist=euclidean_norm)
        ds.append(d)
        if d < min_d:
            min_index = i
            min_d = d
            best_y = y2
        #print("i,d is {},{}".format(i,d))
    #print("best index is {}".format(min_index))
    return round(min_d, 2),best_y,x

# You can also visualise the accumulated cost and the shortest path
import matplotlib.pyplot as plt

# plt.imshow(acc_cost_matrix.T, origin='lower', cmap='gray', interpolation='nearest')
# plt.plot(path[0], path[1], 'w')
# plt.show()
# <class 'list'>: [0, 14, 28, 42, 56, 70, 84, 98, 112, 126, 140, 154, 168, 182, 189, 196, 203, 210]

if __name__ == '__main__':
    #x = np.array([24, 24, 26, 26, 26, 26, 23, 26, 20, 26, 24, 24, 25])
    x = np.array([34, 52, 72, 113, 132, 151, 189, 208, 225, 251, 262])
    y = np.array([0, 17, 34, 68, 85, 102, 135, 152, 169, 195, 203])
    print(np.mean(x))
    print(np.mean(y))
    print(np.std(x))
    print(np.std(y))
    off = int(np.mean(y) - np.mean(x))
    print(off)
    #y = [x - int(np.mean(y) - np.mean(x)) for x in y]
    y = [x - off for x in y]
    min_d,best_y,_ = get_dtw_min(x,y,65)
    print("min_d,best_y is {},{}".format(min_d,best_y))

    euclidean_norm = lambda x, y: np.abs(x - y)
    d, cost_matrix, acc_cost_matrix, path = dtw(x, y, dist=euclidean_norm)
    print("d ,np.sum(acc_cost_matrix.shape) is {},{}".format(d ,np.sum(acc_cost_matrix.shape)))
    notes_score = 60 - int(d * np.sum(acc_cost_matrix.shape))
    print("notes_score is {}".format(notes_score))