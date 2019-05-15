import numpy as np
import sys

# We define two sequences x, y as numpy array
# where y is actually a sub-sequence from x
x = np.array([1,2,4,6])
y = np.array([1,2,4])

from dtw import dtw


def get_dtw_min(x,y,move_range,move=True):
    euclidean_norm = lambda x, y: np.abs(x - y)

    d, cost_matrix, acc_cost_matrix, path = dtw(x, y, dist=euclidean_norm)

    #print(d)

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
def get_real_onset_by_path(onset_frames,path):
    select_onset_frames = []
    select_onset_frames.append(onset_frames[0])
    lenght = len(onset_frames) if len(onset_frames) <= len(path) else len(path)
    for i in range(1,lenght):
        if path[i] != path[i-1]:
            select_onset_frames.append(onset_frames[i])
    return select_onset_frames
def get_matched_onset_frames_by_path(x,y):
    xd = np.diff(x)
    yd = np.diff(y)

    euclidean_norm = lambda x, y: np.abs(x - y)
    d, cost_matrix, acc_cost_matrix, path = dtw(xd, yd, dist=euclidean_norm)
    print("d is {}".format(d))
    print("np.sum(acc_cost_matrix.shape) is {}".format(np.sum(acc_cost_matrix.shape)))
    print("cost_matrix is :")
    print(cost_matrix)
    print("acc_cost_matrix is :")
    print(acc_cost_matrix)
    print("path is {}".format(path))
    print("path2 is {}".format(path[0]))
    xc = get_real_onset_by_path(x, path[1])
    yc = get_real_onset_by_path(y, path[0])
    # print(xc)
    #
    # print("path2 is {}".format(path[1]))
    # print(yc)
    # print(np.diff(xc))
    # print(np.diff(yc))
    return xc,yc

def get_matched_onset_frames_by_path_v2(x,y):
    #print("x is {},size is {}".format(x,len(x)))
    #print("y is {},size is {}".format(y,len(y)))
    xd = np.diff(x)
    yd = np.diff(y)
    xc = []
    xc.append(x[0])
    yc = []
    yc.append(y[0])

    length = len(x)
    xi = 0
    yi = 0
    check_gap = 10
    x_start = xi
    x_end = xi + 1
    y_start = yi
    y_end = yi + 1
    gap = x[x_end] - x[x_start]
    y_gap = np.sum(yd[y_start:y_end])
    check_gap = np.abs(gap - y_gap)/y_gap
    cal_number = 0
    while x_end < length:
        while check_gap > 0.45:
            #print("checking ============{}========{}".format(x_start,x_end))
            #如果下一条是多窄
            if gap < np.sum(yd[x_start:y_end]):
                x_end += 1
                if x_end >= length:
                    break
                #print("x_start,x_end is {},{}".format(x_start,x_end))
                gap = x[x_end] - x[x_start]
                y_gap = np.sum(yd[y_start:y_end])
                check_gap = np.abs(gap - y_gap) / y_gap
            else: #如果下一条是多宽
                y_end += 1
                y_gap = np.sum(yd[y_start:y_end])
                check_gap = np.abs(gap - y_gap) / y_gap
            cal_number += 1
            if cal_number > 30:
                return [], []
        if x_end < len(x):
            xc.append(x[x_end])
        if y_end < len(y):
            yc.append(y[y_end])
        x_start = x_end
        x_end = x_start + 1
        if x_end >= length:
            break
        y_start = y_end
        y_end = y_start + 1
        if y_end >= len(y):
            break
        gap = x[x_end] - x[x_start]
        y_gap = np.sum(yd[y_start:y_end])
        check_gap = np.abs(gap - y_gap) / y_gap
    return xc,yc

def get_matched_onset_frames_by_path_v3(x,y):
    euclidean_norm = lambda x, y: np.abs(x - y)
    d, cost_matrix, acc_cost_matrix, path = dtw(x, y, dist=euclidean_norm)
    # print("d ,np.sum(acc_cost_matrix.shape) is {},{}".format(d, np.sum(acc_cost_matrix.shape)))
    # print("acc_cost_matrix is :")
    # print(acc_cost_matrix)
    # print("path is :")
    # print(path[0])
    # print(path[1])
    xc = []
    yc = []
    i = 0

    while i < len(path[0]):
        p1 = int(path[0][i])
        p2 = int(path[1][i])
        p1_indexs = get_indexs(path[0],p1)
        p2_indexs = get_indexs(path[1], p2)
        if len(p1_indexs)== 1 and len(p2_indexs) == 1:
            xc.append(x[p1])
            yc.append(y[p2])
            i += 1
        elif len(p1_indexs) >1:
            x_tmp = x[p1]
            y_tmp = [y[int(path[1][i])] for i in p1_indexs]
            gap = [np.abs(x_tmp - t) for t in y_tmp]
            min_index = get_indexs(gap, np.min(gap))
            xc.append(x_tmp)
            yc.append(y_tmp[min_index[0]])
            i += len(p1_indexs)
        elif len(p2_indexs) >1:
            y_tmp = y[p2]
            x_tmp = [x[int(path[0][i])] for i in p2_indexs]
            gap = [np.abs(y_tmp - t) for t in x_tmp]
            min_index = get_indexs(gap, np.min(gap))
            xc.append(x_tmp[min_index[0]])
            yc.append(y_tmp)
            i += len(p2_indexs)
    return xc,yc


# 获取相同元素出现位置的下标
def get_indexs(a,b):
    from collections import defaultdict
    result = []
    d = defaultdict(list)
    for i, v in enumerate(a):
        d[v].append(i)
    #print(d.items() )
    for (d,x) in d.items():
        #print(d,x)
        if d == b:
            result = x
    return result

# 找出多唱或漏唱的线的帧
def get_mismatch_line(standard_y,recognize_y):
    # standard_y标准线的帧列表 recognize_y识别线的帧列表
    ls = len(standard_y)
    lr = len(recognize_y)

    # 若标准线和识别线数量相同
    if ls == lr:
        return [],[]
    # 若漏唱，即标准线大于识别线数量
    elif ls > lr:
        return [ls-lr],[]
    # 多唱的情况
    elif ls!=0:
        min = 10000
        min_i = 0
        min_j = 0
        for i in standard_y:
            for j in recognize_y:
                if abs(i-j) < min:
                    min = abs(i-j)
                    min_i = i
                    min_j = j
        standard_y.remove(min_i)
        recognize_y.remove(min_j)
        get_mismatch_line(standard_y,recognize_y)
    return standard_y,recognize_y

if __name__ == '__main__':
    #x = np.array([24, 24, 26, 26, 26, 26, 23, 26, 20, 26, 24, 24, 25])
    # x = np.array([1,5,10,13])
    # y = np.array([1,4,13])
    x = [25, 30, 32, 33, 35, 33, 32, 30, 25]
    y = [31, 34, 35, 33, 31, 31, 24]
    # x = [1,1,3,3,8,1]
    # y = [2,0,0,8,7,2]
    a,b = get_mismatch_line(y.copy(),x.copy())
    print("a,b is {},{}".format(a,b))
    print(np.mean(x))
    print(np.mean(y))
    print(np.std(x))
    print(np.std(y))
    off = int(np.mean(y) - np.mean(x))
    print(off)
    #y = [x - int(np.mean(y) - np.mean(x)) for x in y]
    #y = [x - off for x in y]
    min_d,best_y,_ = get_dtw_min(x,y,65)
    print("min_d,best_y is {},{}".format(min_d,best_y))

    euclidean_norm = lambda x, y: np.abs(x - y)
    d, cost_matrix, acc_cost_matrix, path = dtw(x, y, dist=euclidean_norm)
    print("d ,np.sum(acc_cost_matrix.shape) is {},{}".format(d ,np.sum(acc_cost_matrix.shape)))
    notes_score = 60 - int(d * np.sum(acc_cost_matrix.shape))
    print("notes_score is {}".format(notes_score))
    print("acc_cost_matrix is :")
    print(acc_cost_matrix)
    print("path is :")
    print(path[0])
    print(path[1])
    xc, yc = get_matched_onset_frames_by_path_v3(x, y)
    print("xc is {}".format(xc))
    print("yc is {}".format(yc))
    d, cost_matrix, acc_cost_matrix, path = dtw(xc, yc, dist=euclidean_norm)
    print("d ,np.sum(acc_cost_matrix.shape) is {},{}".format(d, np.sum(acc_cost_matrix.shape)))

    x = np.array([1,0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1])
    y = np.array([0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2])
    # x = np.array([42, 49, 65])
    # y = np.array([0, 19, 38])
    # y = [i - (y[0] - x[0]) for i in y]
    xd = np.diff(x)
    yd = np.diff(y)
    print(xd)
    print(yd)

    xc, yc = get_matched_onset_frames_by_path_v2(x, y)
    print(xc)
    print(yc)
    print(np.diff(xc))
    print(np.diff(yc))