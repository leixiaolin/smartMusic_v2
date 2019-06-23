# -*- coding: UTF-8 -*-
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
    x = np.diff(x)
    y = np.diff(y)
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
    x = np.diff(x)
    y = np.diff(y)
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
    return xc,yc,path[0],path[1]

def get_matched_note_lines_compared(x,y):
    flag = False
    if len(x) > len(y):
        t = x
        x = y
        y = t
        flag = True
    y_raw = y
    y = [i - (y[0] - x[0]) for i in y]

    xc1, yc1,path1,path2 = get_matched_onset_frames_by_path_v3(x, y)
    xc1 = get_raw_note_lines_from_diff(x, xc1)
    yc1 = get_raw_note_lines_from_diff(y_raw, yc1)
    if flag:
        t = xc1
        xc1 = yc1
        yc1 = t
    return xc1, yc1

def get_matched_onset_frames_compared(x,y):
    flag = False
    if len(x) > len(y):
        t = x
        x = y
        y = t
        flag = True
    y_raw = y
    y = [i - (y[0] - x[0]) for i in y]

    xc1, yc1,path1,path2 = get_matched_onset_frames_by_path_v3(x, y)
    xc1 = get_raw_data_from_diff(x, xc1)
    yc1 = get_raw_data_from_diff(y_raw, yc1)
    if flag:
        t = xc1
        xc1 = yc1
        yc1 = t
    return xc1, yc1
    # euclidean_norm = lambda x, y: np.abs(x - y)
    # d1, cost_matrix, acc_cost_matrix, path = dtw(xc1, yc1, dist=euclidean_norm)
    # #print("=======================d ,np.sum(acc_cost_matrix.shape) is {},{}".format(d1, np.sum(acc_cost_matrix.shape)))
    #
    #
    # xc2, yc2 = get_matched_onset_frames_by_path_v2(x, y)
    # xc2 = get_raw_data_from_diff(x, xc2)
    # yc2 = get_raw_data_from_diff(y, yc2)
    # if len(xc2)<1 or len(yc2)<1:
    #     return xc1, yc1
    # else:
    #     d2, cost_matrix, acc_cost_matrix, path = dtw(xc2, yc2, dist=euclidean_norm)
    #     #print("===========================d ,np.sum(acc_cost_matrix.shape) is {},{}".format(d, np.sum(acc_cost_matrix.shape)))
    #     if d1 < d2:
    #         return xc1, yc1
    #     else:
    #         return xc2, yc2
    # 获取相同元素出现位置的下标
def get_raw_data_from_diff(x,xc):
    result = []
    result.append(x[0])
    xd = np.diff(x)
    start = 0
    for i in xc:
        if i in xd:
            tmp = list(xd[start:])
            index = tmp.index(i)
            x1 = x[start + index]
            x2 = x[start + index+1]
            # if x1 not in result:
            #     result.append(x1)
            if x2 not in result:
                result.append(x2)
            start += (index + 1)
    return result

def get_raw_note_lines_from_diff(x,xc):
    result = []
    result.append(x[0])
    xd = np.diff(x)
    start = 0
    for i in xc:
        if i in xd:
            tmp = list(xd[start:])
            index = tmp.index(i)
            x1 = x[start + index]
            x2 = x[start + index+1]
            result.append(x2)
            start += (index + 1)
    return result

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
def get_match_lines(standard_y,recognize_y):
    # standard_y标准线的帧列表 recognize_y识别线的帧列表
    ls = len(standard_y)
    lr = len(recognize_y)

    standard_y_diff = np.diff(standard_y)
    recognize_y_diif = np.diff(recognize_y)

    select_standard_y = []
    select_standard_y.append(standard_y[0])
    select_recognize_y = []
    select_recognize_y.append(recognize_y[0])
    lenght = len(recognize_y) if len(recognize_y)<len(standard_y) else len(standard_y)
    recognize_i,standard_i = 1,1
    for i in range(1,lenght):
        if recognize_i > len(recognize_y)-1 or standard_i > len(standard_y)-1:
            #print('1')
            break
        if recognize_y[recognize_i] > select_recognize_y[-1]:
            real_diff = recognize_y_diif[recognize_i-1]
            standard_diff = standard_y_diff[standard_i-1]
            diff_gap = np.abs(real_diff - standard_diff)/standard_diff
            #匹配
            if diff_gap < 0.35:
                select_standard_y.append(standard_y[standard_i])
                select_recognize_y.append(recognize_y[recognize_i])
                standard_i += 1
                recognize_i += 1
            #多唱
            elif real_diff < standard_diff:
                tmp = recognize_y[recognize_i:] #之后的节拍点
                offset = [np.abs(x - recognize_y[recognize_i-1]) for x in tmp] #之后的节拍点到当前节拍的距离

                tmp_gap_indexs = [i for i in range(len(tmp)) if np.abs(offset[i] - standard_diff)/standard_diff < 0.4]
                if len(tmp_gap_indexs) > 0:
                    tmp_gaps = [np.abs(offset[i] - standard_diff)/standard_diff < 0.4 for i in range(len(tmp)) if np.abs(offset[i] - standard_diff)/standard_diff < 0.4]
                    tmp_gaps_min = np.min(tmp_gaps)
                    tmp_gaps_min_index = tmp_gaps.index(tmp_gaps_min)
                    match_index = recognize_i  + tmp_gap_indexs[tmp_gaps_min_index]
                    select_standard_y.append(standard_y[standard_i])
                    select_recognize_y.append(recognize_y[match_index])
                    standard_i += 1
                    recognize_i = match_index + 1
                else:
                    #print('2')
                    #break
                    standard_i += 1
                    recognize_i += 1
            #漏唱
            elif real_diff > standard_diff:
                tmp = standard_y[standard_i:]  # 之后的标准节拍点
                offset = [np.abs(x - standard_y[standard_i-1]) for x in tmp]  # 之后的标准节拍点到当前节拍的距离
                tmp_gap_indexs = [i for i in range(len(tmp)) if np.abs(real_diff - offset[i]) / offset[i] < 0.4]
                if len(tmp_gap_indexs) > 0:
                    tmp_gaps = [np.abs(real_diff - offset[i]) / offset[i] for i in range(len(tmp)) if np.abs(real_diff - offset[i]) / offset[i] < 0.4]
                    tmp_gaps_min = np.min(tmp_gaps)
                    tmp_gaps_min_index = tmp_gaps.index(tmp_gaps_min)
                    match_index = standard_i  + tmp_gap_indexs[tmp_gaps_min_index]
                    select_standard_y.append(standard_y[match_index])
                    select_recognize_y.append(recognize_y[recognize_i])
                    standard_i = match_index + 1
                    recognize_i += 1
                else:
                    standard_i += 1
                    recognize_i += 1
                    #print('3')
                    #break
    return select_standard_y,select_recognize_y
# 找出多唱或漏唱的线的帧
def get_match_lines_v2(standard_y,recognize_y):
    select_standard_y = []
    select_standard_y.append(standard_y[0])
    select_recognize_y = recognize_y.copy()
    recognize_i,standard_i = 0,0
    for i in range(0,len(recognize_y)-1):
        tmp_x = recognize_y[i:i+2]
        tmp_y = standard_y[standard_i:]
        if len(tmp_x) > 1 and len(tmp_y) > 1:
            min_d,best_y,x = get_dtw_min(tmp_y,tmp_x,65,move=False)
            offset = [np.abs(i - tmp_x[-1]) for i in best_y]
            offset_min = np.min(offset)
            offset_min_index = offset.index(offset_min)
            yc = standard_y[standard_i + offset_min_index]
            #xc,yc = get_matched_onset_frames_compared(tmp_x, tmp_y)
            select_recognize_y.append(yc)
            next_start = standard_y.index(yc)
            standard_i = next_start
    return select_standard_y,select_recognize_y

if __name__ == '__main__':
    #x = np.array([24, 24, 26, 26, 26, 26, 23, 26, 20, 26, 24, 24, 25])
    # x = np.array([1,5,10,13])
    # y = np.array([1,4,13])
    x = [-1 ,-3, -2, -1,  1,  2 ,-2 ,-6, -1  ,0]
    y = [ 0 ,-3 ,-2 ,-2,  2 , 0 , 2 ,-2 ,-5 ,-2]
    longest_note, base_notes = get_matched_note_lines_compared(x, y)
    print(longest_note)
    print(base_notes)
    # x = [38, 55, 73, 88]
    # y = [48, 66, 83, 100]
    y = [i -(y[0]-x[0]) for i in y]
    # x = np.diff(x)
    # y = np.diff(y)
    # x = [1,1,3,3,8,1]
    # y = [2,0,0,8,7,2]
    # a,b = get_mismatch_line(y.copy(),x.copy())
    # print("a,b is {},{}".format(a,b))
    # print(np.mean(x))
    # print(np.mean(y))
    # print(np.std(x))
    # print(np.std(y))
    # off = int(np.mean(y) - np.mean(x))
    # print(off)
    # #y = [x - int(np.mean(y) - np.mean(x)) for x in y]
    # #y = [x - off for x in y]
    # min_d,best_y,_ = get_dtw_min(x,y,65)
    # print("min_d,best_y is {},{}".format(min_d,best_y))

    #xd = np.diff(x)
    #yd = np.diff(y)
    euclidean_norm = lambda x, y: np.abs(x - y)
    # d, cost_matrix, acc_cost_matrix, path = dtw(x, y, dist=euclidean_norm)
    # print("d ,np.sum(acc_cost_matrix.shape) is {},{}".format(d ,np.sum(acc_cost_matrix.shape)))
    # notes_score = 60 - int(d * np.sum(acc_cost_matrix.shape))
    # print("notes_score is {}".format(notes_score))
    # print("acc_cost_matrix is :")
    # print(acc_cost_matrix)
    # print("path is :")
    # print(path[0])
    # print(path[1])
    xc, yc,path1,path2 = get_matched_onset_frames_by_path_v3(x, y)
    xc = get_raw_data_from_diff(x, xc)
    yc = get_raw_data_from_diff(y, yc)
    #d, cost_matrix, acc_cost_matrix, path = dtw(xc, yc, dist=euclidean_norm)
    #print("get_matched_onset_frames_by_path_v3=======================d ,np.sum(acc_cost_matrix.shape) is {},{}".format(d, np.sum(acc_cost_matrix.shape)))
    print("x  is {},size is {}".format(x, len(x)))
    print("y  is {},size is {}".format(y, len(y)))
    print("xc is {},size is {}".format(xc,len(xc)))
    print("yc is {},size is {}".format(yc,len(yc)))
    print("path is :")
    print(path1)
    print(path2)

    #x = np.array([1,0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1])
    #y = np.array([0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2])
    # x = np.array([42, 49, 65])
    # y = np.array([0, 19, 38])
    # y = [i - (y[0] - x[0]) for i in y]
    xd = np.diff(x)
    yd = np.diff(y)
    print(xd)
    print(yd)

    # xc, yc = get_matched_onset_frames_by_path_v2(x, y)
    # d, cost_matrix, acc_cost_matrix, path = dtw(xc, yc, dist=euclidean_norm)
    # print("get_matched_onset_frames_by_path_v2===========================d ,np.sum(acc_cost_matrix.shape) is {},{}".format(d, np.sum(acc_cost_matrix.shape)))
    # print(xc)
    # print(yc)
    # print(np.diff(xc))
    # print(np.diff(yc))

    xc, yc = get_matched_onset_frames_compared(x, y)
    print("get_matched_onset_frames_compared===========================compared")
    print(x)
    print(xc)
    print(y)
    print(yc)

    x = [34, 59, 82, 125, 168, 209, 226, 241, 251, 281, 295, 316, 339]
    y = [46, 69, 92, 137, 160, 182, 227, 244, 250, 261, 273, 290, 295, 307, 318, 341, 363]
    # x = [26, 62, 92, 125, 156, 178, 196, 227, 260]
    # y = [26, 96, 165, 182, 191, 200, 269]
    x = [60, 79, 101, 147, 166, 188, 231, 250, 262, 272, 288, 295, 306, 319, 339, 362]
    y = [61, 83, 105, 149, 171, 193, 236, 253, 258, 269, 280, 297, 302, 313, 324, 346, 368]
    xd = np.diff(x)
    print(xd)
    xr = [xd[i]/xd[i+1] for i in range(len(xd)-1)]
    print(xr)
    #x = [59, 82]
    yd = np.diff(y)
    print(yd)
    yr = [yd[i] / yd[i + 1] for i in range(len(yd) - 1)]
    print(yr)
    standard_y = y
    recognize_y = x
    yc,xc = get_match_lines(standard_y,recognize_y)
    #xc, yc = get_match_lines_v2(standard_y,recognize_y)
    print(x,len(x))
    print(xc,len(xc))
    print(y,len(y))
    print(yc,len(yc))


    x_max = np.max(x) if np.max(x) > np.max(y) else np.max(y)
    x_max += 5
    plt.subplot(4,1,1)
    plt.vlines(x, 0, 1, color='y', linestyle='dashed')
    plt.title('x')
    plt.xlim(0,x_max)
    plt.subplot(4, 1, 2)
    plt.vlines(y, 0, 1, color='b', linestyle='dashed')
    plt.title('y')
    plt.xlim(0, x_max)
    plt.subplot(4, 1, 3)
    plt.vlines(xc, 0, 1, color='y', linestyle='dashed')
    plt.title('xc')
    plt.xlim(0, x_max)
    plt.subplot(4, 1, 4)
    plt.vlines(yc, 0, 1, color='b', linestyle='dashed')
    plt.title('yc')
    plt.xlim(0, x_max)
    plt.show()