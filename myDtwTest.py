# -*- coding: UTF-8 -*-
import numpy as np
import sys

# We define two sequences x, y as numpy array
# where y is actually a sub-sequence from x
x = np.array([1,2,4,6])
y = np.array([1,2,4])

from dtw import dtw
from myDtw import *


if __name__ == '__main__':


    # x = [60, 79, 101 ]
    # y = [61, 83, 105, 149]
    # x = [0, 43, 54, 64, 75, 86, 171, 214, 235, 256]
    # y = [26, 67, 77, 98, 112, 195, 238, 259, 282]
    x = [0, 37, 56, 74, 148, 158, 167, 185, 204, 222]
    #y = [37, 57, 78, 99, 119, 205, 225, 245, 264, 274]
    y = [60, 99, 117, 124, 138, 219, 233, 253, 273, 296]

    # x = [0, 22, 43, 54, 65, 76, 86, 108, 118, 129, 172, 183, 193, 204, 215, 247, 258, 268, 279, 301]
    # y = [13, 38, 62, 75, 100, 126, 145, 159, 183, 210, 235, 263, 287, 308, 318, 330, 349]
    x = [n-(x[0]-y[0]) for n in x]
    max_xy = np.max([x[-1],y[-1]]) + 30

    best_d = 10000000
    best_x = []
    best_path = []
    raw_x = x
    for i in range(-100,100,1):
        x = [n+i for n in raw_x]
        euclidean_norm = lambda x, y: np.abs(x - y)
        d, cost_matrix, acc_cost_matrix, path = dtw(x,y, dist=euclidean_norm)
        if d < best_d:
            best_d = d
            best_x = x
            best_path = path
        #print("d is {}".format(d * np.sum(acc_cost_matrix.shape)))
        #print("path is {}".format(path))

    xc, yc,p1,p2 = get_matched_frames(best_x, y)
    print("xc is {}, size {}".format(xc,len(xc)))
    print("yc is {}, size {}".format(yc,len(yc)))
    print("best_x is {}".format(best_x))
    print("best_path is {}".format(best_path))
    print("best_path0 is {}".format(best_path[0]))
    print("best_path1 is {}".format(best_path[1]))
    tmp = [1 if best_path[0][i] != best_path[1][i] else 0 for i in range(len(best_path[0]))]
    print("tmp is {}".format(tmp))
    dif_total = np.sum(tmp)
    print("dif_total is {}".format(dif_total))
    colors = ['red', 'orange', 'yellow', 'green', 'blue','black']

    plt.subplot(4,1,1)
    plt.vlines(best_x, 0, 1, color='b', linestyle='dashed')
    plt.title('x')
    plt.xlim(0,max_xy)

    # Plot x_2
    plt.subplot(4, 1, 2)
    plt.vlines(y, 0, 1, color='b', linestyle='dashed')
    plt.title('y')
    plt.xlim(0, max_xy)


    plt.subplot(4, 1, 3)
    plt.vlines(xc, 0, 1, color='b', linestyle='dashed')
    plt.title('x')
    plt.xlim(0, max_xy)

    # Plot x_2
    plt.subplot(4, 1, 4)
    plt.vlines(yc, 0, 1, color='b', linestyle='dashed')
    plt.title('y')
    plt.xlim(0, max_xy)

    plt.tight_layout()
    plt.show()