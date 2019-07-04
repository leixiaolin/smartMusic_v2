# -*- coding: UTF-8 -*-
import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import scipy.signal as signal
from dtw import dtw
from create_base import *
from rms_helper import *
import os

# 1. Get the file path to the included audio example
# Sonify detected beat events
# 定义加载语音文件并去掉两端静音的函数

test_codes = np.array(['[1000,1000;2000;1000,500,500;2000]',
                       '[2000;1000,1000;500,500,1000;2000]',
                       '[1000,1000;500,500,1000;1000,1000;2000]',
                       '[1000,--(1000);1000,--(1000);500,250,250,1000;--(1000),1000]',
                       '[500;1000,500,1000,500;500,500,500,250,250,500,500;250,250,500,500,1000]',
                       '[1000,--(1000);1000,--(1000);1000,-(500),500;1000,1000]',
                       '[750,250,500,500,500,-(500);500,1000,500,500,-(500);750,250,500,500,500,-(500)]',
                       '[500,1000,500,500,250,250;1000,500,750,250,500;3000]',
                       '[500,500,500;1000,500;500,500,500;1500;500,500,500;1000,500;500;1000;1500]',
                       '[500,500,1000;500,500;1000;375,125,250,250,375,125,250,250;500,500,1000]'])
test_note_codes = np.array(['[3,3,3,3,3,3,3,5,1,2,3]',
                            '[5,5,3,2,1,2,5,3,2]',
                            '[5,5,3,2,1,2,2,3,2,6-,5-]',
                            '[5,1+,7,1+,2+,1+,7,6,5,2,4,3,6,5]',
                            '[3,6,7,1+,2+,1+,7,6,3]',
                            '[1+,7,1+,2+,3+,2+,1+,7,6,7,1+,2+,7,1+,7,1+,2+,1+]',
                            '[5,6,1+,6,2,3,1,6-,5-]',
                            '[5,5,6,5,6,5,1,3,0,2,2,5-,2,1]',
                            '[3,2,1,2,1,1,2,3,4,5,3,6,5,5,3]',
                            '[3,4,5,1+,7,6,5]'])
test_rhythm_codes = np.array(['[500,500,1000;500,500,1000;500,500,750,250;2000]',
                              '[1000,1000;500,500,1000;1000,500,500; 2000]',
                              '[1000,1000;500,500,1000;500,250,250,500,500;2000]',
                              '[500,1000,500;250,250,250,250,500,500;500,500,500,500;2000]',
                              '[1000;500,500,1000;500,500,500,500;2000]',
                              '[500;500,500,500,500;500,500,500,500;500,500,500,500;250,250,250,250,500]',
                              '[1000,750,250,2000;500,500,500,500,2000]',
                              '[1000,1000,1000,500,500;1000,1000,1000,--(1000);1000,1000,1000;1000,4000]',
                              '[1500,500,500,500;2500,500;1000,500,500,500,500;2500,500]',
                              '[500,500;1500,500,500,500;2000]'])


def get_code(index, type):
    if type == 1:
        code = test_codes[index]
    if type == 2:
        code = test_rhythm_codes[index]
    if type == 3:
        code = test_note_codes[index]
    # code = code.replace(";", ',')
    # code = code.replace("[", '')
    # code = code.replace("]", '')
    # code = [x for x in code.split(',')]
    return code


def get_onsets_index_by_filename(filename):
    if filename.find("节奏10") >= 0 or filename.find("节奏十") >= 0 or filename.find("节奏题十") >= 0 or filename.find(
            "节奏题10") >= 0 or filename.find("节10") >= 0:
        return 9
    elif filename.find("节奏1") >= 0 or filename.find("节奏一") >= 0 or filename.find("节奏题一") >= 0 or filename.find(
            "节奏题1") >= 0 or filename.find("节1") >= 0:
        return 0
    elif filename.find("节奏2") >= 0 or filename.find("节奏二") >= 0 or filename.find("节奏题二") >= 0 or filename.find(
            "节奏题2") >= 0 or filename.find("节2") >= 0:
        return 1
    elif filename.find("节奏3") >= 0 or filename.find("节奏三") >= 0 or filename.find("节奏题三") >= 0 or filename.find(
            "节奏题3") >= 0 or filename.find("节3") >= 0:
        return 2
    elif filename.find("节奏4") >= 0 or filename.find("节奏四") >= 0 or filename.find("节奏题四") >= 0 or filename.find(
            "节奏题4") >= 0 or filename.find("节4") >= 0:
        return 3
    elif filename.find("节奏5") >= 0 or filename.find("节奏五") >= 0 or filename.find("节奏题五") >= 0 or filename.find(
            "节奏题5") >= 0 or filename.find("节5") >= 0:
        return 4
    elif filename.find("节奏6") >= 0 or filename.find("节奏六") >= 0 or filename.find("节奏题六") >= 0 or filename.find(
            "节奏题6") >= 0 or filename.find("节6") >= 0:
        return 5
    elif filename.find("节奏7") >= 0 or filename.find("节奏七") >= 0 or filename.find("节奏题七") >= 0 or filename.find(
            "节奏题7") >= 0 or filename.find("节7") >= 0:
        return 6
    elif filename.find("节奏8") >= 0 or filename.find("节奏八") >= 0 or filename.find("节奏题八") >= 0 or filename.find(
            "节奏题8") >= 0 or filename.find("节8") >= 0:
        return 7
    elif filename.find("节奏9") >= 0 or filename.find("节奏九") >= 0 or filename.find("节奏题九") >= 0 or filename.find(
            "节奏题9") >= 0 or filename.find("节9") >= 0:
        return 8
    else:
        return -1


def get_onsets_index_by_filename_rhythm(filename):
    if filename.find("旋律10") >= 0 or filename.find("旋律十") >= 0 or filename.find("视唱十") >= 0 or filename.find(
            "视唱10") >= 0 or filename.find("旋10") >= 0:
        return 9
    elif filename.find("旋律1") >= 0 or filename.find("旋律一") >= 0 or filename.find("视唱一") >= 0 or filename.find(
            "视唱1") >= 0 or filename.find("旋1") >= 0:
        return 0
    elif filename.find("旋律2") >= 0 or filename.find("旋律二") >= 0 or filename.find("视唱二") >= 0 or filename.find(
            "视唱2") >= 0 or filename.find("旋2") >= 0:
        return 1
    elif filename.find("旋律3") >= 0 or filename.find("旋律三") >= 0 or filename.find("视唱三") >= 0 or filename.find(
            "视唱3") >= 0 or filename.find("旋3") >= 0:
        return 2
    elif filename.find("旋律4") >= 0 or filename.find("旋律四") >= 0 or filename.find("视唱四") >= 0 or filename.find(
            "视唱4") >= 0 or filename.find("旋4") >= 0:
        return 3
    elif filename.find("旋律5") >= 0 or filename.find("旋律五") >= 0 or filename.find("视唱五") >= 0 or filename.find(
            "视唱5") >= 0 or filename.find("旋5") >= 0:
        return 4
    elif filename.find("旋律6") >= 0 or filename.find("旋律六") >= 0 or filename.find("视唱六") >= 0 or filename.find(
            "视唱6") >= 0 or filename.find("旋6") >= 0:
        return 5
    elif filename.find("旋律7") >= 0 or filename.find("旋律七") >= 0 or filename.find("视唱七") >= 0 or filename.find(
            "视唱7") >= 0 or filename.find("旋7") >= 0:
        return 6
    elif filename.find("旋律8") >= 0 or filename.find("旋律八") >= 0 or filename.find("视唱八") >= 0 or filename.find(
            "视唱8") >= 0 or filename.find("旋8") >= 0:
        return 7
    elif filename.find("旋律9") >= 0 or filename.find("旋律九") >= 0 or filename.find("视唱九") >= 0 or filename.find(
            "视唱9") >= 0 or filename.find("旋9") >= 0:
        return 8
    else:
        return -1


def write_txt(content, filename, mode='w'):
    """保存txt数据
    :param content:需要保存的数据,type->list
    :param filename:文件名
    :param mode:读写模式:'w' or 'a'
    :return: void
    """
    with open(filename, mode) as f:
        f.write(content)
def get_cqt_diff(filename):
    y, sr = librosa.load(filename)
    rms = librosa.feature.rmse(y=y)[0]
    rms = [x / np.std(rms) for x in rms]

    CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref=np.max)
    w, h = CQT.shape

    CQT = np.where(CQT > -22, np.max(CQT), np.min(CQT))
    min_cqt = np.min(CQT)
    max_cqt = np.max(CQT)
    result = [0,0,0,0,0,0,0,0,0,0]
    for i in range(10,h):
        col_cqt = CQT[:,i]
        before_col_cqt = CQT[:,i-1]
        diff_sum = np.sum([1 if col_cqt[i] == max_cqt and  before_col_cqt[i] == min_cqt else 0 for i in range(len(col_cqt))])
        result.append(diff_sum)


    # b, a = signal.butter(4, 0.3, analog=False)
    #
    # sig_ff = signal.filtfilt(b, a, result)
    from scipy.signal import savgol_filter
    sig_ff = savgol_filter(rms, 11, 1)  # window size 51, polynomial order 3
    sig_ff = [x / np.max(sig_ff) for x in sig_ff]


    max_indexs = [i for i in range(1,len(sig_ff)-1) if sig_ff[i]>sig_ff[i-1] and sig_ff[i]>sig_ff[i+1] and sig_ff[i] > 0.2]
    return result,sig_ff,max_indexs

def get_cqt_start_indexs(filename,filter_p1 = 7,filter_p2 = 1,row_level=30,sum_cols_threshold=1):
    #print("filename is {}".format(filename))
    # y, sr = librosa.load(filename)
    # CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref=np.max)
    # w, h = CQT.shape
    # CQT = np.where(CQT > -22, np.max(CQT), np.min(CQT))
    # min_cqt = np.min(CQT)
    # max_cqt = np.max(CQT)
    #
    # CQT = signal.medfilt(CQT, (15, 15))  # 二维中值滤波
    #
    #
    # result = []
    # last_i = 0
    # for i in range(10,h):
    #     col_cqt = CQT[30:,i]
    #     before_col_cqt_1 = CQT[30:,i-1]
    #     before_col_cqt_2 = CQT[30:, i - 2]
    #     before_col_cqt_3 = CQT[30:, i - 3]
    #     before_col_cqt_4 = CQT[30:, i - 4]
    #     before_col_cqt_5 = CQT[30:, i - 5]
    #     sum_col_cqt = np.sum([1 for i in range(len(col_cqt)) if col_cqt[i] == max_cqt])
    #     sum_before_col_cqt_1 = np.sum([1 for i in range(len(before_col_cqt_1)) if before_col_cqt_1[i] == max_cqt])
    #     sum_before_col_cqt_2 = np.sum([1 for i in range(len(before_col_cqt_2)) if before_col_cqt_2[i] == max_cqt])
    #     sum_before_col_cqt_3 = np.sum([1 for i in range(len(before_col_cqt_3)) if before_col_cqt_3[i] == max_cqt])
    #     sum_before_col_cqt_4 = np.sum([1 for i in range(len(before_col_cqt_4)) if before_col_cqt_4[i] == max_cqt])
    #     sum_before_col_cqt_5 = np.sum([1 for i in range(len(before_col_cqt_5)) if before_col_cqt_5[i] == max_cqt])
    #     sum_min = np.min([sum_before_col_cqt_1,sum_before_col_cqt_2,sum_before_col_cqt_3,sum_before_col_cqt_4,sum_before_col_cqt_5])
    #     #if sum_col_cqt >=5 + sum_min and sum_min <3 and i - last_i > 5:
    #     if sum_col_cqt >= 1 + sum_min and i - last_i > 5:
    #         result.append(i)
    #         last_i = i
    #
    # return result
    sum_cols, sig_ff = get_sum_max_for_cols(filename,filter_p1,filter_p2,row_level)
    sig_ff = [x / np.std(sig_ff) for x in sig_ff]
    starts = [i for i in range(1,len(sig_ff)-1) if sig_ff[i] > sig_ff[i-1] and sig_ff[i] >= sig_ff[i+1] and sig_ff[i] >np.mean(sig_ff)*0.5]
    if len(starts) == 0:
        return []
    selected_starts = [starts[0]]
    for i in range(1,len(starts)):
        s = selected_starts[-1]
        e = starts[i]
        if np.min(sum_cols[s:e]) <= sum_cols_threshold and sig_ff[e] - np.min(sig_ff[s:e]) > 1:
            selected_starts.append(e)
    selected_starts = [x -4 for x in selected_starts]

    return selected_starts



def get_cqt_col_diff(filename):
    y, sr = librosa.load(filename)

    CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref=np.max)
    w, h = CQT.shape
    CQT = np.where(CQT > -22, np.max(CQT), np.min(CQT))
    CQT = signal.medfilt(CQT, (15, 15))  # 二维中值滤波

    result = []
    for i in range(1,h-4):
        col_cqt = CQT[10:,i]
        before_col_cqt = CQT[10:,i-1]
        sum = np.sum([1 if before_col_cqt[i] != col_cqt[i] else 0 for i in range(len(col_cqt))])
        result.append(sum)

    return result

def get_onset_frame_length(filename):
    sum_cols, sig_ff = get_sum_max_for_cols(filename)
    #cqt_col_diff = get_cqt_col_diff(filename)
    cqt_col_diff = np.array(sum_cols)
    cqt_col_diff[-10:] = 0
    #cqt_col_diff = [x if x > 2 else 0 for x in cqt_col_diff]
    end = len(cqt_col_diff)
    starts = get_cqt_start_indexs(filename)

    if len(starts) == 0:
        return 0,0,0
    start = starts[0]
    # for i in range(2,len(cqt_col_diff)):
    #     if np.max(cqt_col_diff[:i-1]) <= 2 and cqt_col_diff[i] >2:
    #         start = i

    for i in range(len(cqt_col_diff)-6,0,-1):
        if np.max(cqt_col_diff[i+1:]) <= 1 and cqt_col_diff[i] >=1:
            end = i


    return start,end,end-start

def get_sum_max_for_cols(filename,filter_p1 = 7,filter_p2 = 1,row_level=30):
    y, sr = librosa.load(filename)
    CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref=np.max)
    w, h = CQT.shape
    CQT = np.where(CQT > -22, np.max(CQT), np.min(CQT))
    min_cqt = np.min(CQT)
    max_cqt = np.max(CQT)
    CQT = signal.medfilt(CQT, (5, 5))  #二维中值滤波

    # 滤波去噪声
    for i in range(10, h):
        col_cqt = CQT[:, i]
        sum_col = np.sum([1 if x == max_cqt else 0 for x in col_cqt])
        # if sum_col <= 5:
        #     CQT[:, i] = min_cqt
        #     continue
        for j in range(10, w - 5):
            if col_cqt[j + 1] == min_cqt and col_cqt[j - 1] == min_cqt and col_cqt[j] == max_cqt:
                CQT[j, i] = min_cqt

    result = [0,0,0,0,0,0,0,0,0,0]
    for i in range(10, h):
        col_cqt = CQT[row_level:, i]
        sum_col_cqt = np.sum([1 for i in range(len(col_cqt)) if col_cqt[i] == max_cqt])

        result.append(sum_col_cqt)

    from scipy.signal import savgol_filter
    sig_ff = savgol_filter(result, filter_p1, filter_p2)  # window size 51, polynomial order 3

    # b, a = signal.butter(4, 0.2, analog=False)
    # sig_ff = signal.filtfilt(b, a, result)
    #sig_ff = [x / np.max(sig_ff) for x in sig_ff]
    return result,sig_ff

def parse_onset_code(onset_code):
    code = onset_code
    indexs = []
    code = code.replace(";", ',')
    code = code.replace("[", '')
    code = code.replace("]", '')
    tmp = [x for x in code.split(',')]
    for i in range(len(tmp)):
        if tmp[i].find("(") >= 0:
            indexs.append(i)
    while code.find("(") >= 0:
        code = code.replace("(", '')
        code = code.replace(")", '')
        code = code.replace("-", '')
        code = code.replace("--", '')
    code = [x for x in code.split(',')]
    result = []
    for i in range(len(code)):
        if i in indexs:
            continue
        elif i+1 not in indexs:
            result.append(code[i])
        else:
            t = int(code[i]) + int(code[i+1])
            result.append(t)
    return result

def del_code_less_500(onset_code):
    code = parse_onset_code(onset_code)
    result = []
    less_end = 0
    for i in range(len(code)):
        x = int(code[i])
        if i >= less_end:
            if x < 500:
                for n in range(i,len(code)):
                    if int(code[n]) >= 500:
                        less_end = n
                        break
                tmp = np.sum([int(code[m]) for m in range(i,less_end)])
                result.append(tmp)
            else:
                result.append(x)
    return result


def add_loss_small_onset(start_indexs,onset_code):

    if len(start_indexs) == 0:
        return []
    #print("start_index is {},size is {}".format(start_indexs,len(start_indexs)))
    code = parse_onset_code(onset_code)
    code = [int(x) for x in code]

    #print("code is {},size is {}".format(code, len(code)))

    total_length_no_last = np.sum(code[:-1])
    real_total_length_no_last = start_indexs[-1] - start_indexs[0]
    rate = real_total_length_no_last/total_length_no_last
    code_dict = {}
    for x in range(125, 4000, 125):
        code_dict[x] = int(x * rate)
    # width_2000 = int(2000 * rate)
    # width_1500 = int(1500 * rate)
    # width_1000 = int(1000 * rate)
    # width_750 = int(750 * rate)
    # width_500 = int(500 * rate)
    # width_375 = int(375 * rate)
    # width_250 = int(250 * rate)
    # width_125 = int(125 * rate)
    # code_dict = {2000: width_2000, 1500: width_1500, 1000: width_1000, 750: width_750, 500: width_500, 375: width_375, 250: width_250, 125: width_125}

    less_500 = [i for i in range(len(code)) if int(code[i]) < 500 and int(code[i]) == np.min(code)]
    if len(less_500) == len(code) - len(start_indexs):
        for i in less_500:
            start_indexs.append(start_indexs[i] + code_dict.get(int(code[i])))
            start_indexs.sort()
    else:
        for i in range(len(code) -1):
            if i+1 >= len(start_indexs) -1:
                return start_indexs
            onset_width = start_indexs[i+1] - start_indexs[i]
            base_width = code_dict.get(code[i])
            next_base_width = code_dict.get(code[i+1])
            offset = np.abs((base_width - onset_width)/base_width) # 宽度误差
            offset_with_next = np.abs((base_width + next_base_width - onset_width) / (base_width + next_base_width))  # 宽度误差
            next_code = int(code[i+1])
            if next_code < 500:
                if offset_with_next < 0.35 and len(code) > len(start_indexs):
                    start_indexs.append(start_indexs[i] + int(onset_width*base_width/(base_width + next_base_width)))
                    start_indexs.sort()
                    #print("add")
            else:
                #print("ok")
                pass
    return start_indexs

def get_250_mayby_indexs(filename,onset_code):
    sum_cols, sig_ff = get_sum_max_for_cols(filename, filter_p1=31, filter_p2=12)
    starts = [i for i in range(1, len(sig_ff) - 1) if sig_ff[i] > sig_ff[i - 1] and sig_ff[i] >= sig_ff[i + 1] and sig_ff[i] > 3]
    starts_diff = np.diff(starts)
    #print("starts_diff is {},size is {}".format(starts_diff, len(starts_diff)))
    code = parse_onset_code(onset_code)
    code = [int(x) for x in code]

    #print("code is {},size is {}".format(code, len(code)))

    total_length_no_last = np.sum(code[:-1])
    if len(start_indexs) == 0:
        return
    real_total_length_no_last = start_indexs[-1] - start_indexs[0]
    rate = real_total_length_no_last / total_length_no_last
    width_375 = int(375 * rate)
    #print("width_375 is {}".format(width_375))
    width_250 = int(250 * rate)
    #print("width_250 is {}".format(width_250))
    width_125 = int(125 * rate)
    #print("width_125 is {}".format(width_125))

def get_dtw(onset_frames,base_frames):
    euclidean_norm = lambda x, y: np.abs(x - y)
    d, cost_matrix, acc_cost_matrix, path = dtw(onset_frames,base_frames, dist=euclidean_norm)
    dis = d * np.sum(acc_cost_matrix.shape)
    return dis

def get_base_frames_for_onset(filename):
    start, end, total_frames_number = get_onset_frame_length(filename)

    base_frames = onsets_base_frames(onset_code, total_frames_number)
    return base_frames

def modify_row_level(filename):
    base_frames = get_base_frames_for_onset(filename)
    base_frames_diff = np.diff(base_frames)
    best_dis = 0
    best_start_indexs = get_cqt_start_indexs(filename)
    best_row_level = 31
    for row_level in range(31,55):
        start_indexs = get_cqt_start_indexs(filename, filter_p1=31, filter_p2=12, row_level=row_level)
        start_indexs_diff = np.diff(start_indexs)
        base_frames = [x - (base_frames[0]-start_indexs[0]) for x in base_frames]
        dis = get_dtw(start_indexs_diff,base_frames_diff)
        print("row_level,dis is {},{}".format(row_level, dis))
        if dis > best_dis and len(start_indexs) == len(base_frames):
            best_dis = dis
            best_start_indexs = start_indexs
            best_row_level = row_level
    print("best_row_level,best_start_indexs is {},{}".format(best_row_level,best_start_indexs))
    return best_row_level,best_start_indexs

def check_each_onset(start_indexs,onset_code):

    if len(start_indexs) == 0:
        return []
    #print("start_index is {},size is {}".format(start_indexs,len(start_indexs)))
    code = parse_onset_code(onset_code)
    code = [int(x) for x in code]

    #print("code is {},size is {}".format(code, len(code)))

    total_length_no_last = np.sum(code[:-1])
    real_total_length_no_last = start_indexs[-1] - start_indexs[0]
    rate = real_total_length_no_last/total_length_no_last
    code_dict = {}
    for x in range(125,4000,125):
        code_dict[x] = int(x * rate)


    for i in range(len(code)):
        #print("code {} is {}".format(i,code[i]))
        if i + 1 > len(start_indexs) - 1:
            return start_indexs
        onset_width = start_indexs[i+1] - start_indexs[i]
        if i+2 <= len(start_indexs) - 1:
            next_onset_width = start_indexs[i+2] - start_indexs[i+1]
        else:
            next_onset_width = 1
        base_width = code_dict.get(code[i])
        if i+1 <= len(code) -1:
            next_base_width = code_dict.get(code[i+1])
        else:
            next_base_width = 0
        offset = np.abs((base_width - onset_width)/base_width) # 宽度误差
        offset_with_next = np.abs((base_width + next_base_width - onset_width) / (base_width + next_base_width))  # 宽度误差
        offset_with_next_real = np.abs((onset_width + next_onset_width - base_width) / base_width)  # 宽度误差

        #如果是2000，则要判断是否拖带尾声
        if int(code[i]) == 2000:
            if i == len(code)-1 and start_indexs[i + 1] is not None:  #最后一个拖带尾声
                start_indexs.remove(start_indexs[i + 1])
                #print("remove")
            elif onset_width <  base_width * 0.4 and offset_with_next_real < 0.48: #拖带尾声
                start_indexs.remove(start_indexs[i+1])
                #print("remove")

        #如果是375或250，则要判断是否漏检
        elif int(code[i]) == 375 or int(code[i]) == 250:
            if offset_with_next < 0.3: #漏检
                start_indexs.append(start_indexs[i] + int(onset_width * base_width / (base_width + next_base_width)))
                start_indexs.sort()
                #print("add")
        else:
            #print("ok")
            pass
    return start_indexs

if __name__ == "__main__":
    # y, sr = load_and_trim('F:/项目/花城音乐项目/样式数据/ALL/旋律/1.31MP3/旋律1.100分.wav')
    filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律2.1(80).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/ALL/旋律/1.31MP3/旋律3.100分.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律一（9）（100）.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律一（14）（95）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋律五（3）（63）.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏一（4）（96）.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋3王（80）.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋4谭（95）.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋4文(75).wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋8录音1(80).wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1.3(93).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋3罗（80）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律1_40312（95）.wav'
    # filename = 'e:/test_image/m1/A/旋律1_40312（95）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋3罗（80）.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋律十（2）（80）.wav'

    # filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律8录音3(95).wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1王（98）.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/Archive/dada1.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1王（98）.wav'

    # filename = 'F:/项目/花城音乐项目/样式数据/1-2/旋律mp3/旋律8.100分.mp3'
    # filename = 'F:/项目/花城音乐项目/样式数据/1-2/旋律mp3/旋律7.100分.mp3'
    # filename = 'F:/项目/花城音乐项目/样式数据/1-2/旋律mp3/旋律5.100分.mp3'
    # filename = 'F:/项目/花城音乐项目/样式数据/1-2/旋律mp3/旋律六.5（100）.mp3'
    # filename = 'F:/项目/花城音乐项目/样式数据/1-2/旋律mp3/旋律6.75分.mp3'
    # filename =  'F:/项目/花城音乐项目/样式数据/1-2/旋律mp3/旋律1.40分.mp3'

    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1.2(92).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1谭（98）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1王（98）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋3.3(96).wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋4谭（95）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋律8录音3(95).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1谭（98）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1王（98）.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋8文(58).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋律四（1）（20）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋4王（56）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋4欧(25).wav'

    # filename = 'F:/项目/花城音乐项目/样式数据/1-2/旋律/旋律八（9）(90).wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/1-2/旋律/旋律二（2）（90分）.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/1-2/旋律/旋律九（4）(95).wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/1-2/旋律/旋律三（2）（90分）.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/1-2/旋律/旋律四.1（100）.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/1-2/旋律/旋律四.3（100）.wav'
    # # filename = 'F:/项目/花城音乐项目/样式数据/1-2/旋律/旋律十（5）(50).wav'
    # # filename = 'F:/项目/花城音乐项目/样式数据/1-2/旋律/旋律七(5)（55）.wav'
    # # filename = 'F:/项目/花城音乐项目/样式数据/1-2/旋律/旋律1.90分.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/1-2/旋律/旋律四.10（100）.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/1-2/旋律/旋律三（3）（80分）.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/1-2/旋律/旋律三（8）(80).wav'
    # # filename = 'F:/项目/花城音乐项目/样式数据/1-2/旋律/旋律二（2）（90分）.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/1-2/旋律/旋律三.10（100）.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/1-2/旋律/旋律一.6（100）.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/1-2/旋律/旋律九（6）(50).wav'

    filename = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/小学8题20190624-3898-2.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏一（4）（96）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/节奏/节10.4(60).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/节奏/节1文(95).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/节奏/节奏一（1）（100）.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/节奏/节奏一录音一(82).wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/节奏/节奏一（3）（90）.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/节奏/节奏十（1）（100）.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/节奏/节1谭（96）.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/节奏/节2录音1(100).wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/节奏/节2录音3(100).wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/节奏/节4谭（95）.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/节奏/节4.1(95).wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/节奏/节4.1(95).wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/节奏/节10.4(60).wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/节奏/节10.1(97).wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/节奏/节4熙(95).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/节奏/节奏三（1）（100）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/节奏/节7.1(80).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/节奏/节8.1(78).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/节奏/节8文(5).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/节奏/节10桢(80).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/节奏/节8文(5).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/节奏/节10文(85).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/节奏/节1.2(100).wav'

    filename = 'F:/项目/花城音乐项目/样式数据/6.18MP3/节奏/1；100.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/6.18MP3/节奏/12；98.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/6.18MP3/节奏/01，100.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/6.18MP3/节奏/节奏3，90.wav'


    result_path = 'e:/test_image/n/'
    plt.close()
    type_index = get_onsets_index_by_filename(filename)
    onset_code = get_code(type_index, 1)
    rhythm_code = get_code(type_index, 2)
    pitch_code = get_code(type_index, 3)

    onset_code = '[500,500,250,250,250,250;500,250,250,1000;250,250,250,250,750,250;250,250,500,1000]'

    # filename = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/两只老虎20190624-2939.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/小学8题20190624-3898-1.wav'

    # filename,onset_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/小学8题20190624-3898-1.wav','[1000,1000;500,250,250,500;1000,500,500;2000]'  # 第1条 这个可以给满分                       90
    # filename,onset_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/小学8题20190624-3898-2.wav','[1000,500,500;2000;250,250,500,500,500;2000]'  # 第2条 基本上可以是满分                      97
    # filename,onset_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/小学8题20190624-3898-3.wav','[2000;250,250,250,250,1000;2000;500,500,1000]'  # 第3条 故意错一个，扣一分即可               89
    # filename,onset_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/小学8题20190624-3898-4.wav','[1000,250,250,250,250;2000;1000,500,500;2000]'  # 第4条 故意错了两处，应该扣两分左右即可     85
    # filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.01MP3/旋律/小学8题20190625-2251 节拍题一.wav', '[1000,1000;500,250,250,500;1000,500,500;2000]'  # 应该有七分左右                     74
    # filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.01MP3/旋律/小学8题20190625-2251 节拍题三.wav', '[2000;250,250,250,250,1000;2000;500,500,1000]'  # 应该接近满分                       97
    # filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.01MP3/旋律/中学8题20190701-4154 节拍题二.wav', '[1000,1000;1500,500;500,250,250,500,500;2000]'  # 可给满分                           100
    # filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.01MP3/旋律/中学8题20190701-4154 节拍题三.wav', '[500,1000,500;2000;500,250,250,500,500;2000]'  # 可给接近满分                        100
    # filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.01MP3/旋律/录音题E20190701-9528 第一题.wav', '[1000,1000;500,250,250,1000;500,500,500,500;2000]'  # 可给满分                         89
    # filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.01MP3/旋律/录音题E20190701-9528 第二题.wav', '[1000,500,500;500,250,250,500;500,500,1000;2000]'  # 可给接近满分                      90
    # filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.01MP3/旋律/中学8题20190701-1547 节奏一.wav', '[500,250,250,500,500;1500,500;1000,1000;2000]'  # 可给接近满分                         92
    # filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.01MP3/旋律/中学8题20190701-1547 节奏二.wav', '[1000,1000;1500,500;500,250,250,500,500;2000]'  # 可给接近满分                         78
    # filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.01MP3/旋律/中学8题20190701-1547 节奏三.wav', '[500,1000,500;2000;500,250,250,500,500;2000]'  #可给接近满分                          94
    # filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.01MP3/旋律/中学8题20190701-1547 节奏四.wav', '[500,1000,500;2000;500,500,500,250,250;2000]'  #应该给接近九分                        87
    # filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.01MP3/旋律/中学8题20190701-1547 节奏四.wav', '[500,1000,500;2000;500,500,500,250,250;2000]'  #应该给接近九分                        87

    # rhythm_code = '[1000,1000;500,500,1000;500,250,250,500,500;2000]'
    # melody_code = '[5,5,3,2,1,2,2,3,2,6-,5-]'
    print("rhythm_code is {}".format(rhythm_code))
    print("pitch_code is {}".format(pitch_code))
    # plt, total_score, onset_score, note_scroe, detail_content = draw_plt(filename, rhythm_code, pitch_code)
    # plt.show()
    # plt.clf()
    y, sr = librosa.load(filename)
    CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref=np.max)
    CQT = np.where(CQT > -22, np.max(CQT), np.min(CQT))

    plt.subplot(2,1,1)
    rms, sig_ff, max_indexs = get_cqt_diff(filename)
    times = librosa.frames_to_time(np.arange(len(rms)))
    librosa.display.specshow(CQT, x_axis='time')
    #plt.plot(times, rms)
    #plt.plot(times, sig_ff)
    plt.xlim(0, np.max(times))
    max_index_times = librosa.frames_to_time(max_indexs)
    #plt.vlines(max_index_times, 0, np.max(rms), color='r', linestyle='dashed')

    start, end, length = get_onset_frame_length(filename)
    base_frames = onsets_base_frames(onset_code, length)
    base_frames_diff =np.diff(base_frames)

    start_indexs = get_cqt_start_indexs(filename)
    start_indexs_diff = np.diff(start_indexs)

    rms, rms_diff, sig_ff, max_indexs = get_rms_max_indexs_for_onset(filename)
    max_indexs = [x for x in max_indexs if x > start-5 and x <  end]
    max_indexs_diff = np.diff(max_indexs)

    if len(start_indexs) > 1 and len(max_indexs) > 1:
        dis_with_starts = get_dtw(start_indexs_diff, base_frames_diff)
        print("dis_with_starts is {}".format(dis_with_starts))
        dis_with_starts_no_first = get_dtw(start_indexs_diff[1:], base_frames_diff)
        print("dis_with_starts_no_first is {}".format(dis_with_starts_no_first))
        dis_with_maxs = get_dtw(max_indexs_diff, base_frames_diff)
        print("dis_with_maxs is {}".format(dis_with_maxs))
        dis_with_maxs_on_first = get_dtw(max_indexs_diff[1:], base_frames_diff)
        print("dis_with_maxs_on_first is {}".format(dis_with_maxs_on_first))
        all_dis = [dis_with_starts,dis_with_starts_no_first,dis_with_maxs,dis_with_maxs_on_first]
        dis_min = np.min(all_dis)
        min_index = all_dis.index(dis_min)
        if 0 == min_index:
            start_indexs = start_indexs
        elif 1 == min_index:
            sum_cols, sig_ff = get_sum_max_for_cols(filename)
            first_range = np.sum([1 if i > start and i < start + start_indexs_diff[0] and sum_cols[i] > sum_cols[start+3]*0.2 else 0 for i in range(start,start + start_indexs_diff[0])])  #根据节拍长度判断是否为真实节拍
            if first_range > base_frames_diff[0]*0.3:
                start_indexs = start_indexs
            else:
                start_indexs = start_indexs[1:]
        elif 2 == min_index:
            start_indexs = max_indexs
        elif 3 == min_index:
            sum_cols, sig_ff = get_sum_max_for_cols(filename)
            first_range = np.sum([1 if i > start and i < start + start_indexs_diff[0] and sum_cols[i] > sum_cols[start+3]*0.2 else 0 for i in range(start,start + start_indexs_diff[0])])  #根据节拍长度判断是否为真实节拍
            if first_range > base_frames_diff[0]*0.3:
                start_indexs = max_indexs
            else:
                start_indexs = max_indexs[1:]
        # if dis_with_starts < dis_with_maxs:
        #     onsets_frames = start_indexs
        # else:
        #     onsets_frames = max_index

    else:
        start_indexs = max_indexs


    start_indexs = check_each_onset(start_indexs, onset_code)
    start_indexs = add_loss_small_onset(start_indexs, onset_code)
    print("start_indexs is {},size is {}".format(start_indexs,len(start_indexs)))
    start_indexs_time = librosa.frames_to_time(start_indexs)
    max_indexs_time = librosa.frames_to_time(max_indexs)
    plt.vlines(start_indexs_time, 0,84, color='b', linestyle='solid')
    plt.vlines(max_indexs_time, 0, 84, color='r', linestyle='dashed')


    start_time = librosa.frames_to_time(start)
    end_time = librosa.frames_to_time(end)
    plt.vlines(start_time, 0, 84, color='r', linestyle='dashed')
    plt.vlines(end_time, 0, 84, color='r', linestyle='dashed')

    plt.subplot(2,1,2)
    sum_cols, sig_ff = get_sum_max_for_cols(filename)
    #sum_cols,sig_ff = get_sum_max_for_cols(filename,filter_p1 = 31,filter_p2 = 12,row_level = 50)
    sig_ff = [x/np.std(sig_ff) for x in sig_ff]
    starts = [i for i in range(1, len(sig_ff) - 1) if sig_ff[i] > sig_ff[i - 1] and sig_ff[i] >= sig_ff[i + 1] and sig_ff[i] > 3]

    # best_row_level, best_start_indexs = modify_row_level(filename)
    # best_start_indexs_times = librosa.frames_to_time(best_start_indexs)
    # plt.vlines(best_start_indexs_times, 0, 15, color='r', linestyle='solid')

    times = librosa.frames_to_time(np.arange(len(rms)))
    sum_cols_diff = list(np.diff(sum_cols))
    sum_cols_diff.insert(0,0)
    plt.plot(times,sum_cols)
    #plt.plot(times, sum_cols_diff)
    plt.plot(times, sig_ff)
    plt.xlim(0, np.max(times))
    plt.show()
    start_indexs = add_loss_small_onset(start_indexs,onset_code)
    get_250_mayby_indexs(filename, onset_code)
    code = del_code_less_500(onset_code)
    print("code without less 500 is {}".format(code))
