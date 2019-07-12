# -*- coding: UTF-8 -*-
import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import scipy.signal as signal
from dtw import dtw
from create_base import *
from rms_helper_for_note import *
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

def get_cqt_start_indexs(filename,filter_p1 = 51,filter_p2 = 3,row_level=20,sum_cols_threshold=1):
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
    selected_starts = [i for i in range(1,len(sum_cols)-1) if sum_cols[i] > sum_cols[i-1] and sum_cols[i-1] == 0]

    tmp = selected_starts[0]
    for x in range(selected_starts[0]-1,5,-1):
        if sum_cols[x-1] <= sum_cols[x] and sum_cols[x] <sum_cols[x+1] and sum_cols[x] < sum_cols[selected_starts[0]]*0.4:
            tmp = x
            break
    selected_starts[0] = tmp
    result = []
    result.append(selected_starts[0])
    for i in range(1,len(selected_starts)):
        if selected_starts[i] - result[-1] > 5:
            result.append(selected_starts[i])
    result_width = []
    for i in range(len(result)):
        start = result[i]
        if i+1 <len(result):
            end = result[i+1]
        else:
            end = len(sum_cols)
        tmp = sum_cols[start+1:end]
        tmp_max_index = tmp.index(np.max(tmp))
        tmp[:tmp_max_index] = np.ones(tmp_max_index)
        if np.min(tmp) == 0:
            width = list(tmp).index(0)
        else:
            width = end - start
        result_width.append(width)

    selected_result = []
    selected_result_width = []
    for i in range(len(result)):
        if result_width[i] >= 4:
            selected_result.append(result[i])
            selected_result_width.append(result_width[i])

    maybe_starts = [i for i in range(10, len(sum_cols) - 5) if np.abs(sum_cols[i] - sum_cols[i + 3]) > 0 and sum_cols[i + 3] > 0 and sum_cols[i] > 0 and sum_cols[i-6] > 0 and np.abs(sum_cols[i] - sum_cols[i + 1]) > 0]
    maybe_starts = [ x for x in maybe_starts if x not in result]
    maybe_result = []
    maybe_result.append(maybe_starts[0])
    for i in range(1, len(maybe_starts)):
        if maybe_starts[i] - maybe_result[-1] > 5:
            maybe_result.append(maybe_starts[i])
    return selected_result,maybe_result,selected_result_width

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
    starts,maybe_starts,starts_width = get_cqt_start_indexs(filename)

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

def get_sum_max_for_cols(filename,filter_p1 = 51,filter_p2 = 3,row_level=20):
    y, sr = librosa.load(filename)
    CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref=np.max)
    w, h = CQT.shape
    CQT = np.where(CQT > -22, np.max(CQT), np.min(CQT))
    min_cqt = np.min(CQT)
    max_cqt = np.max(CQT)
    CQT = signal.medfilt(CQT, (3, 3))  # 二维中值滤波

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
        sum_col_cqt = np.sum([n for n in range(len(col_cqt)) if col_cqt[n] == max_cqt])

        result.append(sum_col_cqt)

    # from scipy.signal import savgol_filter
    # sig_ff = savgol_filter(result, filter_p1, filter_p2)  # window size 51, polynomial order 3

    b, a = signal.butter(4, 0.2, analog=False)
    sig_ff = signal.filtfilt(b, a, result)
    sig_ff = [x / np.max(sig_ff) for x in sig_ff]
    start_index = int(len(result)*0.25)
    result[:start_index] = signal.medfilt(result[:start_index], 5)  # 一维中值滤波
    return result,sig_ff

def parse_rhythm_code(rhythm_code):
    code = rhythm_code
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

def del_code_less_500(rhythm_code):
    code = parse_rhythm_code(rhythm_code)
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


def add_loss_small_onset(start_indexs,rhythm_code):

    if len(start_indexs) == 0:
        return []
    #print("start_index is {},size is {}".format(start_indexs,len(start_indexs)))
    code = parse_rhythm_code(rhythm_code)
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

def get_250_mayby_indexs(filename,rhythm_code):
    sum_cols, sig_ff = get_sum_max_for_cols(filename, filter_p1=31, filter_p2=12)
    starts = [i for i in range(1, len(sig_ff) - 1) if sig_ff[i] > sig_ff[i - 1] and sig_ff[i] >= sig_ff[i + 1] and sig_ff[i] > 3]
    starts_diff = np.diff(starts)
    #print("starts_diff is {},size is {}".format(starts_diff, len(starts_diff)))
    code = parse_rhythm_code(rhythm_code)
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

    base_frames = onsets_base_frames(rhythm_code, total_frames_number)
    return base_frames

def modify_row_level(filename):
    base_frames = get_base_frames_for_onset(filename)
    base_frames_diff = np.diff(base_frames)
    best_dis = 0
    best_start_indexs,mayby_indexs,starts_width = get_cqt_start_indexs(filename)
    best_row_level = 31
    for row_level in range(31,55):
        start_indexs,mayby_indexs,starts_width = get_cqt_start_indexs(filename, filter_p1=31, filter_p2=12, row_level=row_level)
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

def check_each_onset(start_indexs,rhythm_code):

    if len(start_indexs) == 0:
        return []
    #print("start_index is {},size is {}".format(start_indexs,len(start_indexs)))
    code = parse_rhythm_code(rhythm_code)
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
            if onset_width > base_width:
                next_base_width = [base_width]
                tmp_sum = base_width
                for m in range(4):
                    if i+ m <= len(code)-1:
                        tmp_sum += code_dict.get(code[i+1])
                        next_base_width.append(base_width + code_dict.get(code[i+1]))

                offset = [np.abs(onset_width - x) for x in next_base_width]
                min_index = offset.index(np.min(offset))
                if min_index == 1:
                   start_indexs.append(start_indexs[i] + int(code_dict.get(code[i+1])))
                elif min_index == 2:
                   start_indexs.append(start_indexs[i] + int(code_dict.get(code[i+1])))
                   start_indexs.append(start_indexs[i] + int(code_dict.get(code[i + 1]) + code_dict.get(code[i + 2])))
                elif min_index == 3:
                   start_indexs.append(start_indexs[i] + int(code_dict.get(code[i+1])))
                   start_indexs.append(start_indexs[i] + int(code_dict.get(code[i + 1]) + code_dict.get(code[i + 2])))
                   start_indexs.append(start_indexs[i] + int(code_dict.get(code[i + 1]) + code_dict.get(code[i + 2]) + code_dict.get(code[i + 3])))
                start_indexs.sort()
                #print("ok")
            pass
    return start_indexs

def check_rms_max_by_dtw(max_indexs,base_frames,start_indexs):
    base_frames_diff = np.diff(base_frames)
    max_indexs_diff = np.diff(max_indexs)
    raw_dis = get_dtw(max_indexs_diff,base_frames_diff)
    del_maxs = []
    last_del_i = 0
    last_del_dis = 100000
    selected_max_indexs = max_indexs.copy()
    for i in range(1,len(max_indexs)):
        x = max_indexs[i]
        offset = [np.abs(x - s) for s in start_indexs]
        if np.min(offset)<=2: #即在cqt识别出的节拍中，则不作删除处理，相信cqt识别结果
            continue
        tmp = [m for m in max_indexs if m != x]
        dis = get_dtw(np.diff(tmp), base_frames_diff)
        dis_selected = get_dtw(np.diff(selected_max_indexs), base_frames_diff)
        if dis < raw_dis and dis < dis_selected:
            if i - last_del_i == 1:
                if dis < last_del_dis:
                    del_maxs.append(x)
                    if max_indexs[last_del_i] in del_maxs:
                        del_maxs.remove(max_indexs[last_del_i])
                # else:
                #     del_maxs.append(max_indexs[last_del_i])
            else:
                del_maxs.append(x)
            last_del_i = i
            last_del_dis = dis
        selected_max_indexs = [x for x in max_indexs if x not in del_maxs]
    return selected_max_indexs

def get_onset_type(onset_frames,rhythm_code):

    if len(onset_frames) == 0:
        return []
    #print("start_index is {},size is {}".format(start_indexs,len(start_indexs)))
    code = parse_rhythm_code(rhythm_code)
    code = [int(x) for x in code]

    #print("code is {},size is {}".format(code, len(code)))

    total_length_no_last = np.sum(code[:-1])
    real_total_length_no_last = onset_frames[-1] - onset_frames[0]
    rate = real_total_length_no_last/total_length_no_last
    code_dict = {}
    for x in range(125, 4000, 125):
        code_dict[x] = int(x * rate)

    width_3000 = int(3000 * rate)
    width_2000 = int(2000 * rate)
    width_1500 = int(1500 * rate)
    width_1000 = int(1000 * rate)
    width_750 = int(750 * rate)
    width_500 = int(500 * rate)
    width_375 = int(375 * rate)
    width_250 = int(250 * rate)
    width_125 = int(125 * rate)
    types = []
    for x in np.diff(onset_frames):
        if np.abs(x - width_125) < width_125/2:
            type = 125
        elif np.abs(x - width_250) < width_250/2:
            type = 250
        elif np.abs(x - width_375) < width_375/2:
            type = 375
        elif np.abs(x - width_500) < width_500/2:
            type = 500
        elif np.abs(x - width_750) < width_750/2:
            type = 750
        elif np.abs(x - width_1000) < width_1000/2:
            type = 1000
        elif np.abs(x - width_1500) < width_1500/2:
            type = 1500
        elif np.abs(x - width_2000) < width_2000/2:
            type = 2000
        elif np.abs(x - width_3000) < width_3000 / 2:
            type = 3000
        types.append(type)
    #print("types is {},size {}".format(types,len(types)))
    return types

def get_best_onset_types(start_indexs,onset_frames,rhythm_code):


    #通过与cqt起始点的距离判断可能的伪节拍
    fake_onset_frames = []
    for x in onset_frames:
        offset = [np.abs(x-s) for s in start_indexs]
        if np.min(offset) > 5:
            fake_onset_frames.append(x)

    code = parse_rhythm_code(rhythm_code)
    code = [int(x) for x in code]

    if len(onset_frames) - len(fake_onset_frames) == len(code):
        tmp = [ x for x in onset_frames if x not in fake_onset_frames]
        return tmp,fake_onset_frames

    best_dis = 1000000
    best_onset_frames = onset_frames
    if len(fake_onset_frames)>0:
        while len(best_onset_frames) > len(code):
            onset_frames_tmp = best_onset_frames.copy()
            flag = True
            for f in fake_onset_frames:
                tmp = [o for o in onset_frames_tmp if o != f]
                types = get_onset_type(tmp, rhythm_code)
                dis = get_dtw(types, code[:-1])
                if dis < best_dis:
                    best_onset_frames = tmp
                    best_dis = dis
                    flag = False
            if flag == True:
                return best_onset_frames, fake_onset_frames
    else:
        return onset_frames,fake_onset_frames
    return best_onset_frames,fake_onset_frames

def get_losses_from_rms_max(start_indexs,max_indexs,rhythm_code,end_frame):

    code = parse_rhythm_code(rhythm_code)
    code = [int(x) for x in code]
    selected_start_indexs = start_indexs.copy()

    total_length_no_last = np.sum(code[:-1])
    real_total_length_no_last = start_indexs[-1] - start_indexs[0]
    rate = real_total_length_no_last / total_length_no_last
    code_dict = {}
    for x in range(125, 4000, 125):
        code_dict[x] = int(x * rate)

    start_indexs_diff = np.diff(start_indexs)
    index_offset = 0
    current_index = 0
    for i in range(1,len(start_indexs)+1):
        start = start_indexs[i-1]
        if i < len(start_indexs):
            end = start_indexs[i]
            gap = start_indexs_diff[i-1]
        else:
            end = end_frame
            gap = end - start
        mayby_indexs = [x for x in max_indexs if x > start +5 and x < end -5]
        current_code_dict = code_dict.get(code[current_index])
        all_note_widths = [code_dict.get(c) for c in code[current_index:]]
        all_note_width_sum = [np.sum(all_note_widths[:i]) for i in range(1,len(all_note_widths)+1)]

        last_selected_onset = 0
        if gap > current_code_dict and gap > all_note_width_sum[1]:
            offset = [np.abs(gap - x) for x in all_note_width_sum]
            min_index = offset.index(np.min(offset))
            for m in range(0,min_index):
                tmp = code_dict.get(code[current_index + m])
                tmp = start + tmp
                if tmp > mayby_indexs[-1]:
                    break
                offset_tmp = [np.abs(tmp - m) for m in mayby_indexs]
                offset_tmp_min_index = offset_tmp.index(np.min(offset_tmp))
                selected_onset = mayby_indexs[offset_tmp_min_index]
                if selected_onset > last_selected_onset:
                    selected_start_indexs.append(selected_onset)
                    index_offset += 1
                    last_selected_onset = selected_onset
                    start = tmp
                else:
                    selected_onset = mayby_indexs[offset_tmp_min_index+1]
                    selected_start_indexs.append(selected_onset)
                    index_offset += 1
                    last_selected_onset = selected_onset
                    start = tmp
            current_index += index_offset
            index_offset = 0
        current_index += 1
    selected_start_indexs.sort()
    return selected_start_indexs

def get_losses_from_maybe_onset(start_indexs,start_indexs_width,mayby_indexs,rhythm_code,end_frame):

    code = parse_rhythm_code(rhythm_code)
    code = [int(x) for x in code]
    selected_start_indexs = start_indexs.copy()

    total_length_no_last = np.sum(code[:-1])
    real_total_length_no_last = start_indexs[-1] - start_indexs[0]
    rate = real_total_length_no_last / total_length_no_last
    code_dict = {}
    for x in range(125, 5000, 125):
        code_dict[x] = int(x * rate)

    index_offset = 0
    current_index = 0
    # current_code_dict = code_dict.get(code[current_index])
    # all_note_widths = [code_dict.get(c) for c in code[current_index:]]
    # all_note_width_sum = [np.sum(all_note_widths[:i]) for i in range(1, len(all_note_widths) + 1)]
    for i in range(0,len(start_indexs)):
        if current_index > len(code)-1:
            break
        current_code_dict = code_dict.get(code[current_index])
        all_note_widths = [code_dict.get(c) for c in code[current_index:]]
        if all_note_widths[-1] is None:
            break
        #print(" all_note_widths[-1],code_dict.get(2000) is {},{}".format( all_note_widths[-1],code_dict.get(2000)))
        if all_note_widths[-1] >= code_dict.get(2000):
            all_note_widths[-1] = all_note_widths[-1] *0.85
        all_note_width_sum = [np.sum(all_note_widths[:i]) for i in range(1, len(all_note_widths) + 1)]

        current_width = start_indexs_width[i]

        if current_code_dict <= code_dict.get(250):
            if current_width > current_code_dict * 1.3:
                mod = current_width // current_code_dict #求模
                for m in range(1,mod):
                    if len(selected_start_indexs) < len(code):
                        selected_onset = start_indexs[i] + current_code_dict*m
                        selected_start_indexs.append(selected_onset)
                        selected_start_indexs.sort()
                        current_index += 1
                if current_width - current_code_dict * mod > current_code_dict *0.3:
                    if len(selected_start_indexs) < len(code):
                        selected_onset = start_indexs[i] + current_code_dict * mod
                        selected_start_indexs.append(selected_onset)
                        selected_start_indexs.sort()
                        current_index += 1
        else:
            offset = [np.abs(current_width - x) for x in all_note_width_sum] # 到起点的距离，即节拍长度
            min_index =offset.index(np.min(offset))  #匹配节拍的序号
            if min_index == 0:
                if current_width > current_code_dict and i < len(start_indexs)-1 and np.abs(start_indexs[i+1] - start_indexs[i]  - all_note_width_sum[1]) < np.min(offset):
                    if len(selected_start_indexs) < len(code):
                        selected_onset = start_indexs[i] + all_note_width_sum[0]
                        selected_start_indexs.append(selected_onset)
                        selected_start_indexs.sort()
                        current_index += 1
            else:
                for m in range(0,min_index):
                    if len(selected_start_indexs) < len(code):
                        selected_onset = start_indexs[i] + all_note_width_sum[m]
                        selected_start_indexs.append(selected_onset)
                        selected_start_indexs.sort()
                        current_index += 1
        current_index += 1
    return selected_start_indexs

def get_losses_by_dtw(start_indexs,max_indexs,base_frames,end_frame):
    max_indexs = [x for x in max_indexs if x > start_indexs[0] + 4 and x < end_frame - 5 and x not in start_indexs]
    print("max_indexs is {},size {}".format(max_indexs, len(max_indexs)))
    base_frames = [x - (base_frames[0]- start_indexs[0]) for x in base_frames]
    offset_length = len(base_frames) - len(start_indexs)
    best_x = []
    step = 0
    init_tmp = start_indexs.copy()
    init_tmp.append(end_frame)
    base_frames.append(end_frame)
    print("base_frames is {},size {}".format(base_frames,len(base_frames)))
    best_dtw = 1000000000
    while step <= offset_length:
        tmp = init_tmp
        for x in max_indexs:
            tmp = init_tmp.copy()
            tmp.append(x)
            tmp.sort()
            dis = get_dtw(np.diff(tmp),np.diff(base_frames))
            #dis = get_dtw(tmp, base_frames)
            print("================x,dis,best_dtw is {},{},{}".format(x,dis,best_dtw))
            if dis < best_dtw:
                best_dtw = dis
                best_x = x
                best_tmp = tmp
        print("best_dtw is {}".format(best_dtw))
        print("best_x is {}".format(best_x))
        print("best_tmp is {},size {}".format(best_tmp,len(best_tmp)))
        init_tmp = best_tmp
        max_indexs = [x for x in max_indexs if x != best_x]
        step += 1
    return best_tmp



def get_all_notes(onset_frames,cqt,end_frame):
    #cqt = signal.medfilt(cqt, (5, 5))  # 二维中值滤波
    w,h = cqt.shape
    min_cqt = np.min(cqt)
    max_cqt = np.max(cqt)
    all_notes = []
    for i in range(len(onset_frames)):
        col_sum = np.zeros(84)
        x = onset_frames[i]
        if x == onset_frames[-1]:
            start = x
            end = end_frame
        else:
            start = x
            end = onset_frames[i+1]
        cols_cqt = cqt[:,start:end]
        for i in range(10,w-5):
            row_cqt = [1 if x == max_cqt else 0 for x in cols_cqt[i]]
            col_sum[i] = np.sum(row_cqt)
        max_indexs = [i for i in range(1,len(col_sum) -1) if col_sum[i] > col_sum[i-1] and col_sum[i] >= col_sum[i+1] and col_sum[i] > np.max(col_sum)*0.5]
        if len(max_indexs) > 0:
            if len(all_notes) ==0:
                all_notes.append(max_indexs[0])
            elif np.abs(max_indexs[0] - all_notes[-1]) < 8:
                all_notes.append(max_indexs[0])
            else:
                all_notes.append(max_indexs[0] - 12)
            #print("x is {}, note is {}".format(x,max_indexs[0]))
        else:
            #print("x is {}, note is {}".format(x,None))
            pass
    #print("all_notes is {},size {}".format(all_notes,len(all_notes)))
    return all_notes

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
    # filename = 'F:/项目/花城音乐项目/样式数据/6.18MP3/节奏/节奏3，78.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/6.18MP3/节奏/节奏3，80.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/6.18MP3/节奏/2，88（声音偏小）.wav'

    result_path = 'e:/test_image/n/'
    plt.close()
    type_index = get_onsets_index_by_filename(filename)
    # rhythm_code = get_code(type_index, 1)
    rhythm_code = get_code(type_index, 2)
    pitch_code = get_code(type_index, 3)

    # rhythm_code = '[500,500,250,250,250,250;500,250,250,1000;250,250,250,250,750,250;250,250,500,1000]'

    # filename = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/两只老虎20190624-2939.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/小学8题20190624-3898-1.wav'

    filename,rhythm_code,pitch_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/两只老虎20190624-1089.wav','[500,500,500,500;500,500,500,500;500,500,1000;500,500;1000]','[1,2,3,1,1,2,3,1,3,4,5,3,4,5]'         #音准节奏均正确，给分偏低
    # filename,rhythm_code,pitch_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/两只老虎20190624-1328.wav','[500,500,500,500;500,500,500,500;500,500,1000;500,500;1000]','[1,2,3,1,1,2,3,1,3,4,5,3,4,5]'       #音准节奏均正确，给分偏低
    # filename,rhythm_code,pitch_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/两只老虎20190624-1586.wav','[500,500,500,500;500,500,500,500;500,500,1000;500,500;1000]','[1,2,3,1,1,2,3,1,3,4,5,3,4,5]'      #音准节奏均正确，给分偏低
    # filename,rhythm_code,pitch_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/两只老虎20190624-2939.wav','[500,500,500,500;500,500,500,500;500,500,1000;500,500;1000]','[1,2,3,1,1,2,3,1,3,4,5,3,4,5]'      #音准节奏均正确，给分偏低
    # filename,rhythm_code,pitch_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/两只老虎20190624-7881.wav','[500,500,500,500;500,500,500,500;500,500,1000;500,500;1000]','[1,2,3,1,1,2,3,1,3,4,5,3,4,5]'          # 音准节奏均正确，给分偏低
    # filename,rhythm_code,pitch_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/两只老虎20190624-8973.wav','[500,500,500,500;500,500,500,500;500,500,1000;500,500;1000]','[1,2,3,1,1,2,3,1,3,4,5,3,4,5]'       #音准节奏均正确，给分偏低

    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1.2(92).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1谭（98）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1王（98）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋3.3(96).wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋4谭（95）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋律8录音3(95).wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1谭（98）.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1王（98）.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋8文(58).wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋律四（1）（20）.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋4王（56）.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋4欧(25).wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1录音3(90).wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋9.1(73).wav'

    # ============================== ok start ===============================
    # filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋4王（56）.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1谭（98）.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1王（98）.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋4欧(25).wav'
    # ============================== ok end ===============================

    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋10文(97).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋4.4(0).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋2.3(95).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋2录音1(90).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/6.18MP3/旋律/4，98.wav'

    result_path = 'e:/test_image/n/'
    type_index = get_onsets_index_by_filename_rhythm(filename)
    rhythm_code = get_code(type_index, 2)
    pitch_code = get_code(type_index, 3)

    rhythm_code, pitch_code = '[500,250,250,500,500;250,250,250,250,500,500;500,250,250,500,500;500,250,250,1000]', '[5,5,6,5,3,4,5,4,5,4,2,3,3,4,3,1,2,3,5,1]'
    # filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/两只老虎20190624-7881.wav', '[500,500,500,500;500,500,500,500;500,500,1000;500,500;1000]', '[1,2,3,1,1,2,3,1,3,4,5,3,4,5]'  # 音准节奏均正确，给分偏低 66
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
    CQT = signal.medfilt(CQT, (3, 3))  # 二维中值滤波
    librosa.display.specshow(CQT, x_axis='time')
    #plt.plot(times, rms)
    #plt.plot(times, sig_ff)
    plt.xlim(0, np.max(times))
    max_index_times = librosa.frames_to_time(max_indexs)
    #plt.vlines(max_index_times, 0, np.max(rms), color='r', linestyle='dashed')

    start, end, length = get_onset_frame_length(filename)
    base_frames = onsets_base_frames_rhythm(rhythm_code, length)
    base_frames_diff =np.diff(base_frames)

    start_indexs,maybe_start_indexs,starts_width = get_cqt_start_indexs(filename)
    print("0 start_indexs is {},size {}".format(start_indexs,len(start_indexs)))
    print("0 starts_width is {},size {}".format(starts_width, len(starts_width)))
    start_indexs = [x for x in start_indexs if x > start - 5 and x < end]
    raw_start_indexs = start_indexs.copy()
    start_indexs_diff = np.diff(start_indexs)

    rms, rms_diff, sig_ff, max_indexs = get_rms_max_indexs_for_onset(filename)
    max_indexs = [x for x in max_indexs if x > start - 5 and x < end]

    raw_start_indexs = start_indexs.copy()
    if len(start_indexs) > 2:
        dis_with_starts = get_dtw(start_indexs_diff, base_frames_diff)
        print("dis_with_starts is {}".format(dis_with_starts))
        dis_with_starts_no_first = get_dtw(start_indexs_diff[1:], base_frames_diff)
        print("dis_with_starts_no_first is {}".format(dis_with_starts_no_first))

        all_dis = [dis_with_starts,dis_with_starts_no_first]
        dis_min = np.min(all_dis)
        min_index = all_dis.index(dis_min)
        if 0 == min_index:
            start_indexs = start_indexs
        elif 1 == min_index:
            sum_cols, sig_ff = get_sum_max_for_cols(filename)
            first_range = np.sum([1 if i > start and i < start + start_indexs_diff[0] and sum_cols[i] > sum_cols[start+3]*0.2 else 0 for i in range(start,start + start_indexs_diff[0])])  #根据节拍长度判断是否为真实节拍
            if len(start_indexs) == len(base_frames) + 1:
                start_indexs = start_indexs[1:]
            elif first_range > base_frames_diff[0]*0.3:
                start_indexs = start_indexs
            else:
                start_indexs = start_indexs[1:]

        # if dis_with_starts < dis_with_maxs:
        #     onsets_frames = start_indexs
        # else:
        #     onsets_frames = max_index

    else:
        start_indexs = max_indexs

    print("3 start_indexs is {},size is {}".format(start_indexs, len(start_indexs)))
    print("3 starts_width is {},size is {}".format(starts_width, len(starts_width)))
    print("3 rhythm_code is {},size is {}".format(rhythm_code, len(rhythm_code)))
    # if len(start_indexs) != len(base_frames):
    #     start_indexs = check_each_onset(start_indexs, rhythm_code)
    #     print("4 start_indexs is {},size is {}".format(start_indexs, len(start_indexs)))
    #     start_indexs = add_loss_small_onset(start_indexs, rhythm_code)
    #     print("5 start_indexs is {},size is {}".format(start_indexs,len(start_indexs)))
    # types = get_onset_type(start_indexs, rhythm_code)
    # types[7] = types[6] + types[7]
    # types.remove(125)
    # code = parse_rhythm_code(rhythm_code)
    # code = [int(x) for x in code]
    # dis = get_dtw(types, code[:-1])
    # print("dis is {}".format(dis))
    start_indexs,fake_onset_frames = get_best_onset_types(raw_start_indexs, start_indexs, rhythm_code)
    print("6 start_indexs is {},size is {}".format(start_indexs, len(start_indexs)))
    print("raw_start_indexs is {},size is {}".format(raw_start_indexs, len(raw_start_indexs)))
    print("max_indexs is {},size is {}".format(max_indexs, len(max_indexs)))
    #selected_start_indexs = get_losses_from_rms_max(raw_start_indexs, max_indexs, rhythm_code,end)
    #selected_start_indexs = get_losses_by_dtw(raw_start_indexs,maybe_start_indexs,base_frames,end)
    selected_start_indexs = get_losses_from_maybe_onset(raw_start_indexs,starts_width,maybe_start_indexs,rhythm_code,end)
    print("selected_start_indexs is {},size is {}".format(selected_start_indexs, len(selected_start_indexs)))

    raw_start_indexs_time = librosa.frames_to_time(raw_start_indexs)
    maybe_start_indexs_time = librosa.frames_to_time(maybe_start_indexs)
    start_indexs_time = librosa.frames_to_time(start_indexs)
    max_indexs_time = librosa.frames_to_time(max_indexs)
    fake_onset_frames_time = librosa.frames_to_time(fake_onset_frames)
    selected_start_indexs_time = librosa.frames_to_time(selected_start_indexs)
    plt.vlines(raw_start_indexs_time, 0, 84, color='w', linestyle='solid')
    #plt.vlines(start_indexs_time, 0,84, color='b', linestyle='solid')
    #plt.vlines(max_indexs_time, 0, 84, color='r', linestyle='dashed')
    plt.vlines(selected_start_indexs_time, 0, 84, color='b', linestyle='dashed')
    #plt.vlines(maybe_start_indexs_time, 0, 84, color='y', linestyle='dashed')


    start_time = librosa.frames_to_time(start)
    end_time = librosa.frames_to_time(end)
    plt.vlines(start_time, 0, 40, color='r', linestyle='dashed')
    plt.vlines(end_time, 0, 40, color='r', linestyle='dashed')
    get_all_notes(selected_start_indexs, CQT, end)

    plt.subplot(2,1,2)
    sum_cols, sig_ff = get_sum_max_for_cols(filename)
    #sum_cols,sig_ff = get_sum_max_for_cols(filename,filter_p1 = 31,filter_p2 = 12,row_level = 50)
    sig_ff = [x/np.std(sig_ff) for x in sig_ff]
    starts = [i for i in range(1, len(sig_ff) - 1) if sig_ff[i] > sig_ff[i - 1] and sig_ff[i] >= sig_ff[i + 1] and sig_ff[i] > 3]

    # best_row_level, best_start_indexs = modify_row_level(filename)
    base_frames = [x - (base_frames[0]-start_indexs[0]) for x in base_frames]
    base_frames_times = librosa.frames_to_time(base_frames)
    plt.vlines(base_frames_times, 0, np.max(sum_cols)/2, color='r', linestyle='solid')

    times = librosa.frames_to_time(np.arange(len(rms)))
    sum_cols_diff = list(np.diff(sum_cols))
    sum_cols_diff.insert(0,0)
    plt.plot(times,sum_cols)
    #plt.plot(times, sum_cols_diff)
    plt.plot(times, sig_ff)
    plt.xlim(0, np.max(times))
    plt.show()
    start_indexs = add_loss_small_onset(start_indexs,rhythm_code)
    get_250_mayby_indexs(filename, rhythm_code)
    code = del_code_less_500(rhythm_code)
    print("code without less 500 is {}".format(code))
