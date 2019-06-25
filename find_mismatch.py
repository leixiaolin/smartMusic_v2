# -*- coding: UTF-8 -*-
import numpy as np
from create_base import *
from myDtw import *
from grade import *
from cqt_rms import *
from note_lines_helper_test import *

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


def get_wrong(standard_y,recognize_y):
    if len(standard_y) > 0:
        lost_num = standard_y[0]
    else:
        lost_num = 0
    ex_frames = []
    for i in recognize_y:
        ex_frames.append(i)
    return lost_num,ex_frames

'''
计算多唱扣分，漏唱扣分
'''
def get_scores(standard_y,recognize_y,onsets_total,onsets_strength):
    standard_y, recognize_y = get_mismatch_line(standard_y, recognize_y)
    lost_num, ex_frames = get_wrong(standard_y, recognize_y)
    # print(standard_y,recognize_y)
    lost_score = 0
    ex_score =0
    if lost_num:
        print('漏唱了' + str(lost_num) + '句')
        lost_score = 100 /onsets_total * lost_num
    elif len(ex_frames) > 1:
        for x in ex_frames:
            strength = onsets_strength[int(x)]
            ex_score += int(100 /onsets_total * strength)
    else:
        print('节拍数一致')
    return lost_score,ex_score

def get_deviation(standard_y,recognize_y,codes,each_onset_score,total_frames_number,loss_indexs):
    #each_onset_score = 100/len(standard_y)
    score = 0
    total = 0
    a = 0
    b = 0
    c = 0
    detail_list = []
    continue_right = []
    for i in range(len(standard_y)):
        if i < len(standard_y)-1:
            offset =np.abs((recognize_y[i+1]-recognize_y[i]) /(standard_y[i+1] - standard_y[i]) -1)
        else:
            offset = np.abs((total_frames_number - recognize_y[i]) / (total_frames_number - standard_y[i]) - 1)
        standard_offset = get_code_offset(codes[i])
        if offset <= standard_offset:
            score = 0
            if i in loss_indexs:
                detail_list.append("?")
            else:
                a += 1
                detail_list.append("1")
                continue_right.append(1)
        elif offset >= 1:
            score = each_onset_score
            if i in loss_indexs:
                detail_list.append("?")
            else:
                b += 1
                detail_list.append("0")
                continue_right.append(0)
        else:
            score = each_onset_score * offset
            if i in loss_indexs:
                detail_list.append("?")
            else:
                c += 1
                detail_list.append("-")
                continue_right.append(0)
        total +=score
    # if b == 1:
    #     total -= int(each_onset_score*0.5)
    str_detail_list = '识别的结果是：' + str(detail_list)
    str_detail_list = str_detail_list.replace("1","√")
    total_continue = continueOne(continue_right)
    if total_continue >= 4 and total > 20:
        total -= 15
        str_continue = '连续唱对的节拍数为' + str(total_continue) + '个。'
        str_detail_list = str_continue + str_detail_list

    #print(total_continue)
    detail_content = '未能匹配的节奏有'+ str(len(loss_indexs)) + '，节奏时长偏差较大的有' + str(b) + '个，偏差较小的有' + str(c) + '个，偏差在合理区间的有' + str(a) + '个，' + str_detail_list
    return total,detail_content,a

def get_deviation_for_note(standard_y,recognize_y,codes,each_onset_score):
    #each_onset_score = 100/len(standard_y)
    score = 0
    total = 0
    length = len(standard_y) if len(standard_y) < len(recognize_y) else len(recognize_y)
    a = 0
    b = 0
    c = 0
    for i in range(length-1):
        offset =np.abs((recognize_y[i+1]-recognize_y[i]) /(standard_y[i+1] - standard_y[i]) -1)
        standard_offset = get_code_offset(codes[i])
        if offset <= standard_offset:
            score = 0
            a += 1
        elif offset >= 1:
            score = each_onset_score
            b += 1
        else:
            score = each_onset_score * offset
            c += 1
        total +=score
    detail_content = '节奏时长偏差较大的有' + str(b) + '个，偏差较小的有' + str(c) + '个，偏差在合理区间的有' + str(a) + '个'
    return total,detail_content

def get_code_offset(code):
    offset = 0
    code = re.sub("\D", "", code)  # 筛选数字
    code = int(code)
    if code >= 4000:
        offset = 1/32
    elif code >= 2000:
        offset = 1/16
    elif code >= 1000:
        offset = 1/8
    elif code >= 500:
        offset = 1/4
    elif code >= 250:
        offset = 1/2
    return offset

def get_score(filename,code):



    type_index = get_onsets_index_by_filename(filename)
    y, sr = load_and_trim(filename)
    total_frames_number = get_total_frames_number(filename)

    onsets_frames, onsets_frames_strength = get_onsets_by_all(y, sr)

    # 在此处赋值防止后面实线被移动找不到强度
    recognize_y = onsets_frames

    # 标准节拍时间点
    base_frames = onsets_base_frames(code, total_frames_number)
    print("base_frames is {}".format(base_frames))

    min_d, best_y, onsets_frames = get_dtw_min(onsets_frames, base_frames, 65,move=False)
    base_onsets = librosa.frames_to_time(best_y, sr=sr)
    print("base_onsets is {}".format(base_onsets))

    # 节拍时间点
    onstm = librosa.frames_to_time(onsets_frames, sr=sr)
    print("onstm is {}".format(onstm))

    plt.vlines(onstm, -1 * np.max(y), np.max(y), color='b', linestyle='solid')
    plt.vlines(base_onsets, -1 * np.max(y), np.max(y), color='r', linestyle='dashed')

    standard_y = best_y

    codes = get_code(type_index, 1)
    min_d = get_deviation(standard_y, recognize_y, codes)
    score = get_score1(standard_y, recognize_y, len(base_frames), onsets_frames_strength, min_d)

    # # 计算成绩测试
    # print('偏移分值为：{}'.format(min_d))
    # score = get_score1(standard_y, recognize_y, len(base_frames), onsets_frames_strength, min_d)
    # print('最终得分为：{}'.format(score))
    # standard_y, recognize_y = get_mismatch_line(standard_y, recognize_y)
    # lost_num, ex_frames = get_wrong(standard_y, recognize_y)
    #
    # if lost_num:
    #     print('漏唱了' + str(lost_num) + '句')
    # elif len(ex_frames) > 1:
    #     print('多唱的帧 is {}'.format(ex_frames))
    #     ex_frames_time = librosa.frames_to_time(ex_frames[1:], sr=sr)
    #     plt.vlines(ex_frames_time, -1 * np.max(y), np.max(y), color='black', linestyle='solid')
    # else:
    #     print('节拍数一致')
    #

    '''
    调试需要查看图片则取消这部分注释
    '''
    # lost_score, ex_score = get_scores(standard_y, recognize_y, len(base_frames), onsets_frames_strength)
    # print("lost_score, ex_score is : {},{}".format(lost_score, ex_score))
    # plt.show()

    return score

'''
调试则调用此函数
'''
def debug_get_score(filename):

    type_index = get_onsets_index_by_filename(filename)
    #y, sr = load_and_trim(filename)
    y, sr = librosa.load(filename)
    total_frames_number = get_total_frames_number(filename)

    #onsets_frames, onsets_frames_strength = get_onsets_by_all(y, sr)
    #onsets_frames = get_onsets_frames_for_jz(filename)
    onsets_frames, best_threshold = get_onsets_by_cqt_rms_optimised(filename)
    # if len(onset_frames_cqt)<topN:
    onsets_frames = get_miss_onsets_by_cqt(y, onsets_frames)
    print("onsets_frames len is {}".format(len(onsets_frames)))
    onsets_frames_strength = librosa.onset.onset_strength(y=y, sr=sr)
    onsets_frames_strength = [x/np.max(onsets_frames_strength) for x in onsets_frames_strength]
    # 在此处赋值防止后面实线被移动找不到强度
    recognize_y = onsets_frames

    # 标准节拍时间点
    base_frames = onsets_base_frames(codes[type_index], total_frames_number - onsets_frames[0])
    base_frames = [x + (onsets_frames[0] - base_frames[0] - 1) for x in base_frames]
    print("base_frames is {}".format(base_frames))
    print("base_frames len is {}".format(len(base_frames)))

    min_d, best_y, onsets_frames = get_dtw_min(onsets_frames, base_frames, 65)
    base_onsets = librosa.frames_to_time(best_y, sr=sr)
    print("base_onsets is {}".format(base_onsets))

    # 节拍时间点
    onstm = librosa.frames_to_time(onsets_frames, sr=sr)
    print("onstm is {}".format(onstm))

    plt.vlines(onstm, -1 * np.max(y), np.max(y), color='b', linestyle='solid')
    plt.vlines(base_onsets, -1 * np.max(y), np.max(y), color='r', linestyle='dashed')

    standard_y = best_y.copy()

    code = get_code(type_index,1)
    modify_recognize_y = recognize_y
    each_onset_score = 100 / len(standard_y)
    ex_recognize_y = []
    #多唱的情况
    if len(standard_y) < len(recognize_y):
        _, ex_recognize_y = get_mismatch_line(standard_y.copy(), recognize_y.copy())
        modify_recognize_y = [x for x in recognize_y if x not in ex_recognize_y]
        min_d = get_deviation(standard_y,modify_recognize_y,code,each_onset_score)
    #漏唱的情况
    if len(standard_y) > len(recognize_y):
        _, lost_standard_y = get_mismatch_line(recognize_y.copy(),standard_y.copy())
        modify_standard_y = [x for x in standard_y if x not in lost_standard_y]
        min_d = get_deviation(modify_standard_y, recognize_y, code,each_onset_score)
    if len(standard_y) == len(recognize_y):
        min_d = get_deviation(standard_y, recognize_y, code, each_onset_score)
    #score = get_score1(standard_y, recognize_y, len(base_frames), onsets_frames_strength, min_d)

    # # 计算成绩测试
    print('偏移分值为：{}'.format(min_d))
    score,lost_score,ex_score,min_d = get_score1(standard_y, recognize_y, len(base_frames), onsets_frames_strength, min_d)
    print('最终得分为：{}'.format(score))

    print("lost_score, ex_score,min_d is : {},{},{}".format(lost_score, ex_score,min_d))

    # 打印多唱的节拍
    if len(ex_recognize_y) > 0:
        ex_recognize_y_time = librosa.frames_to_time(ex_recognize_y)
        plt.vlines(ex_recognize_y_time, -1 * np.max(y), np.max(y), color='black', linestyle='solid')
    #plt.text(0.2, 0.2, '偏移分值为:'+ str(round(min_d,2)))
    plt.show()

    return score

'''
计算节奏型音频的分数
'''
def get_score_jz(filename,onset_code):
    # onset_code = onset_code.replace(";", ',')
    # onset_code = onset_code.replace("[", '')
    # onset_code = onset_code.replace("]", '')
    # onset_code = [x for x in onset_code.split(',')]
    y, sr = librosa.load(filename)
    rms = librosa.feature.rmse(y=y)[0]
    rms = [x / np.std(rms) for x in rms]
    score,lost_score,ex_score,min_d,standard_y, recognize_y,onsets_frames_strength,detail_content = get_score_jz_by_cqt_rms_optimised(filename,onset_code)
    #print('最终得分为：{}'.format(score))

    # if int(score) < 90:
    #     # 标准节拍时间点
    #     if len(recognize_y) > 0:
    #         onsets_frames = recognize_y[1:]
    #         total_frames_number = get_total_frames_number(filename)
    #         base_frames = onsets_base_frames(onset_code, total_frames_number - onsets_frames[0])
    #         base_frames = [x + (onsets_frames[0] - base_frames[0]) for x in base_frames]
    #     score2, lost_score2, ex_score2, min_d2, standard_y2, recognize_y2, detail_content2 = get_score_for_onset_by_frames(recognize_y[1:], base_frames, onsets_frames_strength, onset_code,rms)
    #     if score2 > score:
    #         return int(score2), int(lost_score2), int(ex_score2), int(min_d2),standard_y2, recognize_y2,onsets_frames_strength,detail_content2

    return int(score),int(lost_score),int(ex_score),int(min_d),standard_y, recognize_y,onsets_frames_strength,detail_content


def find_loss_by_rms_for_onsets(onsets_frames,rms,onset_code):
    onset_code = onset_code.replace(";", ',')
    onset_code = onset_code.replace("[", '')
    onset_code = onset_code.replace("]", '')
    onset_code = [x for x in onset_code.split(',')]
    result = []
    keyMap = {}
    indexMap = {}
    # select_onset_frames = onsets_frames.copy()
    #select_onset_frames = onsets_frames.copy()
    select_onset_frames = []
    #select_onset_frames.append(onsets_frames[0])
    # print("all onsets_frames is {}".format(onsets_frames))
    new_added = []
    small_code_indexs = [i for i in range(len(onset_code)) if onset_code[i] == '250']
    topN = len(onset_code)
    threshold = 9
    maybe_number = 0
    threshold_length_before = 4
    threshold_length_midle = 4
    threshold_length_after = 4
    if len(small_code_indexs) < 1:
        threshold_length = 9
    else:
        before_half = [i for i in range(len(small_code_indexs)) if small_code_indexs[i] <= int(len(onset_code) / 3)]
        middle_half = [i for i in range(len(small_code_indexs)) if small_code_indexs[i] > int(len(onset_code) / 3) and small_code_indexs[i] <= int( len(onset_code) * 2 / 3)]
        after_half = [i for i in range(len(small_code_indexs)) if small_code_indexs[i] > int(len(onset_code) * 2 / 3)]
        if len(before_half) < 1:
            threshold_length_before = 10
        if len(middle_half) < 1:
            threshold_length_midle = 10
        if len(after_half) < 1:
            threshold_length_after = 10
    for i in range(1,len(rms)-20):
        if (i==1 and rms[2] > rms [1]) or (rms[i+1] > rms [i] and rms[i] == rms[i-1]) or (rms[i+1] > rms[i] and rms[i-1] > rms[i]):
            hightest_point_after = find_hightest_after(i, rms)
            if i == onsets_frames[0]:
                rms_theshold = 0.5
            else:
                rms_theshold = 0.15

            if rms[hightest_point_after] - rms[i] > rms_theshold:
                #print("rms[hightest_point_after] - rms[i],i is {}=={}".format(rms[hightest_point_after] - rms[i],i))
                value = rms[hightest_point_after] - rms[i]
                result.append(value)  #保存振幅增值
                keyMap[value] = i
                indexMap[i] = value
                if rms[hightest_point_after] - rms[i] > 1.0:
                    maybe_number += 1
    if maybe_number > topN:
        topN = maybe_number
    topN_index = find_n_largest(result,topN)
    topN_key = [result[i] for i in topN_index] #topN的振幅增值
    for x in onsets_frames:
        hightest_point_after = find_hightest_after(x, rms)
        value = rms[hightest_point_after] - rms[i]
        if value > np.min(topN_key):
            select_onset_frames.append(x)

    for key in topN_key:
        index = keyMap.get(key)
        if index <= int(len(rms) / 3):
            threshold_length = threshold_length_before
        elif index > int(len(rms) / 3) and index <= int(len(rms) * 2 / 3):
            threshold_length = threshold_length_midle
        else:
            threshold_length = threshold_length_after
        if len(select_onset_frames) == 0:
            offset_min = threshold_length + 1
        else:
            offset = [np.abs(index - x) for x in select_onset_frames]
            offset_min = np.min(offset)
        if offset_min > threshold_length:
            select_onset_frames.append(index)
            new_added.append(index)
    select_onset_frames.sort()
    return select_onset_frames

def find_loss_by_rms_for_onsets_in_range(onsets_frames,rms,start,end,loss_number,onset_code):

    result = []
    keyMap = {}
    indexMap = {}
    loss_onset_frames = []
    loss_rms_values = []
    onset_code = onset_code.replace(";", ',')
    onset_code = onset_code.replace("[", '')
    onset_code = onset_code.replace("]", '')
    onset_code = [x for x in onset_code.split(',')]

    new_added = []
    small_code_indexs = [i for i in range(len(onset_code)) if onset_code[i] == '250']
    topN = len(onset_code)
    threshold = 9
    if len(small_code_indexs) > 0:
        threshold = 3
    for i in range(5,len(rms)-20):
        if (i==1 and rms[2] > rms [1]) or (rms[i+1] > rms [i] and rms[i] == rms[i-1]) or (rms[i+1] > rms[i] and rms[i-1] > rms[i]):
            hightest_point_after = find_hightest_after(i, rms)

            rms_theshold = 0.8

            if rms[hightest_point_after] - rms[i] > rms_theshold:
                #print("rms[hightest_point_after] - rms[i],i is {}=={}".format(rms[hightest_point_after] - rms[i],i))
                value = rms[hightest_point_after] - rms[i]
                if i>start and i<end-10:
                    result.append(value)  #保存振幅增值
                    keyMap[value] = i
                    indexMap[i] = value
    loss_index = find_n_largest(result,loss_number)
    topN_key = [result[i] for i in loss_index] #topN的振幅增值


    for key in topN_key:
        index = keyMap.get(key)
        value = indexMap.get(index)
        if len(loss_onset_frames) == 0:
            offset_min = threshold + 1
        else:
            offset = [np.abs(index - x) for x in loss_onset_frames]
            offset_min = np.min(offset)
        if offset_min > threshold:
            loss_onset_frames.append(index)
            new_added.append(index)
            loss_rms_values.append(value)
    loss_onset_frames.sort()
    return loss_onset_frames,loss_rms_values

def del_same_onsets_by(onsets_frames,CQT,rms,base_frames):
    select_onsets_frames = []
    if len(onsets_frames) > 0:
        select_onsets_frames.append(onsets_frames[0])
        cqt_max = np.max(CQT)
        base_frames_min = np.min(np.diff(base_frames))
        for i in range(1, len(onsets_frames)):
            hightest_point_after = find_hightest_after(onsets_frames[i], rms)
            if onsets_frames[i] - onsets_frames[i - 1] > base_frames_min * 0.6:
                select_onsets_frames.append(onsets_frames[i])
            elif rms[hightest_point_after] - rms[onsets_frames[i]] > 1.2:
                #print("========= {}".format(rms[hightest_point_after] - rms[onsets_frames[i]]))
                select_onsets_frames.append(onsets_frames[i])
    return select_onsets_frames


def get_highest_point(CQT,start,end):
    sub_cqt = CQT[:, start:end]
    w, h = sub_cqt.shape
    cqt_max = np.max(sub_cqt)
    first_longest_number = 0
    for i in range(h):
        col_cqt = sub_cqt[:, i]
        if np.max(col_cqt) == cqt_max:
            col_list = list(col_cqt)
            col_list.reverse()
            highest_index = w - col_list.index(cqt_max)
            if highest_index > first_longest_number:
                first_longest_number = highest_index
    return first_longest_number

def get_same_number_in_two_cqt(cqt1,cqt2):
    cqt_max = np.max(cqt1)
    w1,h1 = cqt1.shape
    w2,h2 = cqt2.shape
    # if h1>5 and h2>5:
    #     h = 5
    # else:
    #     h = np.min(h1,h2)
    # c1 = cqt1[:,0:h]
    # c2 = cqt2[:,0:h]
    c1 = cqt1
    c2 = cqt2
    h = h1 if h1 < h2 else h2
    sum_max = 0
    sum_max1 = 0
    sum_max2 = 0
    for i in range(w1):
        for j in range(h):
            if c1[i,j] == c2[i,j] and c1[i,j] == cqt_max:
                sum_max += 1
            if c1[i,j] == cqt_max:
                sum_max1 += 1

            if c2[i,j] == cqt_max:
                sum_max2 += 1
    return sum_max,sum_max1,sum_max2

def check_middle_for_col_cqt(col_cqt,cqt,onset_frame):
    cqt_min = np.min(cqt)
    cqt_max = np.max(cqt)
    start_points = [i for i in range(len(col_cqt)-1) if col_cqt[i]== cqt_min and col_cqt[i+1] == cqt_max]
    end_points = [i for i in range(len(col_cqt)-1) if col_cqt[i]== cqt_max and col_cqt[i+1] == cqt_min]
    flag = True
    check_sum = 0
    if len(start_points) > 1 and len(end_points) > 1:
        for i in range(len(start_points)):
            sp = start_points[i]
            if sp < 81:
                if np.min(cqt[sp,onset_frame-2:onset_frame+1]) == cqt_max \
                        or np.min(cqt[sp+1,onset_frame-2:onset_frame+1]) == cqt_max \
                        or np.min(cqt[sp+2,onset_frame-2:onset_frame+1]) == cqt_max \
                        or np.min(cqt[sp+3,onset_frame-2:onset_frame+1]) == cqt_max:
                    check_sum += 1
            else:
                if np.min(cqt[sp,onset_frame-2:onset_frame+1]) == cqt_max \
                        or np.min(cqt[sp+1,onset_frame-2:onset_frame+1]) == cqt_max \
                        or np.min(cqt[sp+2,onset_frame-2:onset_frame+1]) == cqt_max :
                    check_sum += 1
            if check_sum == 3:
                break
    if check_sum > 2 :
         flag = False
    return flag,check_sum

def del_middle_false_onset_frames(cqt,onset_frames):
    result = []
    result2 = []
    for x in onset_frames:
        onset_frame = x
        col_cqt = cqt[:,onset_frame]
        flag,check_sum = check_middle_for_col_cqt(col_cqt,cqt,onset_frame)
        if flag:
            result.append(onset_frame)
            if check_sum < 1:
                result2.append(onset_frame)
    return result,result2
'''
计算节奏型音频的分数
'''
def get_score_jz_by_cqt_rms_optimised(filename,onset_code):

    #type_index = get_onsets_index_by_filename(filename)
    #y, sr = load_and_trim(filename)
    y, sr = librosa.load(filename)
    total_frames_number = get_total_frames_number(filename)

    #onsets_frames, onsets_frames_strength = get_onsets_by_all(y, sr)
    #onsets_frames = get_onsets_frames_for_jz(filename)
    onsets_frames, best_threshold = get_onsets_by_cqt_rms_optimised(filename,onset_code)
    # if len(onset_frames_cqt)<topN:
    onsets_frames = get_miss_onsets_by_cqt(y, onsets_frames)
    #print("onsets_frames len is {}".format(len(onsets_frames)))
    onsets_frames_strength = librosa.onset.onset_strength(y=y, sr=sr)
    onsets_frames_strength = [x/np.max(onsets_frames_strength) for x in onsets_frames_strength]
    # 在此处赋值防止后面实线被移动找不到强度
    onsets_frames = list(set(onsets_frames))
    onsets_frames.sort()
    rms = librosa.feature.rmse(y=y)[0]
    rms = [x / np.std(rms) for x in rms]
    onsets_frames = find_loss_by_rms_for_onsets(onsets_frames, rms, onset_code)
    CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref=np.max)
    CQT = np.where(CQT > -22, np.max(CQT), np.min(CQT))
    onsets_frames, note_lines, times = get_note_lines(CQT, onsets_frames)
    onsets_frames = [onsets_frames[i] for i in range(len(times)) if times[i] > 3]
    note_lines = [note_lines[i] for i in range(len(times)) if times[i] > 3]
    times = [times[i] for i in range(len(times)) if times[i] > 3]

    #判断每个节拍是不是中间的伪节拍,onsets_frames可能包含中间伪节拍，onsets_frames2一定不包含
    onsets_frames,onsets_frames2 = del_middle_false_onset_frames(CQT, onsets_frames)

    cqt1 = CQT[10:, onsets_frames[0]:onsets_frames[1]]
    cqt2 = CQT[10:, onsets_frames[1]:onsets_frames[2]]
    cqt3 = CQT[10:, onsets_frames[2]:onsets_frames[3]]
    same_sum1_2, sum_max1, sum_max2 = get_same_number_in_two_cqt(cqt1, cqt2)
    same_sum1_3, sum_max2_, sum_max3 = get_same_number_in_two_cqt(cqt1, cqt3)
    #print("same_sum, sum_max1, sum_max2 is {},{},{}".format(same_sum, sum_max1, sum_max2))
    deleted_frame = 0
    same_min1 = 40 if 40 < sum_max2/4 else sum_max2/4
    same_min2 = 40 if 40 < sum_max3/4 else sum_max3/4
    if same_sum1_2 < same_min1 and same_sum1_3 < same_min2:  # 处理这一个节拍为噪声的情况
        deleted_frame = onsets_frames[0]
        onsets_frames.remove(deleted_frame)
        if deleted_frame in onsets_frames2:
            onsets_frames2.remove(deleted_frame)


    score, lost_score, ex_score, min_d, standard_y, recognize_y, detail_content = get_score_for_onset_by_frames(onsets_frames, total_frames_number, onsets_frames_strength, onset_code,rms,CQT)
    if deleted_frame > 0 and same_sum1_2 > sum_max2/10 and same_sum1_3 > sum_max3/10:
        onsets_frames.append(deleted_frame)
        onsets_frames.sort()
        score2, lost_score2, ex_score2, min_d2, standard_y2, recognize_y2, detail_content2 = get_score_for_onset_by_frames( onsets_frames, total_frames_number, onsets_frames_strength, onset_code, rms, CQT)
        if score >= score2:
            return int(score),int(lost_score),int(ex_score),int(min_d),standard_y, recognize_y,onsets_frames_strength,detail_content
        else:
            return int(score2), int(lost_score2), int(ex_score2), int(min_d2), standard_y2, recognize_y2, onsets_frames_strength, detail_content2
    else:
        if len(onsets_frames2)>0:
            score3, lost_score3, ex_score3, min_d3, standard_y3, recognize_y3, detail_content3 = get_score_for_onset_by_frames(
                onsets_frames2, total_frames_number, onsets_frames_strength, onset_code, rms, CQT)
            if score >= score3:
                return int(score), int(lost_score), int(ex_score), int(
                    min_d), standard_y, recognize_y, onsets_frames_strength, detail_content
            else:
                return int(score3), int(lost_score3), int(ex_score3), int(
                    min_d3), standard_y3, recognize_y3, onsets_frames_strength, detail_content3
            return int(score), int(lost_score), int(ex_score), int(min_d), standard_y, recognize_y, onsets_frames_strength, detail_content
        else:
            return int(score), int(lost_score), int(ex_score), int(
                min_d), standard_y, recognize_y, onsets_frames_strength, detail_content
def find_loss_by_compare_with_base(onsets_frames,base_frames,rms,onset_code):
    result = []
    gap = 0
    length = len(onsets_frames) if len(onsets_frames) < len(base_frames) else len(base_frames)
    for i in range(length-1):
        offset =np.abs((onsets_frames[i+1]-onsets_frames[i]) /(base_frames[i+1] - base_frames[i]) -1)
        if offset > 0.3:
            start, end = onsets_frames[i],onsets_frames[i+1]
            loss_onset_frames, loss_rms_values = find_loss_by_rms_for_onsets_in_range(onsets_frames, rms, start, end,1, onset_code)
            if len(loss_onset_frames) > 0:
                result.append(loss_onset_frames[0])
                break
                gap += 1
    result.sort()
    return result

def get_score_for_onset_by_frames(onsets_frames,total_frames_number,onsets_frames_strength,onset_code,rms,CQT):
    # 标准节拍时间点
    if len(onsets_frames) > 0:
        base_frames = onsets_base_frames(onset_code, total_frames_number - onsets_frames[0])
        base_frames = [x + (onsets_frames[0] - base_frames[0]) for x in base_frames]
        # min_d, best_y, _ = get_dtw_min(onsets_frames.copy(), base_frames, 65)
    else:
        base_frames = onsets_base_frames(onset_code, total_frames_number)

    onsets_frames = del_same_onsets_by(onsets_frames, CQT,rms, base_frames)

    recognize_y = onsets_frames
    min_d, best_y, onsets_frames = get_dtw_min(onsets_frames, base_frames, 65)

    standard_y = best_y.copy()

    code = onset_code
    index = 0
    code = code.replace(";", ',')
    code = code.replace("[", '')
    code = code.replace("]", '')
    if code.find("(") >= 0:
        tmp = [x for x in code.split(',')]
        for i in range(len(tmp)):
            if tmp[i].find("(") >= 0:
                index = i
                break
        code = code.replace("(", '')
        code = code.replace(")", '')
        code = code.replace("-", '')
        code = code.replace("--", '')
    code = [x for x in code.split(',')]
    # code = [int(x) for x in code]
    if index > 0:
        code[index - 1] += code[index]
        del code[index]
    each_onset_score = 100 / len(standard_y)
    try:
        #xc, yc = get_matched_onset_frames_compared(standard_y, recognize_y)
        if len(standard_y) != len(recognize_y):
            xc,yc = get_match_lines(standard_y,recognize_y)
        else:
            xc, yc = standard_y, recognize_y
    except AssertionError as e:
        lenght = len(standard_y) if len(standard_y) <= len(recognize_y) else len(recognize_y)
        xc, yc = standard_y[:lenght], recognize_y[:lenght]
    std_number = len(standard_y) - len(xc) + len(recognize_y) - len(yc)
    # 未匹配节拍的序号
    loss_indexs = [i for i in range(len(standard_y)) if standard_y[i] not in xc]
    #多出节拍的序号
    ex_indexs = [i for i in range(len(recognize_y)) if recognize_y[i] not in yc]

    if len(loss_indexs) > 0:
        for i in loss_indexs:
            xc.append(standard_y[i])  # 补齐便为比较
            yc.append(yc[i-1]+(standard_y[i] - standard_y[i-1]))
            yc.sort()

    xc.sort()
    yc.sort()
    # code = [code[i] for i in range(len(code)) if i not in loss_indexs]
    min_d, detail_content,a = get_deviation(xc, yc, code, each_onset_score,total_frames_number,loss_indexs)

    lost_score = int(each_onset_score * len(loss_indexs))
    ex_score = int(each_onset_score * len(ex_indexs))
    score = 100 - lost_score - ex_score - int(min_d)
    # score, lost_score, ex_score, min_d = get_score1(standard_y.copy(), recognize_y.copy(), len(base_frames),
    #                                                 onsets_frames_strength, min_d)
    # print('最终得分为：{}'.format(score))

    return int(score), int(lost_score), int(ex_score), int(min_d), standard_y, recognize_y, detail_content

def modify_detail_content(detail_content,loss_indexs):
    detail_content = detail_content.split('[')
    detail_content_list = detail_content.split('[')

'''
计算节奏型音频的分数
'''
def get_score_for_note(onsets_frames,base_frames,code):


    recognize_y = onsets_frames

    #min_d, best_y, onsets_frames = get_dtw_min(onsets_frames, base_frames, 65)

    base_frames = [x - (base_frames[0] - onsets_frames[0]) for x in base_frames]
    standard_y = base_frames
    print("standard_y is {}".format(standard_y))

    #code = get_code(type_index,2)
    each_onset_score = 100 / len(standard_y)
    print("each_onset_score is {}".format(each_onset_score))
    ex_recognize_y = []

    ex_recognize_y = []
    # 多唱的情况
    if len(standard_y) < len(recognize_y):
        _, ex_recognize_y = get_mismatch_line(standard_y.copy(), recognize_y.copy())
        # 剥离多唱节拍，便于计算整体偏差分
        modify_recognize_y = [x for x in recognize_y if x not in ex_recognize_y]
        min_d = get_deviation_for_note(standard_y, modify_recognize_y, code, each_onset_score)
    # 漏唱的情况
    if len(standard_y) > len(recognize_y):
        _, lost_standard_y = get_mismatch_line(recognize_y.copy(), standard_y.copy())
        # 加上漏唱节拍，便于计算整体偏差分
        modify_recognize_y = recognize_y.copy()
        for x in lost_standard_y:
            modify_recognize_y.append(x)
            modify_recognize_y.sort()
        min_d = get_deviation_for_note(standard_y, modify_recognize_y, code, each_onset_score)
    if len(standard_y) == len(recognize_y):
        min_d = get_deviation_for_note(standard_y, recognize_y, code, each_onset_score)
    #score = get_score1(standard_y, recognize_y, len(base_frames), onsets_frames_strength, min_d)

    # # 计算成绩测试
    #print('偏移分值为：{}'.format(min_d))
    onsets_frames_strength = np.ones(len(recognize_y))
    onsets_frames_strength = [x *0.5 for x in onsets_frames_strength]
    score,lost_score,ex_score,min_d = get_score_detail_for_note(standard_y, recognize_y, len(base_frames), onsets_frames_strength, min_d)
    #print('最终得分为：{}'.format(score))

    return int(score),int(lost_score),int(ex_score),int(min_d)

'''
计算节奏型音频的分数
'''
def get_score_for_note_v2(onsets_frames,base_frames,rhythm_code):
    rhythm_code = rhythm_code.replace(";", ',')
    rhythm_code = rhythm_code.replace("[", '')
    rhythm_code = rhythm_code.replace("]", '')
    rhythm_code = [x for x in rhythm_code.split(',')]
    recognize_y = onsets_frames
    standard_y = base_frames
    #rhythm_code = get_code(type_index, 2)
    each_onset_score = 100 / len(standard_y)

    if len(standard_y) == len(recognize_y):
        xc, yc = standard_y, recognize_y
    else:
        xc,yc = get_matched_onset_frames_compared(standard_y, recognize_y)

    detail_content = ''
    if len(xc)<1 or len(yc) <1:
        detail_content = '未能识别出匹配的节拍点'
        return 0,0,0,0,detail_content
    std_number = len(standard_y) - len(xc) + len(recognize_y) - len(yc)
    #print("std_number is {}".format(std_number))

    min_d,onset_detail_content = get_deviation_for_note(xc,yc, rhythm_code, each_onset_score)

    detail_content += onset_detail_content

    if (len(standard_y) - len(xc))/len(standard_y) > 0.45:
        detail_content = '与标准节奏相比，存在过多未匹配的节拍'
        score = 30-min_d if 30-min_d > 0 else 10
        return score, 0, 0, 0,detail_content
    # # 计算成绩测试
    #print('偏移分值为：{}'.format(min_d))
    onsets_frames_strength = np.ones(len(recognize_y))
    onsets_frames_strength = [x *0.5 for x in onsets_frames_strength]
    score,lost_score,ex_score,min_d = get_score_detail_for_note(standard_y, recognize_y, len(base_frames), onsets_frames_strength, min_d)
    #print('最终得分为：{}'.format(score))
    # if std_number >= 4:
    #     #print(len(base_frames))
    #     score = int(score - each_onset_score*std_number*0.5)
    #     detail_content = '与标准节奏相比，存在较多未匹配的节拍，整体得分扣减相关的分值'

    return int(score),int(lost_score),int(ex_score),int(min_d),detail_content

'''
计算节奏型音频的分数
'''
def get_score_jz_by_onsets_frames_rhythm(filename,onset_code):

    #type_index = get_onsets_index_by_filename(filename)
    #y, sr = load_and_trim(filename)
    y, sr = librosa.load(filename)
    total_frames_number = get_total_frames_number(filename)

    #onsets_frames, onsets_frames_strength = get_onsets_by_all(y, sr)
    #onsets_frames = get_onsets_frames_for_jz(filename)
    onsets_frames = get_real_onsets_frames_rhythm(y,modify_by_energy=True,gap=0.1)
    if onsets_frames:
        min_width = 5
        # print("min_width is {}".format(min_width))
        onsets_frames = del_overcrowding(onsets_frames, min_width)
        #print("0. onset_frames_cqt is {}".format(onsets_frames))
    #print("onsets_frames len is {}".format(len(onsets_frames)))
    onsets_frames_strength = librosa.onset.onset_strength(y=y, sr=sr)
    onsets_frames_strength = [x/np.max(onsets_frames_strength) for x in onsets_frames_strength]
    # 在此处赋值防止后面实线被移动找不到强度
    onsets_frames = list(set(onsets_frames))
    onsets_frames.sort()
    rms = librosa.feature.rmse(y=y)[0]
    rms = [x / np.std(rms) for x in rms]
    onsets_frames = find_loss_by_rms_for_onsets(onsets_frames, rms, onset_code)
    CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref=np.max)
    CQT = np.where(CQT > -22, np.max(CQT), np.min(CQT))
    onsets_frames, note_lines, times = get_note_lines(CQT, onsets_frames)
    onsets_frames = [onsets_frames[i] for i in range(len(times)) if times[i] > 3]

    # 标准节拍时间点
    if len(onsets_frames) > 0:
        base_frames = onsets_base_frames(onset_code, total_frames_number - onsets_frames[0])
        base_frames = [x + (onsets_frames[0] - base_frames[0]) for x in base_frames]
        min_d, best_y, _ = get_dtw_min(onsets_frames.copy(), base_frames, 65)
    else:
        base_frames = onsets_base_frames(onset_code, total_frames_number)

    onsets_frames = del_same_onsets_by(onsets_frames,CQT,base_frames)
    recognize_y = onsets_frames
    #print("base_frames is {}".format(base_frames))
    #print("base_frames len is {}".format(len(base_frames)))

    min_d, best_y, onsets_frames = get_dtw_min(onsets_frames, base_frames, 65)


    standard_y = best_y.copy()

    code = onset_code
    index = 0
    code = code.replace(";", ',')
    code = code.replace("[", '')
    code = code.replace("]", '')
    if code.find("(") >= 0:
        tmp = [x for x in code.split(',')]
        for i in range(len(tmp)):
            if tmp[i].find("(") >= 0:
                index = i
                break
        code = code.replace("(", '')
        code = code.replace(")", '')
        code = code.replace("-", '')
        code = code.replace("--", '')
    code = [x for x in code.split(',')]
    # code = [int(x) for x in code]
    if index > 0:
        code[index - 1] += code[index]
        del code[index]
    each_onset_score = 100 / len(standard_y)
    xc, yc = get_matched_onset_frames_compared(standard_y, recognize_y)
    std_number = len(standard_y) - len(xc) + len(recognize_y) - len(yc)
    # 去掉未匹配的节拍
    loss_indexs = [i for i in range(len(standard_y)) if standard_y[i] not in xc]
    code = [code[i] for i in range(len(code)) if i not in loss_indexs]
    min_d,detail_content = get_deviation(xc, yc, code, each_onset_score)

    if std_number >= 1:
        # print(len(base_frames))
        if std_number <= 3:
            min_d = int(min_d + each_onset_score * std_number * 0.5)
            detail_content += '。与标准节奏相比，存在少量未匹配的节拍，整体得分扣减相关的分值'
        elif std_number/len(standard_y) < 0.45:
            min_d = int(min_d + each_onset_score * std_number * 0.5)
            detail_content += '。与标准节奏相比，存在较多未匹配的节拍，整体得分扣减相关的分值'
        else:
            detail_content = '与标准节奏相比，存在过多未能匹配对齐的节拍，得分计为不合格'
            return 20, 0, 0, 0, standard_y, recognize_y, detail_content
    # ex_recognize_y = []
    # #多唱的情况
    # if len(standard_y) < len(recognize_y):
    #     _, ex_recognize_y = get_mismatch_line(standard_y.copy(), recognize_y.copy())
    #     # 剥离多唱节拍，便于计算整体偏差分
    #     modify_recognize_y = [x for x in recognize_y if x not in ex_recognize_y]
    #     min_d = get_deviation(standard_y,modify_recognize_y,code,each_onset_score)
    # #漏唱的情况
    # if len(standard_y) > len(recognize_y):
    #     _, lost_standard_y = get_mismatch_line(recognize_y.copy(),standard_y.copy())
    #     # 剥离漏唱节拍，便于计算整体偏差分
    #     modify_standard_y = [x for x in standard_y if x not in lost_standard_y]
    #     min_d = get_deviation(modify_standard_y, recognize_y, code,each_onset_score)
    # if len(standard_y) == len(recognize_y):
    #     min_d = get_deviation(standard_y, recognize_y, code, each_onset_score)
    #score = get_score1(standard_y, recognize_y, len(base_frames), onsets_frames_strength, min_d)

    # # 计算成绩测试
    #print('偏移分值为：{}'.format(min_d))
    score,lost_score,ex_score,min_d = get_score1(standard_y.copy(), recognize_y.copy(), len(base_frames), onsets_frames_strength, min_d)
    #print('最终得分为：{}'.format(score))

    return int(score),int(lost_score),int(ex_score),int(min_d),standard_y, recognize_y,detail_content

def check_gap(b,c):
    print("b is {}".format(b))
    print("c is {}".format(c))
    diff1 = np.diff(b)
    print(diff1)
    #b = [15, 72, 90, 109, 128, 221, 240, 277, 296, 315, 334, 352, 446]
    diff2 = np.diff(c)
    print(diff2)
    c = [np.abs(diff1[i] - diff2[i]) for i in range(len(diff1))]
    print(np.sum(c))
    return np.sum(c)

def find_loss(s1,s2):
    best_gap = 100000
    best_x = 0
    for x in s2[1:]:
        tmp = s1.copy()
        tmp.append(x)
        tmp.sort()
        gap = check_gap(tmp, s2)
        if gap < best_gap:
            best_x = x
            best_gap = gap
    return best_x,best_gap



if __name__ == '__main__':

    # filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏1周(95).wav'

    # filename = './mp3/节奏/节奏1_40227（100）.wav'
    filename = './mp3/节奏/节奏4-01（88）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏1_40441（96）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏8_40213（30）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏四（9）（70）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏十（5）（100）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/节奏/节8王（60）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏十（7）（100）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/节奏/节奏五（4）（100）.wav'
    #filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/节奏/节5熙(35).wav'
    # filename = './mp3/节奏/节奏四（4）（60）.wav'
    # filename = './mp3/节奏/节奏2-02（20）.wav'

    #score, lost_score, ex_score, min_d = get_score_jz(filename)
    #print("score, lost_score, ex_score, min_d is {},{},{},{}".format(score, lost_score, ex_score, min_d))

    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋4谭（95）.wav'
    onsets_frames = [75, 96, 133, 155, 163, 173, 183, 194, 232, 251, 268, 286, 308]
    base_frames = [0, 17, 51, 68, 76, 85, 93, 102, 119, 135, 152, 169, 186, 203]
    recognize_times = [12, 31, 16, 6, 9, 6, 11, 6, 10, 14, 15, 14, 38]
    type_index = get_onsets_index_by_filename_rhythm(filename)
    code = get_code(type_index,2)
    score, lost_score, ex_score, min_d = get_score_for_note(onsets_frames, base_frames, code)
    print("score, lost_score, ex_score, min_d is {},{},{},{}".format(score, lost_score, ex_score, min_d))

    s1 = [42, 49, 65, 124, 169, 213, 237, 258, 294, 307]
    s2 = [0, 19, 38, 75, 93, 112, 149, 168, 186, 214, 223]
    best_x, best_gap = find_loss(s1, s2)
    print("best_x,best_gap is {}===={}".format(best_x,best_gap))
