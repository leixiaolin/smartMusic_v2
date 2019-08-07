# -*- coding: UTF-8 -*-
import numpy as np
import scipy.signal as signal
from base_helper import *
from LscHelper import *
from create_base import *

def get_start_and_end(cqt):
    cqt_bak = cqt.copy()
    cqt = signal.medfilt(cqt_bak, (9, 7))  # 二维中值滤波
    start = 0
    end = 0
    base_pitch = get_base_pitch_from_cqt(cqt)
    base_pitch = signal.medfilt(base_pitch, 7)  # 二维中值滤波
    for i in range(5,len(base_pitch)-1):
        if base_pitch[i] > 0:
            start = i
            break
    cqt = signal.medfilt(cqt_bak, (5, 5))  # 二维中值滤波
    base_pitch = get_base_pitch_from_cqt(cqt)
    base_pitch = signal.medfilt(base_pitch, 7)  # 二维中值滤波
    for i in range(len(base_pitch)-1,10,-1):
        if base_pitch[i] > 0:
            end = i
            break
    return start,end,end - start


def get_base_pitch_from_cqt(cqt):
    h,w = cqt.shape
    cqt_max = np.max(cqt)
    cqt_min = np.min(cqt)
    base_pitch = np.zeros(w)
    for i in range(w):
        col_cqt = cqt[:,i]
        tmp = [i for i in range(10,h-14) if (col_cqt[i] == cqt_max and col_cqt[i+13] == cqt_max and np.min(col_cqt[i:i+13]) == cqt_min) or (col_cqt[i] == cqt_max and np.max(col_cqt[i+13:]) == cqt_min)]
        if len(tmp) > 0:
            base_pitch[i] = tmp[0]
    # selected_base_pitch = []
    # selected_base_pitch.append(base_pitch[0])
    # for i in range(1,len(base_pitch)):
    #     if np.abs(base_pitch[i] - base_pitch[i-1]) > 11 and base_pitch[i] != 0 and base_pitch[i-1] != 0:
    #         base_pitch[i] = base_pitch[i-1]

    return base_pitch

def get_gap_on_cqt(cqt):
    h,w = cqt.shape
    cqt_max = np.max(cqt)
    cqt_min = np.min(cqt)
    gaps = np.zeros(w)
    for i in range(5,w-1):
        current_col_cqt = cqt[10:,i]
        before_col_cqt = cqt[10:,i-1]
        tmp = [1 if np.min(current_col_cqt[i:i+2]) == cqt_max and np.max(before_col_cqt[i:i+2]) == cqt_min else 0 for i in range(len(current_col_cqt)-5)]
        gaps[i] = np.sum(tmp)
    return gaps

def get_start_for_note(cqt):
    h, w = cqt.shape
    cqt_max = np.max(cqt)
    cqt_min = np.min(cqt)
    starts = []
    for i in range(5, w - 1):
        current_col_cqt = cqt[10:70, i]
        before_col_cqt = cqt[10:70, i - 1]
        if np.max(before_col_cqt) == cqt_min and np.max(current_col_cqt) == cqt_max:
            starts.append(i)
    return starts

def get_start_for_amplitude(cqt):
    gaps = get_gap_on_cqt(cqt)
    b, a = signal.butter(8, 0.35, analog=False)
    gaps = signal.filtfilt(b, a, gaps)
    starts = [i for i in range(1,len(gaps)-1) if gaps[i] > gaps[i-1] and gaps[i] > gaps[i+1] and gaps[i] > np.max(gaps)*0.3]
    return starts

def get_change_point_on_pitch(CQT,first_type):
    base_pitch = get_base_pitch_from_cqt(CQT)
    base_pitch = signal.medfilt(base_pitch, 17)  # 二维中值滤波
    all_note_types, all_note_type_position = get_all_note_type(base_pitch, first_type)
    start, end, length = get_start_and_end(CQT)
    all_note_type_position.append(end)
    position_diff = np.diff(all_note_type_position)
    position_diff = [x for x in position_diff if x > 0]
    min_gap_on_position = np.min(position_diff)
    max_gap_on_position = np.max(position_diff)
    max_index = position_diff.index(max_gap_on_position)
    max_position = all_note_type_position[max_index+1]  #长度最大的节拍所在位置
    threshold =  min_gap_on_position * 2.5 if  min_gap_on_position * 2.5 > 4 else 4
    change_points = [all_note_type_position[i] for i in range(0,len(all_note_type_position)-1) if all_note_type_position[i+1] - all_note_type_position[i] < threshold and base_pitch[all_note_type_position[i] + 5] != 0]
    if len(change_points) < 1:
        return []
    select_change_points = []
    for i in range(len(change_points)-1):
        if change_points[i+1] - change_points[i] > threshold:
            select_change_points.append(change_points[i])
    select_change_points.append(change_points[-1])
    select_change_points.append(max_position)
    select_change_points.sort()
    return select_change_points

def get_all_starts_for_note(CQT,first_type,rhythm_code):
    start_for_note = get_start_for_note(CQT)
    # print("start_for_note is{}, size {}".format(start_for_note, len(start_for_note)))
    change_points = get_change_point_on_pitch(CQT, first_type)
    all_starts = start_for_note.copy()
    for c in change_points:
        offset = [np.abs(c - s) for s in start_for_note]
        if np.min(offset) > 6 and c not in all_starts:
            all_starts.append(c)
    all_starts.sort()
    # print("all_starts is{}, size {}".format(all_starts,len(all_starts)))

    start_for_amplitude = get_start_for_amplitude(CQT)
    tmp = all_starts.copy()
    for a in start_for_amplitude:
        offset = [np.abs(a - s) for s in tmp]
        if np.min(offset) > 6 and a not in all_starts:
            all_starts.append(a)
    all_starts.sort()
    # print("all_starts is{}, size {}".format(all_starts, len(all_starts)))

    base_pitch = get_base_pitch_from_cqt(CQT)
    base_pitch = signal.medfilt(base_pitch, 7)  # 二维中值滤波
    start, end, length = get_start_and_end(CQT)
    select_all_starts = []
    for i in range(len(all_starts)):
        if i < len(all_starts)-1:
            if all_starts[i+1] - all_starts[i] > 6 and all_starts[i] > start - 5 and all_starts[i] < end and base_pitch[all_starts[i]] > 5:
                select_all_starts.append(all_starts[i])
        else:
            if all_starts[i] > start - 5 and all_starts[i] < end and base_pitch[all_starts[i]] > 5:
                select_all_starts.append(all_starts[i])
    code = parse_rhythm_code(rhythm_code)
    code = [int(x) for x in code]
    if code[0] >= 1000 and select_all_starts[1] - select_all_starts[0] < length * 250/np.sum(code):
        select_all_starts = select_all_starts[1:]
    # print("select_all_starts is{}, size {}".format(select_all_starts, len(select_all_starts)))
    return  select_all_starts

def get_all_notes(CQT,rhythm_code,filename):

    base_pitch = get_base_pitch_from_cqt(CQT)
    base_pitch = signal.medfilt(base_pitch, 7)  # 二维中值滤波
    s, e, length = get_start_and_end(CQT)
    onset_types, all_starts = get_onset_from_heights_v2(CQT, rhythm_code,filename)
    all_notes = []
    for i in range(len(all_starts)):
        if i < len(all_starts) - 1:
            start = all_starts[i]
            end = all_starts[i+1]
        else:
            start = all_starts[i]
            end = e
        some_base_pitch = base_pitch[start:end]
        some_base_pitch = [x for x in some_base_pitch if x > 0]
        max_item, max_time = get_longest_note(some_base_pitch)
        # note = int(np.mean(some_base_pitch))
        note = max_item
        if note is not None:
            if len(all_notes) > 0 and note - all_notes[-1] > 10:
                note = note -12
            all_notes.append(note)
    return all_notes

def get_longest_note(li):
    max_time = 0  # 已知最大连续出现次数初始为0
    max_item = None
    cur_time = 1  # 记录当前元素是第几次连续出现
    pre_element = None  # 记录上一个元素是什么

    for i in li:
        if i == pre_element:  # 如果当前元素和上一个元素相同,连续出现次数+1,并更新最大值
            cur_time += 1
            if cur_time > max_time:
                max_item = i
            max_time = max((cur_time, max_time))
        else:  # 不同则刷新计数器
            pre_element = i
            cur_time = 1
    return max_item,max_time

def get_all_pitch_type(CQT, first_type,rhythm_code,filename):
    all_notes = get_all_notes(CQT,rhythm_code,filename)
    # print("all_notes is {} ,size {}".format(all_notes,len(all_notes)))
    if len(all_notes) < 1:
        return []
    first_pitch = all_notes[0]
    first_symbol = get_note_symbol(first_type)
    all_note_type = []
    all_note_type.append(first_symbol)
    for i in range(1,len(all_notes)):
        c_pitch = all_notes[i]
        b_pitch = all_notes[i-1]
        c_note = get_note_type(c_pitch, first_pitch, first_type)
        if c_note is not None and b_pitch is not None:
            all_note_type.append(c_note)
        else:
            all_note_type.append(first_type)
    return all_note_type

def get_all_symbols_for_pitch_code(pitch_code):
    code = pitch_code
    code = code.replace("[", '')
    code = code.replace("]", '')
    code = [x for x in code.split(',')]
    result = ''
    for i in range(len(code)):
        c = code[i]
        s = get_note_symbol(c)
        result = result + s
    return result


def get_onset_type(all_starts,rhythm_code,end):
    onset_frames = all_starts.copy()
    onset_frames.append(end)
    if len(onset_frames) == 0:
        return []
    #print("start_index is {},size is {}".format(start_indexs,len(start_indexs)))
    code = parse_rhythm_code(rhythm_code)
    code = [int(x) for x in code]

    #print("code is {},size is {}".format(code, len(code)))

    code_dict = check_code_dict(all_starts, code)
    if code_dict is None:
        total_length_no_last = np.sum(code[0:-1])
        real_total_length_no_last = onset_frames[-1] - onset_frames[0]
        rate = real_total_length_no_last/total_length_no_last
        code_dict = {}
        for x in code:
            code_dict[x] = int(x * rate)
        code_dict[750] = int(750 * rate)
        code_dict[1500] = int(1500 * rate)
        code_dict[2500] = int(2500 * rate)
        code_dict[3000] = int(3000 * rate)
        code_dict[3500] = int(3500 * rate)

    types = []
    for x in np.diff(onset_frames):
        best_min = 100000
        best_key = 1
        for key in code_dict:
            value = code_dict.get(key)
            gap = np.abs(value - x)/x
            if gap<best_min:
                best_min = gap
                best_key = key
        if best_key == 2500 and len(types) <= len(code)-1 and  code[len(types)] == 2000:
            best_key = 2000
        types.append(best_key)
        # if len(types) == 1:
        #     rate = x/best_key
        #     code_dict = get_code_dict(rate, code)

        # if best_key == 250:
        #     current_index = len(types) - 1
        #     if np.diff(onset_frames)[current_index] > code_dict.get(250):
        #         code_dict[250] = int((np.diff(onset_frames)[-1] + code_dict.get(250))/2)
        #         code_dict[500] = code_dict[250]
                # 250 前面不为250的话，后面正常情况下应该为250，所以要在一定范围内修正结果
        # if len(types) >=3:
        #     if types[-3] != 250 and types[-2] == 250 and types[-1] == 500:
        #         current_index = len(types) -1
        #         if np.diff(onset_frames)[current_index] < np.diff(onset_frames)[current_index-1] * 1.5:
        #             types[-1] = 250
        # if len(types) >=4:
        #     if types[-4] == 250 and types[-3] == 250 and types[-2] == 250 and types[-1] == 500:
        #         current_index = len(types) -1
        #         if np.diff(onset_frames)[current_index] < np.diff(onset_frames)[current_index-1] * 1.5:
        #             types[-1] = 250
        onset_frames_diff = np.diff(onset_frames)
        maybe_wrong_indexs = [i for i in range(1,len(types)-1) if types[i-1] == 250 and types[i] == 500]
        all_250 = [onset_frames_diff[i] for i in range(len(types)) if types[i] == 250]
        all_250_mean = np.mean(all_250)
        all_500 = [onset_frames_diff[i] for i in range(len(types)) if types[i] == 500 and i not in maybe_wrong_indexs]
        all_500_mean = np.mean(all_500)
        if len(maybe_wrong_indexs) > 0:
            for i in maybe_wrong_indexs:
                width_500 = onset_frames_diff[i]
                if np.abs(width_500 - all_250_mean) < np.abs(width_500 - all_500_mean):
                    types[i] = 250

    # print("===types is {},size {}".format(types,len(types)))
    # print("===all_starts is {},size {}".format(all_starts, len(all_starts)))
    if code[0] >= 1000 and types[0] == 250:
        types = types[1:]
        all_starts = all_starts[1:]
    if len(types) > 0:
        if types[-1] == 250 and code[-1] >= 1000:  # 最后一个节拍过短，很可能是噪声，可以去掉
            types, all_starts = types[:-1], all_starts[:-1]
        if types[-1] < code[-1]:
            types[-1] = code[-1]  # 最后一个节拍，由于人的习惯不会唱全，所以都识别为标准节拍
    return types,all_starts

def check_code_dict(all_starts,code):
    all_starts_diff = np.diff(all_starts)
    code_rates = []
    for i in range(1,len(code)):
        a = code[i-1]
        b = code[i]
        rate = round(b/a) if b > a else 1/round(a/b)
        rate = round(rate,1)
        code_rates.append(rate)

    starts_rates = []
    for i in range(1,len(all_starts_diff)):
        a = all_starts_diff[i - 1]
        b = all_starts_diff[i]
        rate = round(b / a) if b > a else 1 / round(a / b)
        rate = round(rate, 1)
        starts_rates.append(rate)

    for i in range(1,len(starts_rates)):
        a = starts_rates[i-1]
        b = starts_rates[i]
        for j in range(1,len(code_rates)):
            c = code_rates[j-1]
            d = code_rates[j]
            if a == c and b == d:
                rate = sum(all_starts_diff[j-1:j+2])/sum(code[j-1:j+2])
                code_dict = get_code_dict(rate, code)
                return code_dict
    return None
def get_code_dict(rate,code):
    code_dict = {}
    for x in code:
        code_dict[x] = int(x * rate)
    code_dict[1500] = int(1500 * rate)
    code_dict[2500] = int(2500 * rate)
    code_dict[3000] = int(3000 * rate)
    code_dict[3500] = int(3500 * rate)
    return code_dict

def modify_onset(onset_type,rhythm_code,onset_frames,change_points,end):
    code = parse_rhythm_code(rhythm_code)
    code = [int(x) for x in code]
    modified_onset_frames = onset_frames.copy()
    start_point = 1
    for i in range(1, len(onset_type) - 1):
        b_t = onset_type[i - 1]
        c_t = onset_type[i]
        a_t = onset_type[i + 1]
        end_point = i + 3 if i+3 < len(code) -1 else len(code) -1
        if c_t != code[i] and c_t != code[i - (len(onset_type)-1)]:
            for j in range(start_point, end_point):
                if b_t == code[j - 1] and a_t == code[j + 1] and np.abs(int(c_t) - int(code[j])) > 0:
                    length = onset_frames[i+1] - onset_frames[i]
                    maybe_points = [x for x in change_points if x > onset_frames[i] + length/3 and x < onset_frames[i+1] - length/3]
                    if len(maybe_points) > 0 and len(modified_onset_frames) < len(code):
                        modified_onset_frames.append(int(np.mean(maybe_points)))
                        break
    modified_onset_frames.sort()
    onset_frames = modified_onset_frames
    types, onset_frames = get_onset_type(onset_frames, rhythm_code,end)
    return types,onset_frames
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
        elif i + 1 not in indexs:
            result.append(code[i])
        else:
            t = int(code[i]) + int(code[i + 1])
            result.append(t)
    return result

def parse_pitch_code(pitch_code):
    code = pitch_code
    indexs = []
    code = code.replace("[", '')
    code = code.replace("]", '')
    code = [x for x in code.split(',')]
    result = []
    for i in range(len(code)):
        c = code[i]
        result.append(c)
    return result

def calculate_onset_score(all_starts,end,rhythm_code,threshold_score):
    onset_types,all_starts = get_onset_type(all_starts, rhythm_code,end)
    # print("onset_types  is {} ,size {}".format(onset_types, len(onset_types)))
    all_symbols = get_all_symbols(onset_types)
    #print(all_symbols)
    code = parse_rhythm_code(rhythm_code)
    code = [int(x) for x in code]
    # print("code  is {} ,size {}".format(code, len(code)))
    base_symbols = get_all_symbols(code)
    #print(base_symbols)
    lcs = find_lcseque(base_symbols, all_symbols)
    each_symbol_score = threshold_score/len(code)
    total_score = int(len(lcs)*each_symbol_score)

    detail = get_matched_detail(base_symbols, all_symbols, lcs)

    ex_total = len(all_symbols) - len(lcs) -1
    ex_rate = ex_total / len(base_symbols)
    if len(all_symbols) > len(base_symbols):
        if ex_rate > 0.4:                                # 节奏个数误差超过40%，总分不超过50分
            total_score = total_score if total_score < threshold_score*0.50 else threshold_score*0.50
            detail = detail + "，多唱节奏个数误差超过40%，总分不得超过50分"
        elif ex_rate > 0.3:                             # 节奏个数误差超过30%，总分不超过65分（超过的）（30-40%）
            total_score = total_score if total_score < threshold_score*0.65 else threshold_score*0.65
            detail = detail + "，多唱节奏个数误差超过30%，总分不得超过65分"
        elif ex_rate > 0.2:                             # 节奏个数误差超过20%，总分不超过80分（超过的）（20-30%）
            total_score = total_score if total_score < threshold_score*0.80 else threshold_score*0.80
            detail = detail + "，多唱节奏个数误差超过20%，总分不得超过80分"
        elif ex_rate > 0:                                           # 节奏个数误差不超过20%，总分不超过90分（超过的）（0-20%）
            total_score = total_score if total_score < threshold_score*0.90 else threshold_score*0.90
            detail = detail + "，多唱节奏个数误差在（1-20%），总分不得超过90分"
    return int(total_score),detail

def calculate_note_score(CQT, rhythm_code,pitch_code,threshold_score,filename):
    if pitch_code[2] == '-' or pitch_code[2] == '+':
        first_type = pitch_code[1:3]
    else:
        first_type = pitch_code[1]
    note_types = get_all_pitch_type(CQT, first_type,rhythm_code,filename)
    # print("note_types  is {} ,size {}".format(note_types, len(note_types)))
    all_symbols = get_all_symbols_for_note(note_types)
    # print("1 all_symbols  is {} ,size {}".format(all_symbols, len(all_symbols)))
    # print("all_symbols  is {} ,size {}".format(all_symbols,len(all_symbols)))
    #print(all_symbols)
    code = parse_pitch_code(pitch_code)
    # base_symbols = get_all_symbols_for_note(code)
    base_symbols = get_all_symbols_for_pitch_code(pitch_code)
    # print("base_symbols is {} ,size {}".format(base_symbols, len(base_symbols)))
    # print("2 all_symbols  is {} ,size {}".format(all_symbols, len(all_symbols)))
    all_symbols = modify_pitch(base_symbols, all_symbols)
    # print("3 all_symbols  is {} ,size {}".format(all_symbols, len(all_symbols)))
    #print(base_symbols)
    lcs = find_lcseque_for_note(base_symbols, all_symbols)
    each_symbol_score = threshold_score/len(code)
    total_score = int(len(lcs)*each_symbol_score)

    detail = get_matched_detail(base_symbols, all_symbols, lcs)

    ex_total = len(all_symbols) - len(lcs) -1
    ex_rate = ex_total / len(base_symbols)
    if len(all_symbols) > len(base_symbols):
        if ex_rate > 0.4:                                # 节奏个数误差超过40%，总分不超过50分
            total_score = total_score if total_score < threshold_score*0.50 else threshold_score*0.50
            detail = detail + "，多唱节奏个数误差超过40%，总分不得超过50分"
        elif ex_rate > 0.3:                             # 节奏个数误差超过30%，总分不超过65分（超过的）（30-40%）
            total_score = total_score if total_score < threshold_score*0.65 else threshold_score*0.65
            detail = detail + "，多唱节奏个数误差超过30%，总分不得超过65分"
        elif ex_rate > 0.2:                             # 节奏个数误差超过20%，总分不超过80分（超过的）（20-30%）
            total_score = total_score if total_score < threshold_score*0.80 else threshold_score*0.80
            detail = detail + "，多唱节奏个数误差超过20%，总分不得超过80分"
        elif ex_rate > 0:                                           # 节奏个数误差不超过20%，总分不超过90分（超过的）（0-20%）
            total_score = total_score if total_score < threshold_score*0.90 else threshold_score*0.90
            detail = detail + "，多唱节奏个数误差在（1-20%），总分不得超过90分"
    return int(total_score),detail

def get_all_symbols(types):
    symbols = ''
    for t in types:
        s = get_type_symbol(t)
        symbols = symbols + s
    return symbols

def get_all_symbols_for_note(types):
    symbols = ''
    for t in types:
        # s = get_note_symbol(t)
        symbols = symbols + t
    return symbols

def get_matched_detail(base_symbols, all_symbols,lcs):
    detail_list = np.zeros(len(base_symbols))
    start_index = 0
    for l in lcs:
        index = base_symbols[start_index:].index(l)
        detail_list[start_index + index] = 1
        start_index = start_index + index + 1
    str_detail_list = '识别的结果是：' + str(detail_list)
    str_detail_list = str_detail_list.replace("1", "√")
    str_detail_list = str_detail_list.replace("0", "×")

    ex_total = len(all_symbols) - len(base_symbols)

    if len(all_symbols) > len(base_symbols):
        str_detail_list = str_detail_list + "， 多唱节拍数有：" + str(ex_total)
    return str_detail_list

def modify_pitch(base_symbols, all_symbols):

    modified_symols = [x for x in all_symbols]
    if len(base_symbols) == len(all_symbols):
        for i in range(len(all_symbols)):
            a = all_symbols[i]
            a = get_symbol_index(a)
            b = base_symbols[i]
            b = get_symbol_index(b)
            if np.abs(a - b ) <= 1:
                modified_symols[i] = base_symbols[i]
        return modified_symols
    start_point = 1
    for i in range(1,len(all_symbols)):
        if i < len(all_symbols) - 1:
            b_s = all_symbols[i-1]
            b_s = get_symbol_index(b_s)

            c_s = all_symbols[i]
            c_s = get_symbol_index(c_s)

            a_s = all_symbols[i+1]
            a_s = get_symbol_index(a_s)
            for j in range(start_point,len(base_symbols)-1):
                if np.abs(int(b_s) - get_symbol_index(base_symbols[j-1])) < 1 and np.abs(int(a_s)-get_symbol_index(base_symbols[j+1])) < 1 and np.abs(int(c_s) - get_symbol_index(base_symbols[j])) <= 1:
                    modified_symols[i] = base_symbols[j]
                    start_point = j
                    break
        else:
            c_s = all_symbols[i]
            c_s = get_symbol_index(c_s)
            if np.abs(int(c_s) - get_symbol_index(base_symbols[-1])) <= 1:
                modified_symols[i] = base_symbols[-1]
                break
    result = ''
    for x in modified_symols:
        result = result + x
    return result

def get_starts_with_blank(cqt):
    start, end, length = get_start_and_end(cqt)
    cqt = signal.medfilt(cqt, (3, 3))  # 二维中值滤波
    h,w = cqt.shape
    cqt_max = np.max(cqt)
    cqt_min = np.min(cqt)
    gaps = []
    for i in range(5,w-1):
        current_col_cqt = cqt[30:65,i]
        start_point_on_current = [i+10 for i in range(5,len(current_col_cqt)-1) if current_col_cqt[i] == cqt_max and current_col_cqt[i-1] == cqt_min]
        if len(start_point_on_current) > 0:
            current_col_cqt = cqt[10:start_point_on_current[0] + 13, i]

        before_col_cqt = cqt[30:65,i-1]
        start_point_on_before = [i+10 for i in range(5,len(before_col_cqt)-1) if before_col_cqt[i] == cqt_max and before_col_cqt[i-1] == cqt_min]
        if len(start_point_on_before) > 0:
            before_col_cqt = cqt[10:start_point_on_before[0] + 13, i]

        if np.max(before_col_cqt) == cqt_min and np.max(current_col_cqt) == cqt_max:
            if i > start -5 and i < end:
                gaps.append(i)
    heights = get_height_on_cqt(cqt)
    select_gaps = []
    for g in gaps:
        if heights[g] >= 4 and heights[g] <= heights[g+1]:
            select_gaps.append(g)
    gaps = select_gaps
    if len(select_gaps) > 0 and select_gaps[0] - start > 5:
        select_gaps.append(start)
        select_gaps.sort()
    return gaps

def get_rms_max_indexs_for_onset(filename,threshold = 0.25):
    y, sr = librosa.load(filename)
    CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref=np.max)
    cqt = np.where(CQT > -30, np.max(CQT), np.min(CQT))
    # cqt = signal.medfilt(cqt, (3,3))  # 二维中值滤波
    start, end, length = get_start_and_end(cqt)
    y, sr = librosa.load(filename)
    rms = librosa.feature.rmse(y=y)[0]
    rms_bak = rms.copy();
    rms = [x / np.std(rms) for x in rms]
    rms = list(np.diff(rms))
    rms.insert(0, 0)

    b, a = signal.butter(8, 0.2, analog=False)
    sig_ff = signal.filtfilt(b, a, rms)

    # Savitzky-Golay filter 平滑
    # from scipy.signal import savgol_filter
    # sig_ff = savgol_filter(rms, 5, 1)  # window size 51, polynomial order 3
    # sig_ff = signal.medfilt(rms, 5)  # 二维中值滤波
    sig_ff = [x / np.std(sig_ff) for x in sig_ff]
    max_indexs = [i for i in range(1, len(sig_ff) - 1) if
                  sig_ff[i] > sig_ff[i - 1] and sig_ff[i] > sig_ff[i + 1] and sig_ff[i] > np.max(sig_ff) * 0.15]
    sig_ff_on_max_indexs = [sig_ff[x] for x in max_indexs]
    topN_indexs = find_n_largest(a, 4)
    top_index = sig_ff_on_max_indexs.index(np.max(sig_ff_on_max_indexs))
    hline = np.mean([sig_ff_on_max_indexs[i] for i in range(len(sig_ff_on_max_indexs)) if
                     i in topN_indexs and i != top_index]) * threshold
    max_indexs = [i for i in range(1, len(sig_ff) - 1) if
                  sig_ff[i] > sig_ff[i - 1] and sig_ff[i] > sig_ff[i + 1] and sig_ff[i] > hline]
    max_indexs = [ x for x in max_indexs if x > start - 5 and x < end - 8]
    sig_ff_max = np.max(sig_ff)
    if len(max_indexs) > 0:
        select_max_indexs = []
        select_max_indexs.append(max_indexs[0])
        for i in range(1, len(max_indexs)):
            m = max_indexs[i]
            np_min = np.min(sig_ff[select_max_indexs[-1]:m])
            two_min = min(sig_ff[select_max_indexs[-1]], sig_ff[m])
            two_max = max(sig_ff[select_max_indexs[-1]], sig_ff[m])
            if np_min < two_min and (two_max - np_min) > sig_ff_max * 0.2:
                select_max_indexs.append(m)
    max_indexs = select_max_indexs
    # return max_indexs
    # print("max_indexs is {}, size {}".format(max_indexs,len(max_indexs)))
    return max_indexs

def get_true_max_indexs_for_onset(filename,threshold = 0.85):
    # print("filename is true {}".format(filename))
    y, sr = librosa.load(filename)
    CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref=np.max)
    cqt = np.where(CQT > -30, np.max(CQT), np.min(CQT))
    # cqt = signal.medfilt(cqt, (3,3))  # 二维中值滤波
    start, end, length = get_start_and_end(cqt)
    y, sr = librosa.load(filename)
    rms = librosa.feature.rmse(y=y)[0]
    rms_bak = rms.copy();
    rms = [x / np.std(rms) for x in rms]
    rms = list(np.diff(rms))
    rms.insert(0, 0)

    b, a = signal.butter(8, 0.2, analog=False)
    sig_ff = signal.filtfilt(b, a, rms)

    # Savitzky-Golay filter 平滑
    # from scipy.signal import savgol_filter
    # sig_ff = savgol_filter(rms, 5, 1)  # window size 51, polynomial order 3
    # sig_ff = signal.medfilt(rms, 5)  # 二维中值滤波
    sig_ff = [x / np.std(sig_ff) for x in sig_ff]
    max_indexs = [i for i in range(1, len(sig_ff) - 1) if sig_ff[i] > sig_ff[i - 1] and sig_ff[i] > sig_ff[i + 1] and sig_ff[i] > np.max(sig_ff) * 0.8]
    sig_ff_on_max_indexs = [sig_ff[x] for x in max_indexs]
    topN_indexs = find_n_largest(a, 4)
    top_index = sig_ff_on_max_indexs.index(np.max(sig_ff_on_max_indexs))
    hline = np.mean([sig_ff_on_max_indexs[i] for i in range(len(sig_ff_on_max_indexs)) if i in topN_indexs and i != top_index]) * threshold
    max_indexs = [i for i in range(1, len(sig_ff) - 1) if sig_ff[i] > sig_ff[i - 1] and sig_ff[i] > sig_ff[i + 1] and sig_ff[i] > hline]
    max_indexs = [ x for x in max_indexs if x > start - 5 and x < end - 8]
    # sig_ff_max = np.max(sig_ff)
    if len(max_indexs) > 0:
        select_max_indexs = []
        select_max_indexs.append(max_indexs[0])
        for i in range(1, len(max_indexs)):
            m = max_indexs[i]
            # np_min = np.min(sig_ff[select_max_indexs[-1]:m])
            # two_min = min(sig_ff[select_max_indexs[-1]], sig_ff[m])
            # two_max = max(sig_ff[select_max_indexs[-1]], sig_ff[m])
            # if np_min < two_min and (two_max - np_min) > sig_ff_max * 0.2:
            if m - select_max_indexs[-1] > 10:
                select_max_indexs.append(m)
        max_indexs = select_max_indexs
    # return max_indexs
    # print("max_indexs is {}, size {}".format(max_indexs,len(max_indexs)))
    return max_indexs

def check_max_indexs_by_heigths(filename,cqt_bak,max_indexs,code,end):
    heights = get_height_on_cqt(cqt_bak)
    # Savitzky-Golay filter 平滑
    from scipy.signal import savgol_filter
    b, a = signal.butter(4, 0.25, analog=False)
    sig_ff = signal.filtfilt(b, a, heights)
    select_max_indexs = []
    base_pitch = get_base_pitch_from_cqt(cqt_bak)
    # print("max_indexs is {},size {}".format(max_indexs,len(max_indexs)))
    true_max_indexs = get_true_max_indexs_for_onset(filename)
    # print("true_max_indexs is {},size {}".format(true_max_indexs, len(true_max_indexs)))
    for m in max_indexs:
        check_point = max_indexs.index(m)
        check_result = check_max_indexs_rate(max_indexs.copy(),code,check_point)
        # if np.max(heights[m-3:m+3]) - np.min(heights[m-3:m+3]) > 4 and (sig_ff[m] > np.max(sig_ff) * 0.4 and np.max(base_pitch[m-8:m+2]) - np.min(base_pitch[m-8:m+2]) > 1):
        # if np.max(heights[m-3:m+3]) - np.min(heights[m-3:m+3]) > 4 and np.max(base_pitch[m-8:m+2]) - np.min(base_pitch[m-8:m+2]) > 1:
        # if np.max(heights[m-3:m+3]) - np.min(heights[m-3:m+3]) > 4 and (np.max(base_pitch[m-8:m+2]) - np.min(base_pitch[m-8:m+2]) > 1 or check_result is True):
        if m in true_max_indexs or (np.max(heights[m-3:m+3]) - np.min(heights[m-3:m+3]) > 4 and np.max(base_pitch[m-8:m+2]) - np.min(base_pitch[m-8:m+2]) > 1) or (np.max(heights[m-2:m+2]) - np.min(heights[m-2:m+2]) > 2 and check_result is True):
            select_max_indexs.append(m)
    # print("select_max_indexs is {},size {}".format(select_max_indexs, len(select_max_indexs)))
    return select_max_indexs

def check_max_indexs_rate(all_starts,code,check_point):
    code_rates = []
    for i in range(1,len(code)):
        a = code[i-1]
        b = code[i]
        if a != 0 and b != 0:
            rate = round(b/a) if b > a else 1/round(a/b)
            rate = round(rate,1)
            code_rates.append(rate)

    i = check_point
    if i > 0 and i < len(all_starts) - 2:
        c_onset = all_starts[i+1] - all_starts[i]
        a_onset = all_starts[i+2] - all_starts[i+1]
        b_onset = all_starts[i] - all_starts[i-1]
        a = b_onset
        b = c_onset
        rate = round(b / a) if b > a else 1 / round(a / b)
        rate_first = round(rate, 1)

        a = c_onset
        b = a_onset
        rate = round(b / a) if b > a else 1 / round(a / b)
        rate_second = round(rate, 1)
        range_end = i + 3 if i + 3 < len(code_rates) else len(code_rates)
        for j in range(i-3,range_end):
            c = code_rates[j-1]
            d = code_rates[j]
            if rate_second == c and rate_second == d:
                return True
    return False

def get_starts_by_height_gap(cqt):
    start, end, length = get_start_and_end(cqt)
    heights = get_height_on_cqt(cqt)
    height_max = np.max(heights)
    selected_all_mins, heights_on_min_points = get_all_starts_with_height(cqt)

    starts = []
    for key in heights_on_min_points:
        if key > start - 5 and key < end :
            value = heights_on_min_points.get(key)
            if value >= height_max * 0.3:
                starts.append(key)
    return starts


def get_starts_with_change(cqt):
    start, end, length = get_start_and_end(cqt)
    cqt = signal.medfilt(cqt, (3, 3))  # 二维中值滤波
    h,w = cqt.shape
    cqt_max = np.max(cqt)
    cqt_min = np.min(cqt)
    changes = []
    for n in range(5,w-1):
        current_col_cqt = cqt[10:65,n]
        start_points_on_current = [i for i in range(5,len(current_col_cqt)-1) if (current_col_cqt[i-1] == cqt_min and current_col_cqt[i] == cqt_max) or (current_col_cqt[i-1] == cqt_max and current_col_cqt[i] == cqt_min)]
        before_col_cqt = cqt[10:65,n-1]
        start_points_on_before = [i for i in range(5,len(before_col_cqt)-1) if (before_col_cqt[i-1] == cqt_min and before_col_cqt[i] == cqt_max) or (before_col_cqt[i-1] == cqt_max and before_col_cqt[i] == cqt_min)]
        length = len(start_points_on_current) if len(start_points_on_current) < len(start_points_on_before) else len(start_points_on_before)
        offset = [1 for i in range(0,length-1,2) if (start_points_on_current[i] < start_points_on_before[i] and start_points_on_before[i] - start_points_on_current[i] < 4 and start_points_on_before[i] - start_points_on_current[i] > 0
                  and start_points_on_current[i+1] < start_points_on_before[i+1]  and start_points_on_before[i+1] - start_points_on_current[i+1] < 4 and start_points_on_before[i+1] - start_points_on_current[i+1] > 0)
                  and start_points_on_current[0] != start_points_on_before[0]
                  or (start_points_on_current[i] > start_points_on_before[i] and start_points_on_current[i] - start_points_on_before[i] < 4 and start_points_on_current[i] - start_points_on_before[i] > 0
                  and start_points_on_current[i+1] > start_points_on_before[i+1]  and start_points_on_current[i+1] - start_points_on_before[i+1] < 4 and start_points_on_current[i+1] - start_points_on_before[i+1] > 0)
                  and start_points_on_current[0] != start_points_on_before[0]
                  ]
        if len(offset) > 0:
            changes.append(n)
            # small_number = np.sum([1 for o in offset if o < 5 and o > 0])
            # # if len(start_points_on_current) == 1 or len(start_points_on_before) == 1:
            # #     if small_number == 1:
            # #         changes.append(i)
            # if len(start_points_on_current) > 1 or len(start_points_on_before) > 1:
            #     if small_number > 1:
            #         changes.append(i)
    select_changes = []
    select_changes.append(changes[0])
    for c in changes:
        tmp = cqt[10:65,c+3]
        if np.max(tmp) == cqt_max and c - select_changes[-1] > 5:
            select_changes.append(c)
    return select_changes


def get_height_on_cqt(cqt):

    cqt = signal.medfilt(cqt, (3, 3))  # 二维中值滤波
    h, w = cqt.shape
    cqt_max = np.max(cqt)
    cqt_min = np.min(cqt)
    height = np.zeros(w)
    start,end = 0,0
    for n in range(5, w - 1):
        current_col_cqt = cqt[10:70, n]
        if np.max(current_col_cqt) == cqt_max:
            for i in range(len(current_col_cqt)):
                if current_col_cqt[i] == cqt_max and current_col_cqt[i-1] == cqt_min:
                    start = i
                    break
            for i in range(len(current_col_cqt)-1,0,-1):
                if current_col_cqt[i-1] == cqt_max and current_col_cqt[i] == cqt_min:
                    end = i
                    break
            h = end - start
            height[n] = h
    return height

def get_base_frames(rhythm_code,length):
    base_frames = onsets_base_frames(rhythm_code, length)
    return base_frames

def get_all_max_min_points_on_heigth_cqt(cqt,threshold):
    start, end, length = get_start_and_end(cqt)
    heights = get_height_on_cqt(cqt)
    b, a = signal.butter(4, 0.25, analog=False)
    heights = signal.filtfilt(b, a, heights)
    all_maxs = [i for i in range(1,len(heights)-1) if heights[i] > heights[i-1] and heights[i] > heights[i+1] and i > start and i < end]
    all_mins = [i for i in range(1,len(heights)-1) if heights[i] <= heights[i-1] and heights[i] < heights[i+1] and i > start and i < end]
    all_mins.insert(0,start)
    selected_all_maxs = []
    selected_all_mins = []
    for ma,mi in zip(all_maxs,all_mins):
        if ma > mi and heights[ma] - heights[mi] > threshold:
            selected_all_maxs.append(ma)
            selected_all_mins.append(mi)
    if all_mins[0] not in selected_all_mins:
        selected_all_mins.insert(0,all_mins[0])
        selected_all_maxs.insert(0, all_maxs[0])
    return selected_all_maxs,selected_all_mins


def get_onset_from_heights(cqt,threshold,rhythm_code):
    start, end, length = get_start_and_end(cqt)

    code = parse_rhythm_code(rhythm_code)
    code = [int(x) for x in code]
    base_symbols = get_all_symbols(code)
    # print("base_symbols is {},size {}".format(base_symbols,len(base_symbols)))

    lcs_gap = 10000
    best_onset_types = []
    best_all_starts = []
    best_lcs = []
    best_all_symbols = []
    heights = get_height_on_cqt(cqt)
    height_max = np.max(heights)
    range_start = int(height_max*0.7)
    range_end = int(height_max*0.3) if int(height_max*0.3) < 3 else 3
    starts_with_blank = get_starts_with_blank(cqt)
    for t in range(range_start,range_end,-2):
        threshold = t
        selected_all_maxs, selected_all_mins = get_all_max_min_points_on_heigth_cqt(cqt, threshold)
        if len(selected_all_mins) == len(starts_with_blank):
            onset_types, all_starts = get_onset_type(selected_all_mins, rhythm_code, end)
            best_onset_types = onset_types
            best_all_starts = all_starts
            best_lcs = lcs
            best_all_symbols = all_symbols
            # print("asdfasdfadsfasdfasd=================================" + str(len(starts_with_blank)))
            break
        onset_types, all_starts = get_onset_type(selected_all_mins, rhythm_code, end)
        all_symbols = get_all_symbols(onset_types)
        # print("all_symbols is {},size {}".format(all_symbols, len(all_symbols)))
        lcs = find_lcseque(base_symbols, all_symbols)
        # print("lcs is {},size {}".format(lcs, len(lcs)))
        if len(base_symbols) - len(all_symbols) < lcs_gap and len(all_symbols) <= len(base_symbols):
            best_onset_types = onset_types
            best_all_starts = all_starts
            best_lcs = lcs
            best_all_symbols = all_symbols

    all_symbols, all_starts, lcs = best_all_symbols, best_all_starts, best_lcs
    # print("all_symbols is {},size {}".format(all_symbols,len(all_symbols)))
    # print("all_starts is {},size {}".format(all_starts, len(all_starts)))
    # print("lcs is {},size {}".format(lcs, len(lcs)))
    # print("best_onset_types is {},size {}".format(best_onset_types, len(best_onset_types)))

    if len(lcs) < len(base_symbols) and len(all_symbols) < len(base_symbols):
        threshold = 1
        selected_all_mins = modify_onset_from_heights(cqt, threshold, base_symbols, all_symbols, all_starts, lcs, rhythm_code, end)
        onset_types, all_starts = get_onset_type(selected_all_mins, rhythm_code, end)

    if len(onset_types) > 0:
        if onset_types[-1] == 250 and code[-1] >= 1000:  # 最后一个节拍过短，很可能是噪声，可以去掉
            onset_types, all_starts = onset_types[:-1], all_starts[:-1]
        onset_types[-1] = code[-1] #最后一个节拍，由于人的习惯不会唱全，所以都识别为标准节拍
    else:
        # threshold = 3
        # selected_all_maxs, selected_all_mins = get_all_max_min_points_on_heigth_cqt(cqt, threshold)
        # onset_types, all_starts = get_onset_type(selected_all_mins, rhythm_code, end)
        range_start = int(height_max * 0.7)
        range_end = int(height_max * 0.3)
        for t in range(range_start, range_end, -2):
            threshold = t
            selected_all_maxs, selected_all_mins = get_all_max_min_points_on_heigth_cqt(cqt, threshold)
            onset_types, all_starts = get_onset_type(selected_all_mins, rhythm_code, end)
            all_symbols = get_all_symbols(onset_types)
            # print("all_symbols is {},size {}".format(all_symbols, len(all_symbols)))
            lcs = find_lcseque(base_symbols, all_symbols)
            # print("lcs is {},size {}".format(lcs, len(lcs)))
            if len(base_symbols) - len(all_symbols) < lcs_gap:
                best_onset_types = onset_types
                best_all_starts = all_starts
        onset_types, all_starts = best_onset_types,best_all_starts
    return onset_types,all_starts


def calculate_onset_score_from_symbols(base_symbols, all_symbols,threshold_score):
    # print("base_symbols is {},size {}".format(base_symbols, len(base_symbols)))
    # print("all_symbols is {},size {}".format(all_symbols, len(all_symbols)))
    #print(base_symbols)
    lcs = find_lcseque(base_symbols, all_symbols)
    each_symbol_score = threshold_score/len(base_symbols)
    total_score = int(len(lcs)*each_symbol_score)

    detail = get_matched_detail(base_symbols, all_symbols, lcs)

    ex_total = len(all_symbols) - len(base_symbols)
    ex_rate = ex_total / len(base_symbols)
    if len(all_symbols) > len(base_symbols):
        if ex_rate > 0.4:                                # 节奏个数误差超过40%，总分不超过50分
            total_score = total_score if total_score < threshold_score*0.50 else threshold_score*0.50
            detail = detail + "，多唱节奏个数误差超过40%，总分不得超过总分的0.50"
        elif ex_rate > 0.3:                             # 节奏个数误差超过30%，总分不超过65分（超过的）（30-40%）
            total_score = total_score if total_score < threshold_score*0.65 else threshold_score*0.65
            detail = detail + "，多唱节奏个数误差超过30%，总分不得超过总分的0.65"
        elif ex_rate > 0.2:                             # 节奏个数误差超过20%，总分不超过80分（超过的）（20-30%）
            total_score = total_score if total_score < threshold_score*0.80 else threshold_score*0.80
            detail = detail + "，多唱节奏个数误差超过20%，总分不得超过总分的0.80"
        elif ex_rate > 0:                                           # 节奏个数误差不超过20%，总分不超过90分（超过的）（0-20%）
            total_score = total_score if total_score < threshold_score*0.90 else threshold_score*0.90
            detail = detail + "，多唱节奏个数误差在（1-20%），总分不得超过总分的0.90"
    return int(total_score),detail

def get_wrong_symbols_in_all_symbols(all_symbols,lcs):
    positions = []
    wrong_symbols = []
    all_symbols_list = [x for x in lcs]
    for i in range(len(all_symbols)):
        a = all_symbols[i]
        if i > len(all_symbols_list)-1:
            all_symbols_list.insert(i, a)
            positions.append(i)
            wrong_symbols.append(a)
        if a != all_symbols_list[i]:
            all_symbols_list.insert(i,a)
            positions.append(i)
            wrong_symbols.append(a)
    return wrong_symbols,positions

def get_wrong_symbols_with_base_symbols(base_symbols,all_symbols):
    positions = []
    wrong_symbols = []
    end = len(all_symbols) -1
    for i in range(-1,1-len(all_symbols),-1):
        if all_symbols[i] == base_symbols[i]:
            end = len(all_symbols) + i
        else:
            break
    for i in range(end):
        if i > len(base_symbols) -1:
            return [],[]
        a = all_symbols[i]
        if (a == 'G' and base_symbols[i] == 'I') or (a == 'E' and base_symbols[i] == 'G') or (a == 'D' and base_symbols[i] == 'G'):
            positions.append(i)
            wrong_symbols.append(a)
            break
    return wrong_symbols,positions

def modify_onset_from_heights(cqt, threshold,base_symbols, all_symbols,all_starts,lcs,rhythm_code,end):
    selected_all_maxs, selected_all_mins = get_all_max_min_points_on_heigth_cqt(cqt, threshold)
    new_all_starts = all_starts.copy()
    while len(lcs) < len(base_symbols):
        wrong_symbols, positions = get_wrong_symbols_with_base_symbols(base_symbols,all_symbols)
        if len(wrong_symbols) < 1:
            return new_all_starts
        for w,i in zip(wrong_symbols,positions):
            tmp = [x for x in selected_all_mins if x > all_starts[i] and x < all_starts[i+1]]
            if len(tmp) > 0:
                if w == 'D':
                    new_all_starts.append(all_starts[i] + int((all_starts[i+1] - all_starts[i])/3))
                    new_all_starts.append(all_starts[i] + int((all_starts[i + 1] - all_starts[i])*2 / 3))
                else:
                    new_all_starts.append(all_starts[i] + int((all_starts[i + 1] - all_starts[i]) / 2))
                new_all_starts.sort()
                onset_types, all_starts = get_onset_type(new_all_starts, rhythm_code, end)
                all_symbols = get_all_symbols(onset_types)
                lcs = find_lcseque(base_symbols, all_symbols)
                new_all_starts = all_starts.copy()
            if w == wrong_symbols[-1] and len(tmp) < 1:
                return new_all_starts
    return new_all_starts

def get_all_starts_with_height(cqt):
    threshold = 1
    heights = get_height_on_cqt(cqt)
    b, a = signal.butter(4, 0.25, analog=False)
    heights = signal.filtfilt(b, a, heights)
    all_maxs, all_mins = get_all_max_min_points_on_heigth_cqt(cqt, threshold)
    heights_on_min_points = {}

    selected_all_mins = []
    for ma,mi in zip(all_maxs,all_mins):
        tmp = heights[ma] - heights[mi]
        if tmp > 0:
            heights_on_min_points[mi] = tmp
            selected_all_mins.append(mi)
    return selected_all_mins,heights_on_min_points

def get_onset_from_heights_v2(cqt,rhythm_code,filename):
    start, end, length = get_start_and_end(cqt)
    # print("start ,end is {},{}".format(start,end))
    code = parse_rhythm_code(rhythm_code)
    code = [int(x) for x in code]
    # print("code is {},size {}".format(code, len(code)))
    base_symbols = get_all_symbols(code)
    # print("base_symbols is {},size {}".format(base_symbols,len(base_symbols)))

    starts_with_blank = get_starts_with_blank(cqt)
    max_indexs = get_rms_max_indexs_for_onset(filename, threshold=0.25)
    starts_with_blank = max_indexs
    check_result = check_max_indexs_by_heigths(filename,cqt,max_indexs,code,end)
    starts_with_blank = check_result
    # starts_with_blank = get_starts_by_height_gap(cqt)
    # print("1 starts_with_blank is {},size {}".format(starts_with_blank,len(starts_with_blank)))
    onset_types, all_starts = get_onset_type(starts_with_blank, rhythm_code, end)
    # print("==================================onset_types is {},size {} ========================================".format(onset_types, len(onset_types)))
    all_symbols = get_all_symbols(onset_types)
    # print("all_symbols is {},size {}".format(all_symbols, len(all_symbols)))
    lcs = find_lcseque(base_symbols, all_symbols)
    # print("lcs is {},size {}".format(lcs, len(lcs)))

    if len(all_starts) < len(base_symbols):
        for n in range(8):
            all_starts = modify_onset_from_heights_v2(cqt, onset_types, all_starts, code,end)
            # print("2 starts_with_blank is {},size {}".format(all_starts, len(all_starts)))
            onset_types, all_starts = get_onset_type(all_starts, rhythm_code, end)
            # if all_starts[0] - start > 10:
            #     all_starts.insert(0,start)
            #     all_starts.sort()
            if len(all_starts) >= len(base_symbols):
                break

    if len(onset_types) > 0:
        if onset_types[-1] == 250 and code[-1] >= 1000:  # 最后一个节拍过短，很可能是噪声，可以去掉
            onset_types, all_starts = onset_types[:-1], all_starts[:-1]
        onset_types[-1] = code[-1] #最后一个节拍，由于人的习惯不会唱全，所以都识别为标准节拍
    # print("3 starts_with_blank is {},size {}".format(all_starts, len(all_starts)))
    return onset_types,all_starts

def modify_onset_from_heights_v2(cqt,onset_types, all_starts,code,frame_end):

    selected_all_mins, heights_on_min_points = get_all_starts_with_height(cqt)
    new_all_starts = all_starts.copy()
    for i in range(len(onset_types)):
        check_same = check_from_end(onset_types, code, i)
        check_same = False
        type = onset_types[i]
        if type > code[i] and check_same is False:
            start = all_starts[i]
            if i < len(onset_types) -1:
                end = all_starts[i+1]
            else:
                end = frame_end
            tmp = [x for x in selected_all_mins if x > start and x < end - 6]
            if len(tmp) > 0:
                if type /code[i] == 3:
                    new_all_starts.append(start + int((end - start) / 3))
                    new_all_starts.append(start + int((end - start)*2 / 3))
                elif type /code[i] == 2:
                    new_all_starts.append(start + int((end - start) / 2))
                else:
                    best_index = find_most_heigth(tmp, heights_on_min_points)
                    new_all_starts.append(best_index)
                new_all_starts.sort()
                break
    return new_all_starts

def find_most_heigth(selected_all_mins,heights_on_min_points):
    most = 0
    best_index = None
    for mi in selected_all_mins:
        tmp = heights_on_min_points.get(mi)
        if tmp > most:
            most = tmp
            best_index = mi
    return best_index

#
#   从结尾开始反转判断是否相同
#
def check_from_end(onset_types,code,index):
    index_reverse = len(onset_types) - 1 - index
    base_symbols = get_all_symbols(code)
    all_symbols = get_all_symbols(onset_types)
    base_symbols = base_symbols[::-1]
    all_symbols = all_symbols[::-1]
    if base_symbols[:index_reverse+1] != all_symbols[:index_reverse+1] or index_reverse > len(base_symbols):
        return False
    else:
        return True

def calcalate_total_score(filename, rhythm_code,pitch_code):
    y, sr = librosa.load(filename)
    CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref=np.max)
    cqt = np.where(CQT > -30, np.max(CQT), np.min(CQT))
    onset_types, all_starts = get_onset_from_heights_v2(cqt, rhythm_code,filename)
    # print("==========all_starts is {},size {}".format(all_starts, len(all_starts)))
    all_symbols = get_all_symbols(onset_types)
    # print(all_symbols)

    code = parse_rhythm_code(rhythm_code)
    code = [int(x) for x in code]
    # print("code  is {} ,size {}".format(code, len(code)))
    base_symbols = get_all_symbols(code)
    threshold_score = 40

    # 修正只有一个节拍错误且误差小于500的场景
    all_symbols = modify_onset_when_small_change(code, onset_types,base_symbols, all_symbols)
    onset_score, onset_detail = calculate_onset_score_from_symbols(base_symbols, all_symbols, threshold_score)
    # print("onset_score is {}".format(onset_score))
    # print("detail is {}".format(detail))
    note_score, note_detail = calculate_note_score(cqt, rhythm_code, pitch_code, 60,filename)
    # print("note_score is {}".format(note_score))
    # print("detail is {}".format(detail))
    total_score = onset_score + note_score
    # print("总分 is {}".format(total_score))
    detail = "节奏" + onset_detail + "。 旋律" + note_detail
    return total_score,all_starts,detail

def modify_onset_when_small_change(code, onset_types,base_symbols, all_symbols):
    select_all_symbols = [x for x in all_symbols]
    lcs = find_lcseque(base_symbols, all_symbols)
    if len(base_symbols) - len(lcs) == 1 and len(base_symbols) == len(all_symbols):
        for i in range(len(all_symbols)):
            a = int(onset_types[i])
            b = int(code[i])
            if a != b and np.abs(a - b) <= 500:
                select_all_symbols[i] = base_symbols[i]
                break
    symbols = ''
    for x in select_all_symbols:
        symbols = symbols + x
    return symbols
def draw_plt(filename,rhythm_code, pitch_code):
    y, sr = librosa.load(filename)
    rms = librosa.feature.rmse(y=y)[0]
    rms = [x / np.std(rms) for x in rms]
    time = librosa.get_duration(filename=filename)
    # print("time is {}".format(time))
    CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref=np.max)
    w, h = CQT.shape
    # print("w.h is {},{}".format(w,h))
    # onsets_frames = get_real_onsets_frames_rhythm(y)

    CQT = np.where(CQT > -30, np.max(CQT), np.min(CQT))
    cqt_bak = CQT.copy()

    plt.subplot(3, 1, 1)
    plt.title(filename)
    plt.xlabel("识别结果示意图")
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    CQT = signal.medfilt(cqt_bak, (3, 3))  # 二维中值滤波
    librosa.display.specshow(CQT, x_axis='time')
    starts_with_blank = get_starts_with_blank(cqt_bak)
    # print("starts_with_blank is {}, size {}".format(starts_with_blank,len(starts_with_blank)))
    starts_with_blank_time = librosa.frames_to_time(starts_with_blank)
    plt.vlines(starts_with_blank_time, 0, 12, color='b', linestyle='dashed')

    plt.subplot(3, 1, 2)
    y, sr = librosa.load(filename)
    rms = librosa.feature.rmse(y=y)[0]
    rms_bak = rms.copy();
    rms = [x / np.std(rms) for x in rms]
    rms = list(np.diff(rms))
    rms.insert(0, 0)

    b, a = signal.butter(8, 0.2, analog=False)
    sig_ff = signal.filtfilt(b, a, rms)

    # Savitzky-Golay filter 平滑
    # from scipy.signal import savgol_filter
    # sig_ff = savgol_filter(rms, 5, 1)  # window size 51, polynomial order 3
    # sig_ff = signal.medfilt(rms, 5)  # 二维中值滤波
    sig_ff = [x / np.std(sig_ff) for x in sig_ff]
    # rms = signal.medfilt(rms,3)
    times = librosa.frames_to_time(np.arange(len(rms)))
    plt.plot(times, sig_ff)
    # plt.plot(times, rms)

    max_indexs = get_rms_max_indexs_for_onset(filename, threshold=0.25)
    # print("draw true_max_indexs is {},size {}".format(max_indexs, len(max_indexs)))
    start, end, length = get_start_and_end(cqt_bak)
    # print("start ,end is {},{}".format(start,end))
    code = parse_rhythm_code(rhythm_code)
    code = [int(x) for x in code]
    check_result = check_max_indexs_by_heigths(filename,cqt_bak, max_indexs,code,end)
    max_indexs= check_result
    max_indexs_time = librosa.frames_to_time(max_indexs, sr=sr)
    plt.vlines(max_indexs_time, 0, np.max(rms), color='r', linestyle='solid')
    plt.xlim(0, np.max(times))

    plt.subplot(3, 1, 3)

    heights = get_height_on_cqt(cqt_bak)
    # Savitzky-Golay filter 平滑
    from scipy.signal import savgol_filter
    heights_filter = savgol_filter(heights, 5, 1)  # window size 51, polynomial order 3
    b, a = signal.butter(4, 0.25, analog=False)
    sig_ff = signal.filtfilt(b, a, heights)
    # heights = signal.medfilt(heights, 17)  # 二维中值滤波
    # base_pitch = signal.medfilt(base_pitch, 17)  # 二维中值滤波
    t = librosa.frames_to_time(np.arange(len(heights)))
    plt.plot(t, heights)
    # plt.plot(t, heights_filter)
    plt.plot(t, sig_ff)
    plt.xlim(0, np.max(t))
    plt.ylim(0, 84)
    start, end, length = get_start_and_end(cqt_bak)
    start_time = librosa.frames_to_time(start)
    end_time = librosa.frames_to_time(end)
    plt.vlines(start_time, 40, 80, color='black', linestyle='dashed')
    plt.vlines(end_time, 0, 80, color='black', linestyle='solid')

    base_frames = get_base_frames(rhythm_code, length)
    base_frames = [x - (base_frames[0] - start) for x in base_frames]
    base_frames_time = librosa.frames_to_time(base_frames)
    total_score, all_starts, detail = calcalate_total_score(filename, rhythm_code, pitch_code)
    print("总分 is {}".format(total_score))
    print("detail is {}".format(detail))
    all_mins_time = librosa.frames_to_time(all_starts)
    plt.vlines(all_mins_time, 0, 75, color='r', linestyle='dashed')


    return plt