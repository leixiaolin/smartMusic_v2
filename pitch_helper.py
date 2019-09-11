# -*- coding: UTF-8 -*-
import numpy as np
import scipy.signal as signal
from base_helper import *
from LscHelper import *
from create_base import *
from single_notes.predict_one_onset_alexnet import get_starts_by_alexnet
from split_path_helper import get_split_pic_save_path


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
        tmp = [i for i in range(5,h-14) if (col_cqt[i] == cqt_max and col_cqt[i+13] == cqt_max and np.min(col_cqt[i:i+13]) == cqt_min) or (col_cqt[i] == cqt_max and np.max(col_cqt[i+13:]) == cqt_min)]
        if len(tmp) > 0:
            base_pitch[i] = tmp[0]
    # selected_base_pitch = []
    # selected_base_pitch.append(base_pitch[0])
    # for i in range(1,len(base_pitch)):
    #     if np.abs(base_pitch[i] - base_pitch[i-1]) > 11 and base_pitch[i] != 0 and base_pitch[i-1] != 0:
    #         base_pitch[i] = base_pitch[i-1]
    # base_pitch = signal.medfilt(base_pitch, 17)  # 二维中值滤波
    zero_indexs = [i for i in range(len(base_pitch)) if base_pitch[i] == 0]
    base_pitch = signal.medfilt(base_pitch, 5)  # 二维中值滤波
    # for i in range(10,len(base_pitch)):
    #     b1 = base_pitch[i - 5]
    #     b2 = base_pitch[i]
    #     if np.abs(b2 - b1) > 7 and b1 > 0:
    #         base_pitch[i] = b1
    for z in zero_indexs:
        base_pitch[z] = 0
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
    change_points = [all_note_type_position[i] for i in range(0,len(all_note_type_position)-5) if all_note_type_position[i+1] - all_note_type_position[i] < threshold and base_pitch[all_note_type_position[i] + 5] != 0]
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

def get_all_notes(CQT,rhythm_code,pitch_code,filename):

    base_pitch = get_base_pitch_from_cqt(CQT)
    base_pitch = signal.medfilt(base_pitch, 7)  # 二维中值滤波
    s, e, length = get_start_and_end(CQT)
    onset_types, all_starts = get_onset_from_heights_v3(CQT,rhythm_code,pitch_code,filename)
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

def get_all_notes_from_base_pitch_with_starts(all_starts,base_pitch,start,end):

    s, e = start,end
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
        # print("some_base_pitch is {}".format(some_base_pitch))
        max_item, max_time = get_longest_note(some_base_pitch)
        # print("max_item, max_time is {}, {}".format(max_item, max_time))
        # note = int(np.mean(some_base_pitch))
        note = max_item
        if note is not None:
            if len(all_notes) > 0 and note - all_notes[-1] >= 8:    #大于之前的
                note = note -12
            elif len(all_notes) > 0 and all_notes[-1] - note >= 8:     # 小于之前的
                all_notes[-1] = all_notes[-1] -12
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

def get_all_pitch_type(CQT, first_type,rhythm_code,pitch_code,filename):
    all_notes = get_all_notes(CQT,rhythm_code,pitch_code,filename)
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

def get_all_pitch_type_from_base_pitch_with_starts(first_type,all_starts,base_pitch,start,end):
    all_notes = get_all_notes_from_base_pitch_with_starts(all_starts,base_pitch,start,end)
    print("all_notes is {} ,size {}".format(all_notes,len(all_notes)))
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

def get_all_note_type_for_alexnet(first_type,base_pitch):
    all_notes = base_pitch
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


def get_onset_type(all_starts,rhythm_code,end,init_code_dict=None):
    onset_frames = all_starts.copy()
    onset_frames.append(end)
    if len(onset_frames) == 0:
        return []
    #print("start_index is {},size is {}".format(start_indexs,len(start_indexs)))
    code = parse_rhythm_code(rhythm_code)
    code = [int(x) for x in code]

    #print("code is {},size is {}".format(code, len(code)))

    if init_code_dict is None:
        code_dict = check_code_dict(all_starts, code)
    else:
        code_dict = init_code_dict
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
                rate = sum(all_starts_diff[i-1:i+2])/sum(code[j-1:j+2])
                code_dict = get_code_dict(rate, code)
                return code_dict
    return None

def check_code_dict_with_start_end(all_starts,code,start,end):
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
        rate = 4 if rate == 5 else rate
        starts_rates.append(rate)

    for i in range(1,len(starts_rates)):
        a = starts_rates[i-1]
        b = starts_rates[i]
        for j in range(1,len(code_rates)):
            c = code_rates[j-1]
            d = code_rates[j]
            if a == c and b == d:
                rate = sum(all_starts_diff[i-1:i+2])/sum(code[j-1:j+2])
                base_length = (sum(code)) * rate
                real_length = end -start
                std_length = np.abs(1 -(real_length/base_length))
                if std_length < 0.35:
                    print("real_length/base_length std is {}".format(std_length))
                    code_dict = get_code_dict(rate, code)
                    return code_dict
    rate = 20/500*((end-start)/320)
    code_dict = get_code_dict(rate, code)
    print("code_dict is {},size {}".format(code_dict,len(code_dict)))
    return code_dict

def get_code_dict(rate,code):
    code_dict = {}
    for x in code:
        code_dict[x] = int(x * rate)
    code_dict[750] = int(750 * rate)
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

    detail,detail_list,raw_positions = get_matched_detail(base_symbols, all_symbols, lcs)

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

def calculate_note_score(pitch_code,threshold_score,all_starts,base_pitch,start,end):
    if pitch_code[2] == '-' or pitch_code[2] == '+':
        first_type = pitch_code[1:3]
    else:
        first_type = pitch_code[1]
    note_types = get_all_pitch_type_from_base_pitch_with_starts(first_type,all_starts,base_pitch,start,end)
    # print("note_types  is {} ,size {}".format(note_types, len(note_types)))
    all_symbols = get_all_symbols_for_note(note_types)
    # print("all_symbols  is {} ,size {}".format(all_symbols,len(all_symbols)))
    #print(all_symbols)
    code = parse_pitch_code(pitch_code)
    # print("pitch_code is {} ,size {}".format(code, len(code)))
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

    detail,detail_list,raw_positions = get_matched_detail(base_symbols, all_symbols, lcs)

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

def calculate_note_score_alexnet(pitch_code,threshold_score,all_starts,filename):
    if pitch_code[2] == '-' or pitch_code[2] == '+':
        first_type = pitch_code[1:3]
    else:
        first_type = pitch_code[1]
    base_pitch, base_pitchs = get_base_pitch_by_cqt_and_starts(filename, all_starts)
    tmp = []
    if len(base_pitch) > 1:  # 处理倍频的问题
        if np.abs(base_pitch[0] - base_pitch[1]) >= 8: #第一个节拍倍频问题
            if base_pitch[0] < base_pitch[1]:
                tmp.append(base_pitch[0])
            else:
                tmp.append(base_pitch[1])
        else:
            tmp.append(base_pitch[0])
        for i in range(1,len(base_pitch)):#之后的节拍倍频问题
            if np.abs(base_pitch[i] - tmp[-1]) >= 8:
                tmp.append(base_pitch[i] - 12)
            else:
                tmp.append(base_pitch[i])
        base_pitch = tmp
    # print("base_pitch  is {} ,size {}".format(base_pitch, len(base_pitch)))
    note_types = get_all_note_type_for_alexnet(first_type,base_pitch)
    # print("note_types  is {} ,size {}".format(note_types, len(note_types)))
    all_symbols = get_all_symbols_for_note(note_types)
    # print("all_symbols  is {} ,size {}".format(all_symbols,len(all_symbols)))
    #print(all_symbols)
    code = parse_pitch_code(pitch_code)
    # print("pitch_code is {} ,size {}".format(code, len(code)))
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

    detail,detail_list,raw_positions = get_matched_detail(base_symbols, all_symbols, lcs)

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
    # start_index = 0
    lcseque, positions,raw_positions = my_find_lcseque(base_symbols, all_symbols)
    for index in positions:
        # index = base_symbols[start_index:].index(l)
        detail_list[index] = 1

    str_detail_list = '识别的结果是：' + str(detail_list)
    str_detail_list = str_detail_list.replace("1", "√")
    str_detail_list = str_detail_list.replace("0", "×")

    ex_total = len(all_symbols) - len(base_symbols)

    if len(all_symbols) > len(base_symbols):
        str_detail_list = str_detail_list + "， 多唱节拍数有：" + str(ex_total)
    return str_detail_list,detail_list,raw_positions

def modify_pitch(base_symbols, all_symbols):
    result = ''
    modified_symols = [x for x in all_symbols]
    if len(base_symbols) == len(all_symbols):
        for i in range(len(all_symbols)):
            a = all_symbols[i]
            a = get_symbol_index(a)
            b = base_symbols[i]
            b = get_symbol_index(b)
            if np.abs(a - b ) <= 1:
                modified_symols[i] = base_symbols[i]
        for x in modified_symols:
            result = result + x
        return result
    start_point = -1
    #前向比较
    lenght = len(all_symbols) if len(all_symbols) < len(base_symbols) else len(base_symbols)
    for i in range(lenght):
        a = all_symbols[i]
        a = get_symbol_index(a)
        b = base_symbols[i]
        b = get_symbol_index(b)
        if np.abs(a - b) <= 1:
            modified_symols[i] = base_symbols[i]
        else:
            start_point = i
            break
    if start_point == -1:
        for x in modified_symols:
            result = result + x
        return result
    #后向比较
    end_point = 0
    for i in range(-1,0-lenght-1,-1):
        a = all_symbols[i]
        a = get_symbol_index(a)
        b = base_symbols[i]
        b = get_symbol_index(b)
        if np.abs(a - b) <= 1:
            modified_symols[i] = base_symbols[i]
        else:
            end_point = i
            break
    if end_point == 0:
        for x in modified_symols:
            result = result + x
        return result
    end_point = len(all_symbols) + end_point
    all_symbols = modified_symols
    for i in range(start_point,end_point):
        if i < len(all_symbols) - 1:
            b_s = all_symbols[i-1]
            b_s = get_symbol_index(b_s)

            c_s = all_symbols[i]
            c_s = get_symbol_index(c_s)

            a_s = all_symbols[i+1]
            a_s = get_symbol_index(a_s)
            for j in range(0,len(base_symbols)-1):
                if np.abs(int(b_s) - get_symbol_index(base_symbols[j-1])) <= 1 and np.abs(int(a_s)-get_symbol_index(base_symbols[j+1])) <= 1 and np.abs(int(c_s) - get_symbol_index(base_symbols[j])) <= 1:
                    modified_symols[i] = base_symbols[j]
                    start_point = j
                    break
        else:
            c_s = all_symbols[i]
            c_s = get_symbol_index(c_s)
            if np.abs(int(c_s) - get_symbol_index(base_symbols[-1])) <= 1:
                modified_symols[i] = base_symbols[-1]
                break

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
        heights_tmp = heights[m:]
        tmp = list(heights_tmp).index(0)
        if tmp > 10:
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


def calculate_onset_score_from_symbols(base_symbols, all_symbols,starts, onset_types,threshold_score):
    # print("base_symbols is {},size {}".format(base_symbols, len(base_symbols)))
    # print("all_symbols is {},size {}".format(all_symbols, len(all_symbols)))
    #print(base_symbols)
    # print("2 finally onset_types is {}, size {}".format(onset_types, len(onset_types)))

    offset_detail = ''

    offset_threshold = 180
    types, real_types = get_offset_for_each_onsets_by_speed(starts, onset_types)
    # print("3 finally onset_types is {}, size {}".format(onset_types, len(onset_types)))
    # offset_indexs = [i for i in range(len(types) - 1) if np.abs(types[i] - real_types[i]) > offset_threshold]  # 找出偏差大于125的节拍
    baseline_offset = [np.abs(types[i] - real_types[i]) for i in range(len(types)) if types[i] == np.min(types)]
    baseline_offset = np.min(baseline_offset) #基准偏差
    # 找出偏差大于125的节拍，判断是要减掉基准偏差
    offset_indexs = [i for i in range(len(types)-1) if np.abs(types[i] - real_types[i]) > baseline_offset * int(types[i]/np.min(types)) and np.abs(types[i] - real_types[i]) - baseline_offset * int(types[i]/np.min(types))  > offset_threshold]
    if len(offset_indexs) > 0:
        str_tmp = list(all_symbols)
        for i in offset_indexs:
            str_tmp[i] = '0'
        all_symbols = ''.join(str_tmp)
        offset_values = [np.abs(types[i] - real_types[i]) for i in range(len(types))]
        offset_detail = "。判定音符类型为 {}，实际音符为 {}，偏差值为 {}，其中大于{}的也都会被视为错误节拍（不包括最后一个节拍）".format(types, real_types, offset_values, offset_threshold)

    lcs = find_lcseque(base_symbols, all_symbols)
    each_symbol_score = threshold_score/len(base_symbols)
    total_score = int(len(lcs)*each_symbol_score)

    detail,detail_list,raw_positions = get_matched_detail(base_symbols, all_symbols, lcs)
    detail = detail + offset_detail

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
    return int(total_score),detail,detail_list,raw_positions

def get_offset_for_each_onsets_by_speed(max_indexs, types):
    index_diff = np.diff(max_indexs)
    vs = [int(types[i]) / index_diff[i] for i in range(len(index_diff))]
    real_types = [int(d * np.mean(vs)) for d in index_diff]
    # print("index_diff is {},size is {}".format(index_diff, len(index_diff)))
    # print("vs is {},size is {}".format(vs, len(vs)))
    # print("vs mean is {}".format(np.mean(vs)))
    # print("types is {},size is {}".format(types, len(types)))
    # print("real_types is {},size is {}".format(real_types, len(real_types)))
    # print("code is {},size is {}".format(code, len(code)))
    return types,real_types

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
    code = parse_rhythm_code(rhythm_code)
    code = [int(x) for x in code]

    # print("code is {},size is {}".format(code, len(code)))
    code_dict = check_code_dict_with_start_end(starts_with_blank, code,start,end)
    onset_types, all_starts = get_onset_type(starts_with_blank, rhythm_code, end,init_code_dict=code_dict)
    # print("==================================onset_types is {},size {} ========================================".format(onset_types, len(onset_types)))
    all_symbols = get_all_symbols(onset_types)
    # print("all_symbols is {},size {}".format(all_symbols, len(all_symbols)))
    lcs = find_lcseque(base_symbols, all_symbols)
    # print("lcs is {},size {}".format(lcs, len(lcs)))

    if len(all_starts) < len(base_symbols):
        for n in range(8):
            all_starts = modify_onset_from_heights_v2(cqt, onset_types, all_starts, code,end)
            # print("2 starts_with_blank is {},size {}".format(all_starts, len(all_starts)))
            onset_types, all_starts = get_onset_type(all_starts, rhythm_code, end,init_code_dict=code_dict)
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

def get_onset_from_heights_v3(cqt,rhythm_code,pitch_code,filename):
    start, end, length = get_start_and_end(cqt)
    # print("start ,end is {},{}".format(start,end))
    code = parse_rhythm_code(rhythm_code)
    code = [int(x) for x in code]


    select_starts = get_all_starts_by_rms_and_note_change_position(filename,rhythm_code, pitch_code)

    #最后一个为2000,判断结尾是否需要去掉噪声
    if code[-1] == 2000 and end - select_starts[-1] < 10:
        select_starts = select_starts[:-1]

    # print("select_starts is {},size is {}".format(select_starts, len(select_starts)))
    select_starts.sort()
    code_dict = check_code_dict_with_start_end(select_starts, code,start,end)
    onset_types, all_starts = get_onset_type(select_starts, rhythm_code, end,init_code_dict=code_dict)
    # print("==================================onset_types is {},size {} ========================================".format(onset_types, len(onset_types)))



    if len(onset_types) > 0:
        if onset_types[-1] == 250 and code[-1] >= 1000:  # 最后一个节拍过短，很可能是噪声，可以去掉
            onset_types, all_starts = onset_types[:-1], all_starts[:-1]
        onset_types[-1] = code[-1] #最后一个节拍，由于人的习惯不会唱全，所以都识别为标准节拍
    # print("3 starts_with_blank is {},size {}".format(all_starts, len(all_starts)))
    all_starts.sort()
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
    CQT = signal.medfilt(CQT, (5, 5))  # 二维中值滤波
    CQT = np.where(CQT > -35, np.max(CQT), np.min(CQT))
    start, end, length = get_start_and_end(CQT)
    code = parse_rhythm_code(rhythm_code)
    code = [int(x) for x in code]
    # onset_types, all_starts = get_onset_from_heights_v3(cqt, rhythm_code,pitch_code,filename)
    onset_types, all_starts,base_pitch = get_all_starts_by_optimal(filename, rhythm_code,pitch_code)
    #如果个数相等，修正偏差大于500的节拍
    if len(onset_types) == len(code):
        for i in range(len(onset_types)):
            o = onset_types[i]
            if np.abs(o - code[i]) <= 500:
                onset_types[i] = code[i]
    # print("finally==========onset_types is {},size {}".format(onset_types, len(onset_types)))
    # print("finally==========all_starts is {},size {}".format(np.diff(all_starts), len(onset_types)-1))

    all_symbols = get_all_symbols(onset_types)
    # print(all_symbols)

    # print("code  is {} ,size {}".format(code, len(code)))
    base_symbols = get_all_symbols(code)
    threshold_score = 40

    # 修正只有一个节拍错误且误差小于500的场景
    # print("all_symbols  is {} ,all_symbols {}".format(all_symbols, len(all_symbols)))
    all_symbols = modify_onset_when_small_change(code, onset_types,base_symbols, all_symbols)
    # print("all_symbols  is {} ,all_symbols {}".format(all_symbols, len(all_symbols)))
    # print("base_symbols  is {} ,base_symbols {}".format(base_symbols, len(base_symbols)))
    onset_score, onset_detail,detail_list,raw_positions = calculate_onset_score_from_symbols(base_symbols, all_symbols, threshold_score)
    # print("onset_score is {}".format(onset_score))
    # print("detail is {}".format(detail))
    threshold_score = 60
    note_score, note_detail = calculate_note_score(pitch_code,threshold_score,all_starts,base_pitch,start,end)
    # print("note_score is {}".format(note_score))
    # print("detail is {}".format(detail))
    total_score = onset_score + note_score
    # print("总分 is {}".format(total_score))
    detail = "节奏" + onset_detail + "。 旋律" + note_detail
    return total_score,all_starts,detail

def calcalate_total_score_by_alexnet(filename, rhythm_code,pitch_code):
    y, sr = librosa.load(filename)
    CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref=np.max)
    CQT = signal.medfilt(CQT, (5, 5))  # 二维中值滤波
    CQT = np.where(CQT > -35, np.max(CQT), np.min(CQT))
    start, end, length = get_start_and_end(CQT)
    code = parse_rhythm_code(rhythm_code)
    code = [int(x) for x in code]
    # onset_types, all_starts = get_onset_from_heights_v3(cqt, rhythm_code,pitch_code,filename)
    # savepath = 'E:/t/'  # 保存要测试的目录
    # savepath = '/home/lei/bot-rating/split_pic'
    savepath = get_split_pic_save_path()
    # init_data(filename, rhythm_code, savepath)  # 切分潜在的节拍点，并且保存切分的结果
    onset_types, all_starts,base_pitch,change_points = get_all_starts_by_alexnet(filename, rhythm_code,pitch_code)
    # print("1 finally onset_types is {}, size {}".format(onset_types, len(onset_types)))
    # print("1 finally all_starts is {}, size {}".format(all_starts, len(all_starts)))
    # print("1 finally change_points is {}, size {}".format(change_points, len(change_points)))
    #如果个数相等，修正偏差大于500的节拍
    # if len(onset_types) == len(code):
    #     for i in range(len(onset_types)):
    #         o = onset_types[i]
    #         if np.abs(o - code[i]) <= 500:
    #             onset_types[i] = code[i]
    # print("finally==========onset_types is {},size {}".format(onset_types, len(onset_types)))
    # print("finally==========all_starts is {},size {}".format(np.diff(all_starts), len(onset_types)-1))

    all_symbols = get_all_symbols(onset_types)
    # print(all_symbols)

    # print("code  is {} ,size {}".format(code, len(code)))
    base_symbols = get_all_symbols(code)
    threshold_score = 40

    # 修正只有一个节拍错误且误差小于500的场景
    # print("all_symbols  is {} ,all_symbols {}".format(all_symbols, len(all_symbols)))
    all_symbols = modify_onset_when_small_change(code, onset_types,base_symbols, all_symbols)
    # print("all_symbols  is {} ,all_symbols {}".format(all_symbols, len(all_symbols)))
    # print("base_symbols  is {} ,base_symbols {}".format(base_symbols, len(base_symbols)))
    onset_frames = all_starts.copy()
    onset_frames.append(end)
    onset_frames.sort()
    onset_score, onset_detail,detail_list,raw_positions = calculate_onset_score_from_symbols(base_symbols, all_symbols, onset_frames,onset_types, threshold_score)
    # print("detail_list is {}".format(detail_list))
    # print("raw_positions is {}".format(raw_positions))
    # 如果个数少于标准个数
    if len(onset_frames) < len(code):
        pass
    # print("onset_score is {}".format(onset_score))
    # print("detail is {}".format(detail))
    threshold_score = 60
    note_score1, note_detail1 = calculate_note_score(pitch_code,threshold_score,all_starts,base_pitch,start,end)
    note_score2, note_detail2 = calculate_note_score_alexnet(pitch_code, threshold_score, all_starts, filename)
    if note_score1 > note_score2:
        note_score,note_detail = note_score1, note_detail1
    else:
        note_score, note_detail = note_score2, note_detail2
        # print("note_score is {}".format(note_score))
    # print("detail is {}".format(detail))
    total_score = onset_score + note_score
    # print("总分 is {}".format(total_score))
    detail = "旋律" + note_detail + "。节奏" + onset_detail
    return total_score,all_starts,detail

def add_loss_by_positions(detail_list,raw_positions,onset_types, all_starts,base_pitch,change_points):
    for i,x in enumerate(detail_list):
        if x == 0 and detail_list[i+1] == 0: # 连续错误
           start = i - len([0 for i in range(detail_list[:i])])
def check_total_score_from_starts_and_base_pitch(all_starts,base_pitch,rhythm_code,pitch_code, start, end):
    code = parse_rhythm_code(rhythm_code)
    code = [int(x) for x in code]
    code_dict = get_code_dict_by_min_diff(all_starts, code, start, end)
    onset_types, all_starts = get_onset_type_by_code_dict(all_starts, rhythm_code, end, code_dict)
    if code[-1] - onset_types[-1] <= 1000:
        onset_types[-1] = code[-1]  # 最后一个节拍，由于人的习惯不会唱全，所以都识别为标准节拍

    # 如果个数相等，修正偏差大于500的节拍
    # if len(onset_types) == len(code):
    #     for i in range(len(onset_types)):
    #         o = onset_types[i]
    #         if np.abs(o - code[i]) <= 500:
    #             onset_types[i] = code[i]
    # print("finally==========onset_types is {},size {}".format(onset_types, len(onset_types)))
    # print("finally==========all_starts is {},size {}".format(np.diff(all_starts), len(onset_types) - 1))

    all_symbols = get_all_symbols(onset_types)
    # print(all_symbols)

    # print("code  is {} ,size {}".format(code, len(code)))
    base_symbols = get_all_symbols(code)
    threshold_score = 40

    # 修正只有一个节拍错误且误差小于500的场景
    # print("all_symbols  is {} ,all_symbols {}".format(all_symbols, len(all_symbols)))
    # all_symbols = modify_onset_when_small_change(code, onset_types, base_symbols, all_symbols)
    # print("all_symbols  is {} ,all_symbols {}".format(all_symbols, len(all_symbols)))
    # print("base_symbols  is {} ,base_symbols {}".format(base_symbols, len(base_symbols)))
    onset_score, onset_detail,detail_list,raw_positions = calculate_onset_score_from_symbols(base_symbols, all_symbols, threshold_score)
    threshold_score = 60
    note_score, note_detail = calculate_note_score(pitch_code, threshold_score, all_starts, base_pitch, start, end)
    # print("note_score is {}".format(note_score))
    # print("detail is {}".format(detail))
    total_score = onset_score + note_score
    return total_score

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

def get_starts_from_base_pitch(CQT):
    start, end, length = get_start_and_end(CQT)
    base_pitch = get_base_pitch_from_cqt(CQT)
    starts = [i for i in range(1,len(base_pitch)) if base_pitch[i] > 0 and base_pitch[i-1] == 0]
    starts = [i for i in starts if i > start - 5 and i < end]
    return starts

def get_starts_by_base_pitch(base_pitch,start,end):
    # start, end, length = get_start_and_end(CQT)
    # base_pitch = get_base_pitch_from_cqt(CQT)
    starts = [i for i in range(1,len(base_pitch)) if base_pitch[i] > 0 and base_pitch[i-1] == 0]
    starts = [i for i in starts if i > start - 5 and i < end]
    return starts

def get_starts_from_rms_by_threshold(filename,threshold):
    y, sr = librosa.load(filename)
    rms = librosa.feature.rmse(y=y)[0]
    rms_bak = rms.copy();
    rms = [x / np.std(rms) for x in rms]
    rms = list(np.diff(rms))
    rms.insert(0, 0)

    b, a = signal.butter(8, 0.5, analog=False)
    sig_ff = signal.filtfilt(b, a, rms)
    # starts = [i for i in range(1,len(sig_ff)-1) if sig_ff[i] > sig_ff[i-1] and sig_ff[i] > sig_ff[i+1] and sig_ff[i] > np.max(sig_ff)* threshold]
    # starts = [i for i in range(1,len(sig_ff)-1) if (sig_ff[i] > sig_ff[i-1] and sig_ff[i] > sig_ff[i+1] and sig_ff[i] > np.max(sig_ff)* threshold) or (sig_ff[i] > sig_ff[i-1] and sig_ff[i] > sig_ff[i+1] and sig_ff[i] < 0 and sig_ff[i] - np.min(sig_ff) > np.max(sig_ff)* threshold)]
    starts = [i for i in range(1,len(sig_ff)-1) if sig_ff[i] > sig_ff[i-1] and sig_ff[i] > sig_ff[i+1] and sig_ff[i] > 0.05]
    tmp = []
    for x in starts:
        p_points = [i for i in range(1,len(sig_ff[:x])-1) if sig_ff[i] <sig_ff[i-1] and sig_ff[i] < sig_ff[i+1]]
        if len(p_points) > 0:
            p = p_points[-1]
            if sig_ff[x] - sig_ff[p] > np.max(sig_ff)* threshold:
                tmp.append(x)
    starts = tmp
    select_starts = []
    select_starts.append(starts[0])
    if threshold > 0.1:
        gap = 5
    else:
        gap = 9
    for i in range(1,len(starts)):
        if starts[i] - select_starts[-1] > gap:
            select_starts.append(starts[i])
        else:
            if sig_ff[starts[i]] > sig_ff[select_starts[-1]]:
                select_starts[-1] = starts[i]
    return select_starts

def check_note_type_position(cqt,base_pitch,all_note_type_position):
    select_all_note_type_position = []
    for x in all_note_type_position[2:]:
        if x - 5 < 0:
            continue
        flag = True
        row = int(base_pitch[x])
        row_col_cqt = cqt[row:row+9,x - 4:x + 4]
        h,w = row_col_cqt.shape
        for i in range(h-1):
            if np.min(row_col_cqt[i,:]) == np.max(cqt) and np.max(row_col_cqt[i+1,:]) == np.min(cqt):
                flag = False
                break
        if flag is True and np.max(base_pitch[x-5:x]) < 40:
            if len(select_all_note_type_position) == 0:
                select_all_note_type_position.append(x)
            elif x - select_all_note_type_position[-1] > 10:
                select_all_note_type_position.append(x)
    return select_all_note_type_position

def get_all_starts_by_rms_and_note_change_position(filename,rhythm_code,pitch_code):
    y, sr = librosa.load(filename)

    CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref=np.max)

    CQT = np.where(CQT > -35, np.max(CQT), np.min(CQT))

    CQT = signal.medfilt(CQT, (5, 5))  # 二维中值滤波
    start, end, length = get_start_and_end(CQT)
    base_pitch = get_base_pitch_from_cqt(CQT)
    base_pitch_filter = signal.medfilt(base_pitch, 11)  # 二维中值滤波
    first_type = pitch_code[1]
    all_note_types, all_note_type_position = get_all_note_type(base_pitch, first_type)
    all_note_type_position = check_note_type_position(CQT, base_pitch_filter, all_note_type_position)
    # print("2 all_note_type_position is {} ,size {}".format(all_note_type_position, len(all_note_type_position)))
    starts_on_highest_point = get_starts_on_highest_point_of_cqt(CQT)  #最高点变化曲线的波谷点
    threshold = 0.4  # 振幅大的大概率是节拍
    starts_from_rms_must = get_starts_from_rms_by_threshold(filename, threshold)
    select_starts_from_rms_must = []
    for x in starts_from_rms_must: # 去掉没有音高跳跃点的伪节拍
        offset = [np.abs(x - a) for a in all_note_type_position if a < x + 6]
        offset2 = [np.abs(x - a) for a in starts_on_highest_point if a < x + 2]  # 与最高点变化曲线波谷点的距离
        if ((len(offset) > 0 and np.min(offset) < 12) or (len(offset2) > 0 and np.min(offset2) < 6))and np.max(base_pitch[x+2:x+3]) != 0 and (np.max(base_pitch[x:x + 5]) != 0 or np.min(base_pitch[x-5:x]) == 0):
            select_starts_from_rms_must.append(x)
    starts_from_rms_must = select_starts_from_rms_must
    if starts_from_rms_must[0] - start > 15: #如果没包括开始点，则需要添加开始点
        starts_from_rms_must.append(start)
    starts_from_rms_must.sort()

    threshold = 0.15  # 振幅小的小概率是节拍
    starts_from_rms_maybe = get_starts_from_rms_by_threshold(filename, threshold)
    starts_from_rms_maybe = [x for x in starts_from_rms_maybe if x not in starts_from_rms_must]
    select_starts_from_rms_maybe = []
    for x in starts_from_rms_maybe: # 去掉音高为0的伪节拍
        offset = [np.abs(x - a) for a in all_note_type_position if a < x + 2]
        if np.max(base_pitch[x:x + 5]) != 0:
            select_starts_from_rms_maybe.append(x)
    starts_from_rms_maybe = select_starts_from_rms_maybe
    print("starts_from_rms_maybe is {}, size {}".format(starts_from_rms_maybe,len(starts_from_rms_maybe)))
    print("starts_from_rms_maybe diff is {}, size {}".format(np.diff(starts_from_rms_maybe), len(starts_from_rms_maybe)-1))

    select_starts = starts_from_rms_must.copy()

    for x in starts_from_rms_maybe:
        offset = [np.abs(x - a) for a in all_note_type_position if a < x + 3]   #与音高跳跃点的距离
        offset2 = [np.abs(x - a) for a in starts_on_highest_point if a < x + 2]  #与最高点变化曲线波谷点的距离
        check_by_long_line = check_by_long_line_in_cqt(x, start, end, CQT) # 判断是否位于节拍中部
        # if check_by_long_line is False and len(offset) > 0 and ((np.min(offset) < 15 and np.max(base_pitch[x:x+3]) != 0) or (len(offset2) > 0 and np.min(offset2) < 6)): # 位于音高跳跃点和最高点变化曲线波谷点附近， 且不位于节拍断层区域
        if check_by_long_line is False and len(offset) > 0 and np.min(offset) < 8 and (x+3 < len(base_pitch) and np.max(base_pitch[x+2:x+3]) != 0) and len(offset2) > 0 and np.min(offset2) < 6: # 位于音高跳跃点和最高点变化曲线波谷点附近， 且不位于节拍断层区域
            select_starts.append(x)
    select_starts = [ x for x in select_starts if x > start - 5 and x < end]
    select_starts.sort()

    tmp = []
    tmp.append(int(select_starts[0]))
    # 判断节拍与前一节拍之间是否有波峰和波谷
    for i in range(1,len(select_starts)):
        x = int(select_starts[i])
        y = int(select_starts[i-1])
        offset = [s for s in starts_on_highest_point if s > y and s < x]
        if len(offset) > 0 and x - tmp[-1] > 6:
            tmp.append(x)
    select_starts = tmp
    # code = parse_rhythm_code(rhythm_code)
    # code = [int(x) for x in code]
    # if code[-1] >= 1000:
    #     select_starts = [x for x in select_starts if x <= starts_from_rms_must[-1]]
    # select_starts = add_loss_rms(filename, starts_from_rms_maybe, select_starts)

    return select_starts

def get_all_starts_by_rms_and_note_change_position_v2(filename,rhythm_code,pitch_code):
    y, sr = librosa.load(filename)

    CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref=np.max)

    CQT = np.where(CQT > -35, np.max(CQT), np.min(CQT))

    CQT = signal.medfilt(CQT, (5, 5))  # 二维中值滤波
    start, end, length = get_start_and_end(CQT)
    base_pitch = get_base_pitch_from_cqt(CQT)
    base_pitch_filter = signal.medfilt(base_pitch, 11)  # 二维中值滤波
    first_type = pitch_code[1]
    all_note_types, all_note_type_position = get_all_note_type(base_pitch, first_type)
    all_note_type_position = check_note_type_position(CQT, base_pitch_filter, all_note_type_position)
    # print("2 all_note_type_position is {} ,size {}".format(all_note_type_position, len(all_note_type_position)))
    starts_on_highest_point = get_starts_on_highest_point_of_cqt(CQT)  #最高点变化曲线的波谷点
    # 振幅大的大概率是节拍
    starts_from_rms_must = get_must_starts(filename, 1.5)

    threshold = 0.15  # 振幅小的小概率是节拍
    starts_from_rms_maybe = get_starts_from_rms_by_threshold(filename, threshold)
    starts_from_rms_maybe = [x for x in starts_from_rms_maybe if x not in starts_from_rms_must]
    select_starts_from_rms_maybe = []
    for x in starts_from_rms_maybe: # 去掉音高为0的伪节拍
        offset = [np.abs(x - a) for a in all_note_type_position if a < x + 2]
        if np.max(base_pitch[x:x + 5]) != 0:
            select_starts_from_rms_maybe.append(x)
    starts_from_rms_maybe = select_starts_from_rms_maybe
    print("starts_from_rms_maybe is {}, size {}".format(starts_from_rms_maybe,len(starts_from_rms_maybe)))
    print("starts_from_rms_maybe diff is {}, size {}".format(np.diff(starts_from_rms_maybe), len(starts_from_rms_maybe)-1))

    select_starts = starts_from_rms_must.copy()

    for x in starts_from_rms_maybe:
        offset = [np.abs(x - a) for a in all_note_type_position if a < x + 3]   #与音高跳跃点的距离
        offset2 = [np.abs(x - a) for a in starts_on_highest_point if a < x + 2]  #与最高点变化曲线波谷点的距离
        check_by_long_line = check_by_long_line_in_cqt(x, start, end, CQT) # 判断是否位于节拍中部
        # if check_by_long_line is False and len(offset) > 0 and ((np.min(offset) < 15 and np.max(base_pitch[x:x+3]) != 0) or (len(offset2) > 0 and np.min(offset2) < 6)): # 位于音高跳跃点和最高点变化曲线波谷点附近， 且不位于节拍断层区域
        if check_by_long_line is False and len(offset) > 0 and np.min(offset) < 8 and (x+3 < len(base_pitch) and np.max(base_pitch[x+2:x+3]) != 0) and len(offset2) > 0 and np.min(offset2) < 6: # 位于音高跳跃点和最高点变化曲线波谷点附近， 且不位于节拍断层区域
            select_starts.append(x)
    select_starts = [ x for x in select_starts if x > start - 5 and x < end]
    select_starts.sort()

    tmp = []
    tmp.append(int(select_starts[0]))
    # 判断节拍与前一节拍之间是否有波峰和波谷
    for i in range(1,len(select_starts)):
        x = int(select_starts[i])
        y = int(select_starts[i-1])
        offset = [s for s in starts_on_highest_point if s > y and s < x]
        if len(offset) > 0 and x - tmp[-1] > 6:
            tmp.append(x)
    select_starts = tmp
    # code = parse_rhythm_code(rhythm_code)
    # code = [int(x) for x in code]
    # if code[-1] >= 1000:
    #     select_starts = [x for x in select_starts if x <= starts_from_rms_must[-1]]
    # select_starts = add_loss_rms(filename, starts_from_rms_maybe, select_starts)

    return select_starts

def check_by_long_line_in_cqt(x,start,end,cqt):
    if x < start:
        return True
    s = x - 20 if x - 20 > start else start
    e = x + 20 if x + 20 < end else end
    rows_cqt = cqt[10:,s:e]
    h,w = rows_cqt.shape
    if h == 0 or w == 0:
        return True
    cqt_max = np.max(cqt)
    cqt_min = np.min(cqt)
    flag = False
    for i in range(h-1):
        c = rows_cqt[i,:]
        h = rows_cqt[i+1]
        if (np.min(c) == cqt_max and np.max(h) == cqt_min) or (np.max(c) == cqt_min and np.min(h) == cqt_max):
            flag = True
            break
    return flag
def get_highest_point_on_cqt(cqt):
    cqt = signal.medfilt(cqt, (3, 3))  # 二维中值滤波
    h,w = cqt.shape
    cqt_max = np.max(cqt)
    cqt_min = np.min(cqt)
    gaps = np.zeros(w)
    for i in range(5,w-1):
        current_col_cqt = cqt[10:78,i]
        # if cqt_max in current_col_cqt and list(current_col_cqt).index(cqt_max) > 50:
        #     continue
        current_col_cqt = [1 if x == cqt_max else 0 for x in current_col_cqt]
        tmp = [i for i in range(len(current_col_cqt)) if current_col_cqt[i] == 0 and current_col_cqt[i-1] == 1]
        if len(tmp) == 1:
            gaps[i] = tmp[-1]
        elif len(tmp) >1:
            t1 = tmp[-1]
            t2 = tmp[-2]
            s1 = sum(current_col_cqt[t2:t1])
            if len(tmp) >= 3:
                end = tmp[-3]
            else:
                end = 10
            s2 = sum(current_col_cqt[end:t2])
            if s1 > s2 - 3:
                gaps[i] = tmp[-1]
            else:
                gaps[i] = tmp[-2]
    return gaps

def get_starts_on_highest_point_of_cqt(cqt):
    gaps = get_highest_point_on_cqt(cqt)
    b, a = signal.butter(8, 0.25, analog=False)
    gaps = signal.filtfilt(b, a, gaps)
    # starts = [i for i in range(10,len(gaps)-10) if np.max(gaps[i:i+6]) > gaps[i] + 3 and (gaps[i-3] - gaps[i] > 3 and gaps[i+1] > gaps[i] and gaps[i] > 10) or  (gaps[i+5] - gaps[i] > 3 and gaps[i-1] > gaps[i]) or (gaps[i] > 0 and gaps[i-1] == 0) ]
    starts = [i for i in range(10,len(gaps)-10) if gaps[i-1] >= gaps[i] and gaps[i+1] > gaps[i] ]
    return starts

def get_must_starts_on_highest_point_of_cqt(cqt):
    start, end, length = get_start_and_end(cqt)
    starts_on_highest_point = get_starts_on_highest_point_of_cqt(cqt)  # 所有的波谷点
    gaps = get_highest_point_on_cqt(cqt)
    b, a = signal.butter(8, 0.25, analog=False)
    gaps = signal.filtfilt(b, a, gaps)

    base_pitch = get_base_pitch_from_cqt(cqt)  # 获取音高线
    # print("base_pitch is {},size {}".format(base_pitch[316:], len(base_pitch)))
    base_pitch = modify_base_pitch(base_pitch)  # 修正音高线上的倍频问题

    tmp = []
    for x in starts_on_highest_point:
        gaps_tmp = gaps[x:]
        max_points = [i for i in range(2,len(gaps_tmp)-4) if gaps_tmp[i] > gaps_tmp[i+1] and gaps_tmp[i] > gaps_tmp[i-1]] # 波谷点之后的波峰
        if len(max_points) > 0 and gaps[x+max_points[0]] - gaps[x] > 10: # 波峰与波谷的落差大于10
            if len(tmp) == 0:
                tmp.append(x)
            else:
                if x - tmp[-1] < 6: #去掉挤在一起的
                    tmp[-1] = x
                else:
                    tmp.append(x)
    tmp.sort()
    #淘汰的波谷点（处于高处的，即大于10的）
    other_starts = [x for x in starts_on_highest_point if x not in tmp and gaps[x] > 10]
    for x in other_starts:
        # if x == 339:
        #     print(x)
        gaps_tmp = gaps[:x]
        less_starts = [t for t in tmp if t < x]
        if len(less_starts) == 0:
            less_starts.append(start)
        more_starts = [t for t in tmp if t > x]
        if len(more_starts) == 0:
            more_starts.append(end)
        more_starts_gaps = [1 for m in gaps[x:more_starts[0]] if m < 5]
        max_points = [i for i in range(2,len(gaps_tmp)-4) if gaps_tmp[i] >= gaps_tmp[i+1] and gaps_tmp[i] > gaps_tmp[i-1]] # 波谷点之前的波峰
        if len(max_points) > 0 and (len(less_starts) > 0 and max_points[-1] > less_starts[-1]) and gaps[max_points[-1]] - gaps[x] > 13: # 波峰与波谷的落差大于10
            offset = [np.abs(x - t) for t in tmp]
            if np.min(offset)>6 and ((more_starts[0] - x < 40 and len(more_starts_gaps) < 5) or(more_starts[0] - x > 40)) and end - x > 30: #去掉挤在一起的
                tmp.append(x)
                tmp.sort()
    # print("tmp is {}, size {}".format(tmp,len(tmp)))
    tmp = [x for x in tmp if np.max(base_pitch[x:x+10]) != 0]
    return tmp

def find_2000_in_starts_on_highest_point(starts_on_highest_point,rhythm_code,start,end,length):
    code = parse_rhythm_code(rhythm_code)
    code = [int(x) for x in code]
    indexs = [i for i in range(len(code)) if code[i] == 2000]
    positions = []
    ranges = []
    if len(indexs) > 0:
        for i in indexs:
            gap = length/8000*(sum(code[:i]) + 1000)
            position = start + int(gap)
            positions.append(position)
            for i in range(len(starts_on_highest_point)-1):
                if starts_on_highest_point[i] < position and starts_on_highest_point[i+1] > position:
                    ranges.append(starts_on_highest_point[i])
                    ranges.append(starts_on_highest_point[i+1])
            if position > starts_on_highest_point[-1]:
                ranges.append(starts_on_highest_point[-1])
                ranges.append(end)
    ranges.sort()
    return positions,ranges
def check_onset_score(cqt,select_starts,rhythm_code,pitch_code,code_dict):
    start, end, length = get_start_and_end(cqt)
    # print("start ,end is {},{}".format(start,end))
    code = parse_rhythm_code(rhythm_code)
    code = [int(x) for x in code]

    onset_types, all_starts = get_onset_type(select_starts, rhythm_code, end, init_code_dict=code_dict)
    # print("==================================onset_types is {},size {} ========================================".format(onset_types, len(onset_types)))

    print("==========onset_types is {},size {}".format(onset_types, len(onset_types)))
    all_symbols = get_all_symbols(onset_types)
    # print(all_symbols)

    code = parse_rhythm_code(rhythm_code)
    code = [int(x) for x in code]
    # print("code  is {} ,size {}".format(code, len(code)))
    base_symbols = get_all_symbols(code)
    threshold_score = 40


    all_symbols = modify_onset_when_small_change(code, onset_types, base_symbols, all_symbols)

    onset_score, onset_detail,detail_list,raw_positions = calculate_onset_score_from_symbols(base_symbols, all_symbols, threshold_score)
    return onset_score

def add_loss_rms(filename,maybe,select_starts):
    y, sr = librosa.load(filename)
    rms = librosa.feature.rmse(y=y)[0]
    rms = [x / np.std(rms) for x in rms]
    rms = list(np.diff(rms))
    rms.insert(0, 0)

    b, a = signal.butter(8, 0.5, analog=False)
    sig_ff = signal.filtfilt(b, a, rms)
    rms_on_select_starts = [sig_ff[x] for x in select_starts]
    rms_min = np.min(rms_on_select_starts)
    for m in maybe:
        if sig_ff[m] > rms_min*0.8 and m not in select_starts:
            select_starts.append(m)
    return select_starts
def get_total_length(filename):
    y, sr = librosa.load(filename)
    rms = librosa.feature.rmse(y=y)[0]
    CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref=np.max)
    CQT = np.where(CQT > -30, np.max(CQT), np.min(CQT))
    start, end, length = get_start_and_end(CQT)
    return length

def get_must_starts_on_rms(filename, threshold):

    y, sr = librosa.load(filename)
    CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref=np.max)

    CQT = np.where(CQT > -35, np.max(CQT), np.min(CQT))

    CQT = signal.medfilt(CQT, (5, 5))  # 二维中值滤波
    start, end, length = get_start_and_end(CQT)
    min_gap = 10*length/320
    rms = librosa.feature.rmse(y=y)[0]

    rms = [x / np.std(rms) for x in rms]
    rms = list(np.diff(rms))
    rms.insert(0, 0)


    b, a = signal.butter(8, 0.5, analog=False)
    sig_ff = signal.filtfilt(b, a, rms)
    sig_ff = [x / np.std(sig_ff) for x in sig_ff]
    sig_ff = [x if x > 0 else 0 for x in sig_ff]
    # starts = [i for i in range(1,len(sig_ff)-1) if sig_ff[i] > sig_ff[i-1] and sig_ff[i] > sig_ff[i+1] and sig_ff[i] > np.max(sig_ff)* threshold]
    # starts = [i for i in range(1,len(sig_ff)-1) if (sig_ff[i] > sig_ff[i-1] and sig_ff[i] > sig_ff[i+1] and sig_ff[i] > np.max(sig_ff)* threshold) or (sig_ff[i] > sig_ff[i-1] and sig_ff[i] > sig_ff[i+1] and sig_ff[i] < 0 and sig_ff[i] - np.min(sig_ff) > np.max(sig_ff)* threshold)]
    starts = [i for i in range(1, len(sig_ff) - 1) if sig_ff[i] > sig_ff[i - 1] and sig_ff[i] > sig_ff[i + 1] and sig_ff[i] > threshold] #大于阀值的节拍点
    starts = [x for x in starts if x > start - 5 and x < end] #位于起始点和结束点之间的节拍点
    tmp = []
    tmp.append(starts[0])
    for x in starts:
        if x - tmp[-1] > min_gap*0.8 and np.min(sig_ff[tmp[-1]:x]) == 0 :  #去掉挤在一起的节拍点
            tmp.append(x)
    starts = tmp
    return starts
def get_must_starts(filename,threshold):
    # threshold = 1.8
    starts_on_rms = get_must_starts_on_rms(filename, threshold)
    starts = []

    y, sr = librosa.load(filename)
    CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref=np.max)
    CQT = np.where(CQT > -35, np.max(CQT), np.min(CQT))
    CQT = signal.medfilt(CQT, (5, 5))  # 二维中值滤波
    starts_on_highest_point = get_must_starts_on_highest_point_of_cqt(CQT)
    for x in starts_on_rms:
        offset = [np.abs(x - s) for s in starts_on_highest_point]
        if np.min(offset) < 30:
            starts.append(x)
    starts.sort()
    return starts

def get_range_of_250(filename,rhythm_code):
    code = parse_rhythm_code(rhythm_code)
    code = [int(x) for x in code]
    indexs_of_250 = [i for i in range(len(code)) if code[i] == 250]

    y, sr = librosa.load(filename)
    CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref=np.max)
    CQT = np.where(CQT > -35, np.max(CQT), np.min(CQT))
    CQT = signal.medfilt(CQT, (5, 5))  # 二维中值滤波
    start, end, length = get_start_and_end(CQT)

    start_250 = 0
    end_250 = 0
    if len(indexs_of_250) > 1:
        s = indexs_of_250[0]
        e = indexs_of_250[-1]
        start_250 = int(length*sum(code[:s])/8000)
        end_250 = int(length*sum(code[:e])/8000) + 20
    return start_250, end_250

def modify_base_pitch(base_pitch):
    result = base_pitch.copy()
    jump_point = 0
    for i in range(len(base_pitch)-1):
        if i < jump_point:
            continue
        a = base_pitch[i]
        b = base_pitch[i+1]
        # if a == 11 and b == 30:
        #     print(b)
        if a != 0 and b - a > 7: # 后面比前面大
            tmp = base_pitch[i:]
            if len(tmp) > 0:
                for j in range(1,len(tmp)):
                    t = tmp[j]
                    if t - a >= 7:
                        result[i+j] = a
                        jump_point = i +j
                    else:
                        break
        elif b!= 0 and a - b > 7: # 后面比前面小
            tmp = base_pitch[:i]
            for j in range(len(tmp)):
                if np.min(tmp[j:]) - b > 7:
                    result[j:i+1] = b
    return result

def check_continue_down(base_pitch):
    number = 0
    flag = False
    position = 0
    for i in range(len(base_pitch)-1):
        a = base_pitch[i]
        b = base_pitch[i+1]
        if b != 0 and b < a: # 如果小于即下降，则累加1
            number += 1
        elif b > a and number <2: # 如果大于即上升且累加个数小于2，则退出
            break
        if number >= 2:
            flag = True
            position = i
            break
    return flag,position

def check_continue_up(base_pitch):
    number = 0
    flag = False
    position = 0
    for i in range(len(base_pitch)-1):
        a = base_pitch[i]
        b = base_pitch[i+1]
        if b > a: # 如果大于即上升，则累加1
            number += 1
        elif b < a and number <2: # 如果大于即下降且累加个数小于2，则退出
            break
        if number >= 2:
            flag = True
            position = i
            break
    return flag,position

def get_change_points_on_base_pitch(base_pitch):
    window_width = 9
    change_points = [i for i in range(len(base_pitch)-5) if base_pitch[i] == 0 and base_pitch[i+1] > 0]
    for i in range(len(base_pitch)-window_width-1):
        a = base_pitch[i]
        b = base_pitch[i + 1]
        # if i == 103:
        #     print("=====asdf=adsds==")
        if b > a: #上升
            tmp = base_pitch[i:i+window_width]
            continue_up,position = check_continue_up(tmp)
            if continue_up is True and base_pitch[i + int(position) + 5] != 0:
                # if i+ position == 199:
                #     print("a")
                change_points.append(i + int(position))
        if b < a: #下降
            tmp = base_pitch[i:i+window_width]
            continue_down,position = check_continue_down(tmp)
            if continue_down is True and base_pitch[i + int(position) + 5] != 0:
                change_points.append(i + int(position))
                # if i+ position == 199:
                #     print("a")
                # if np.min(base_pitch[i+ int(position) + 10]) != 0: #排除节拍尾部的下降点
                #     change_points.append(i+ int(position))
    change_points.sort()
    return change_points

def modify_change_points_on_base_pitch(change_points):
    select_change_points = []
    window_width = 6
    for i in range(len(change_points)-1):
        if change_points[i+1] - change_points[i] > window_width:
            select_change_points.append(change_points[i])
    select_change_points.append(change_points[-1])
    return select_change_points

def get_starts_by_change_points_on_base_pitch_and_starts_on_rms(filename,base_pitch,start, end,code):
    # 获取音高线上的跳跃点
    change_points = get_change_points_on_base_pitch(base_pitch)
    change_points = modify_change_points_on_base_pitch(change_points)
    if code[-1] == 2000:
        change_points = [x for x in change_points if x > start - 5 and x < end - 20]
    if change_points[0] - start < 7:
        change_points[0] = start

    # 获取振幅上的节拍点
    threshold = 0.15
    starts_from_rms_maybe = get_starts_from_rms_by_threshold(filename,threshold)
    select_starts_from_rms_maybe = []
    for x in starts_from_rms_maybe: # 去掉音高为0的伪节拍
        if np.max(base_pitch[x:x + 5]) != 0:
            select_starts_from_rms_maybe.append(x)

    select_starts = []
    for x in change_points:
        offset = [np.abs(x - s) for s in starts_from_rms_maybe]
        offset_min = np.min(offset)

        # if x == 260:
        #     print("fasdfasdf")
        if offset_min < 15:
            # 如果是位于两者之间，则取振幅大的
            tmp = [x for x in offset if x < offset_min * 1.25]
            if len(tmp) == 2 and starts_from_rms_maybe[offset.index(tmp[0])] < x and starts_from_rms_maybe[offset.index(tmp[1])] > x:
                y, sr = librosa.load(filename)
                rms = librosa.feature.rmse(y=y)[0]
                rms_bak = rms.copy();
                rms = [x / np.std(rms) for x in rms]
                rms = list(np.diff(rms))
                rms.insert(0, 0)
                b, a = signal.butter(8, 0.5, analog=False)
                sig_ff = signal.filtfilt(b, a, rms)
                a = starts_from_rms_maybe[offset.index(tmp[0])]
                b = starts_from_rms_maybe[offset.index(tmp[1])]
                if sig_ff[a] > sig_ff[b] and a not in select_starts:
                    select_starts.append(a)
                if sig_ff[a] < sig_ff[b] and b not in select_starts:
                    select_starts.append(b)
            else:
                index = offset.index(offset_min)
                if starts_from_rms_maybe[index] not in select_starts:
                    select_starts.append(starts_from_rms_maybe[index])
    return select_starts

def get_all_starts_by_steps(filename,rhythm_code):

    y, sr = librosa.load(filename)
    CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref=np.max)
    CQT = signal.medfilt(CQT, (5, 5))  # 二维中值滤波
    CQT = np.where(CQT > -35, np.max(CQT), np.min(CQT))

    start, end, length = get_start_and_end(CQT)
    # print("start ,end is {},{}".format(start,end))
    code = parse_rhythm_code(rhythm_code)
    code = [int(x) for x in code]
    # print("code  is {} ,size {}".format(code, len(code)))

    base_pitch = get_base_pitch_from_cqt(CQT) # 获取音高线
    # print("base_pitch is {},size {}".format(base_pitch[56:65], len(base_pitch)))
    base_pitch = modify_base_pitch(base_pitch)  # 修正音高线上的倍频问题
    # print("base_pitch is {},size {}".format(base_pitch[316:], len(base_pitch)))
    starts = get_starts_by_change_points_on_base_pitch_and_starts_on_rms(filename, base_pitch,start, end,code)  # 结合音高线跳跃点和振幅波峰点，获取节拍点
    starts.sort()

    #去掉静默区域开始20之内的伪节拍
    silence = [i for i in range(20,len(base_pitch)-20) if base_pitch[i] == 0 and base_pitch[i-1] > 0 and np.max(base_pitch[i:i+20]) == 0]
    if len(silence) > 0:
        starts = [x for x in starts if x < silence[0] - 20 or x > silence[0]]

    starts_on_highest_point = get_must_starts_on_highest_point_of_cqt(CQT)  # cqt波络线上的节拍点
    positions_2000, ranges = find_2000_in_starts_on_highest_point(starts_on_highest_point, rhythm_code, start, end,length)

    starts_from_rms_must = get_must_starts(filename, 2) # 高振幅的节拍点
    select_starts = starts_from_rms_must.copy()  #先将高振幅的节拍点认定为真正的节拍,第一步
    miss_starts = [x for x in starts if x not in starts_from_rms_must]
    for m in miss_starts: #第二步，添加可能的节拍点
        pp = [i for i in range(0,len(ranges),2) if ranges[i] < m and ranges[i+1] > m]
        if len(pp) >0:#如果位于2000节拍区域内
            #判断该2000区域是否已有起始点
            start_point = [x for x in select_starts if np.abs(ranges[pp[0]] - x) < 15]
            if len(start_point) == 0: #该2000区域没有起始点
                select_starts.append(m)
        else:
            select_starts.append(m)
    select_starts.sort()
    # print("select_starts  is {} ,size {}".format(select_starts, len(select_starts)))
    # print("select_starts_diff  is {} ,size {}".format(np.diff(select_starts), len(select_starts)-1))
    # 第三步，合并的过密的节拍点（后面会用算法补加最佳的遗漏节拍）
    tmp = starts_from_rms_must.copy()
    for x in select_starts:
        if x not in starts_from_rms_must:
            offset = [np.abs(x - t) for t in tmp]
            offset_min = np.min(offset)
            if offset_min > 18:
                tmp.append(x)
    select_starts = tmp
    select_starts.sort()
    # select_starts = starts_from_rms_must.copy()


    code_dict = get_code_dict_by_min_diff(select_starts, code, start, end)
    onset_types, all_starts = get_onset_type_by_code_dict(select_starts, rhythm_code, end, code_dict)

    return onset_types, all_starts,base_pitch

def get_all_starts_by_optimal(filename, rhythm_code,pitch_code):
    onset_types, all_starts, base_pitch = get_all_starts_by_steps(filename, rhythm_code)
    # print("onset_types  is {} ,size {}".format(onset_types, len(onset_types)))
    y, sr = librosa.load(filename)
    CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref=np.max)
    CQT = signal.medfilt(CQT, (5, 5))  # 二维中值滤波
    CQT = np.where(CQT > -35, np.max(CQT), np.min(CQT))
    starts_on_highest_point = get_must_starts_on_highest_point_of_cqt(CQT)  # cqt波络线上的节拍点

    start, end, length = get_start_and_end(CQT)
    all_symbols = get_all_symbols(onset_types)
    code = parse_rhythm_code(rhythm_code)
    code = [int(x) for x in code]
    base_symbols = get_all_symbols(code)
    # print("base_symbols  is {} ,size {}".format(base_symbols, len(base_symbols)))
    all_symbols = modify_onset_when_small_change(code, onset_types, base_symbols, all_symbols)
    # print("all_symbols  is {} ,size {}".format(all_symbols, len(all_symbols)))
    lcseque, positions,raw_positions = my_find_lcseque(base_symbols, all_symbols)
    loss_positions = [i for i in range(len(code)) if i not in positions]
    # print("loss_positions is {}, size {}".format(loss_positions, len(loss_positions)))
    onset_score, all_starts = check_onset_score_from_starts(all_starts, rhythm_code, end)
    # print("onset_score is {}".format(onset_score))
    change_points = get_change_points_on_base_pitch(base_pitch)
    change_points = modify_change_points_on_base_pitch(change_points)
    if code[-1] == 2000:
        change_points = [x for x in change_points if x > start - 5 and x < end - 20]
    starts_from_rms_must = get_must_starts(filename, 2)
    threshold = 0.15
    starts_from_rms_maybe = get_starts_from_rms_by_threshold(filename, threshold)
    select_starts_from_rms_maybe = []
    for x in starts_from_rms_maybe:  # 去掉音高为0的伪节拍
        if np.max(base_pitch[x:x + 5]) != 0:
            select_starts_from_rms_maybe.append(x)
    starts_from_rms_maybe = select_starts_from_rms_maybe
    starts_from_rms_maybe = [x for x in starts_from_rms_maybe if x not in starts_from_rms_must]
    all_starts = add_best_miss_points(all_starts, base_pitch,change_points, starts_from_rms_maybe, starts_on_highest_point,rhythm_code,pitch_code,start, end, onset_score)
    code_dict = get_code_dict_by_min_diff(all_starts, code, start, end)
    onset_types, all_starts = get_onset_type_by_code_dict(all_starts, rhythm_code, end, code_dict)
    if code[-1] - onset_types[-1] <= 1000:
        onset_types[-1] = code[-1]  # 最后一个节拍，由于人的习惯不会唱全，所以都识别为标准节拍
    # print("finally all_starts is {}, size {}".format(all_starts, len(all_starts)))
    # print("finally onset_types is {}, size {}".format(onset_types, len(onset_types)))
    return onset_types, all_starts, base_pitch

def get_all_starts_by_alexnet(filename, rhythm_code,pitch_code):
    # onset_types, all_starts, base_pitch = get_all_starts_by_steps(filename, rhythm_code)
    # print("onset_types  is {} ,size {}".format(onset_types, len(onset_types)))
    y, sr = librosa.load(filename)
    CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref=np.max)
    CQT = np.where(CQT > -35, np.max(CQT), np.min(CQT))
    CQT = signal.medfilt(CQT, (5, 5))  # 二维中值滤波

    start, end, length = get_start_and_end(CQT)
    base_pitch = get_base_pitch_from_cqt(CQT)  # 获取音高线
    # print("base_pitch is {},size {}".format(base_pitch[56:65], len(base_pitch)))
    base_pitch = modify_base_pitch(base_pitch)  # 修正音高线上的倍频问题
    code = parse_rhythm_code(rhythm_code)
    code = [int(x) for x in code]

    # savepath = 'E:/t/'  # 保存要测试的目录
    # savepath = '/home/lei/bot-rating/split_pic'

    savepath = get_split_pic_save_path()
    change_points = init_data(filename, rhythm_code, savepath)  # 切分潜在的节拍点，并且保存切分的结果
    onset_frames, onset_frames_by_overage = get_starts_by_alexnet(filename, rhythm_code, savepath)
    #如果没包括起始点
    if len(onset_frames_by_overage) > 0 and np.abs(onset_frames_by_overage[0] - start) > 15:
        rms = librosa.feature.rmse(y=y)[0]
        rms = [x / np.std(rms) for x in rms]
        rms = list(np.diff(rms))
        rms.insert(0, 0)
        b, a = signal.butter(8, 0.5, analog=False)
        sig_ff = signal.filtfilt(b, a, rms)
        sig_ff = [x / np.std(sig_ff) for x in sig_ff]
        # print("asfdasdfadsfas {}".format(sig_ff[start:start+3]))
        # 如果起始点的振幅大于0.9才算真正的起始点
        s = start - 4 if start - 4 > 0 else 0
        sig_ff_tmp = sig_ff[s:start+4]
        if np.max(sig_ff_tmp) > 0.9:
            onset_frames_by_overage.append(start)
            onset_frames_by_overage.sort()
    #判断是否包括了必选节拍
    starts_from_rms_must = get_must_starts(filename, 1.75)
    starts_from_rms_must = [ x for x in starts_from_rms_must if x > start-5 and x < end]
    # print("starts_from_rms_must is {}".format(starts_from_rms_must))
    for s in starts_from_rms_must:
        offset = [np.abs(s - o) for o in onset_frames_by_overage]
        if np.min(offset) >= 8:
            onset_frames_by_overage.append(s)
    onset_frames_by_overage.sort()
    all_starts = onset_frames_by_overage
    # code_dict = get_code_dict_by_min_diff(all_starts, code, start, end)
    # onset_types, all_starts = get_onset_type_by_code_dict(all_starts, rhythm_code, end, code_dict)
    # print("0 finally all_starts is {}, size {}".format(all_starts, len(all_starts)))
    # onset_types, all_starts = get_onset_type_for_alexnet(all_starts, rhythm_code, end)
    onset_types, all_starts,speed = get_onset_type_by_opt(all_starts, rhythm_code, end)
    if code[-1] - onset_types[-1] <= 1000:
        onset_types[-1] = code[-1]  # 最后一个节拍，由于人的习惯不会唱全，所以都识别为标准节拍
    # print("finally all_starts is {}, size {}".format(all_starts, len(all_starts)))
    # print("finally all_starts is {}, size {}".format(all_starts, len(all_starts)))
    # print("finally onset_types is {}, size {}".format(onset_types, len(onset_types)))
    if len(code) > len(all_starts) and len(code) - len(all_starts) <= 2: #如果个数小于标准个数2个之内
        for i,a in enumerate(all_starts):
            if onset_types[i] == 500 and code[i] == 250:
                mb = [x for x in change_points if x > a + 5 and x < all_starts[i+1]-5]
                if len(mb) > 0:
                    tmp = a + int((all_starts[i+1] -a)*0.5)
                    all_starts.append(tmp)
                    all_starts.sort()
                break
        if len(all_starts) < len(code):
            for i in range(-1,0-len(all_starts) + 2,-1):
                if onset_types[i] == 500 and code[i] == 250:
                    a = all_starts[i-1]
                    mb = [x for x in change_points if x > a + 5 and x < all_starts[i] - 5]
                    if len(mb) > 0:
                        tmp = a + int((all_starts[i] - a) * 0.5)
                        all_starts.append(tmp)
                        all_starts.sort()
                        break
        onset_types, all_starts, speed = get_onset_type_by_opt(all_starts, rhythm_code, end)
        if code[-1] - onset_types[-1] <= 1000:
            onset_types[-1] = code[-1]  # 最后一个节拍，由于人的习惯不会唱全，所以都识别为标准节拍
    # print("======finally onset_types is {}, size {}".format(onset_types, len(onset_types)))
    return onset_types, all_starts, base_pitch,change_points

def get_code_dict_by_min_diff(select_starts, code, start, end):
    select_starts_diff = np.diff(select_starts)
    min_diff = np.min(select_starts_diff)
    if min_diff < 20:
        rate = min_diff/np.min(code)
    else:
        rate = 20 / 500 * ((end - start) / 320)
    code_dict = get_code_dict(rate, code)
    return code_dict

def get_onset_type_by_code_dict(all_starts,rhythm_code,end,code_dict):
    onset_frames = all_starts.copy()
    onset_frames.append(end)
    if len(onset_frames) == 0:
        return []
    #print("start_index is {},size is {}".format(start_indexs,len(start_indexs)))
    code = parse_rhythm_code(rhythm_code)
    code = [int(x) for x in code]
    # up_250 = code_dict.get(250)*1.25
    # up_500 = code_dict.get(500)*1.25
    # up_750 = code_dict.get(750)*1.25
    # up_1000 = code_dict.get(1000)*1.45
    # up_1500 = code_dict.get(1500)*1.25
    # up_2000 = code_dict.get(2000)*1.25
    # up_2500 = code_dict.get(2500)*1.25
    # up_3000 = code_dict.get(3000) * 1.25

    types = []
    for x in np.diff(onset_frames):
        if 250 in code and  x <= 16:
            best_key = 250
        elif 500 in code and  x <= 30:
            best_key = 500
        elif 750 in code and  x <= 36:
            best_key = 750
        elif 1000 in code and  x <= 60:
            best_key = 1000
        elif 1500 in code and  x <= 70:
            best_key = 1500
        elif 2000 in code and  x <= 100:
            best_key = 2000
        elif 2500 in code and  x <= 140:
            best_key = 2500
        elif 3000 in code and  x <= 160:
            best_key = 3000
        else:
            best_key = 4000

        types.append(best_key)

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
        if types[-1] < code[-1] and code[-1] - types[-1] < 1000:
            types[-1] = code[-1]  # 最后一个节拍，由于人的习惯不会唱全，所以都识别为标准节拍
    return types,all_starts

def get_onset_type_by_code_dict_without_modify(all_starts,rhythm_code,end,code_dict):
    onset_frames = all_starts.copy()
    onset_frames.append(end)
    if len(onset_frames) == 0:
        return []
    #print("start_index is {},size is {}".format(start_indexs,len(start_indexs)))
    code = parse_rhythm_code(rhythm_code)
    code = [int(x) for x in code]
    # up_250 = code_dict.get(250)*1.25
    # up_500 = code_dict.get(500)*1.25
    # up_750 = code_dict.get(750)*1.25
    # up_1000 = code_dict.get(1000)*1.45
    # up_1500 = code_dict.get(1500)*1.25
    # up_2000 = code_dict.get(2000)*1.25
    # up_2500 = code_dict.get(2500)*1.25
    # up_3000 = code_dict.get(3000) * 1.25

    types = []
    for x in np.diff(onset_frames):
        if 250 in code and  x <= 16:
            best_key = 250
        elif 500 in code and  x <= 30:
            best_key = 500
        elif 750 in code and  x <= 36:
            best_key = 750
        elif 1000 in code and  x <= 50:
            best_key = 1000
        elif 1500 in code and  x <= 70:
            best_key = 1500
        elif 2000 in code and  x <= 100:
            best_key = 2000
        elif 2500 in code and  x <= 140:
            best_key = 2500
        elif 3000 in code and  x <= 160:
            best_key = 3000
        else:
            best_key = 4000

        types.append(best_key)
    return types,all_starts

def get_onset_type_for_alexnet(all_starts,rhythm_code,end):
    onset_frames = all_starts.copy()
    onset_frames.append(end)
    if len(onset_frames) == 0:
        return [],[]
    #print("start_index is {},size is {}".format(start_indexs,len(start_indexs)))
    code = parse_rhythm_code(rhythm_code)
    code = [int(x) for x in code]
    onset_types = [0.5,1,2,4,8]
    d = np.diff(onset_frames)

    gap250 = [x for x in d if x < 20]
    gap500 = [x for x in d if x > 20 and x < 35]
    type250 = np.mean(gap250)
    type500 = np.mean(gap500)
    threshold = 0
    if len(gap250) > 0:
        threshold = type250
        onset_types = [1, 2, 4, 8]
    elif len(gap500) > 0:
        threshold = type500
        onset_types = [0.5, 1, 2, 4]
    if threshold == 0:
        return [],[]

    rate = [x / threshold for x in d]

    types = []
    for x in rate:
        offset = [np.abs(x - o) for o in onset_types]
        min_index = offset.index(np.min(offset))
        if threshold == type250:
            type_names = [250, 500, 1000, 2000]
            best_key = type_names[min_index]
        elif threshold == type500:
            type_names = [500, 1000, 2000]
            best_key = type_names[min_index]

        types.append(best_key)
    # print(np.sum(types))
    if np.abs(np.sum(types) - 4000) < np.abs(np.sum(types) - 8000):
        types = [x *2 for x in types]
    elif np.abs(np.sum(types) - 16000) < np.abs(np.sum(types) - 8000):
        types = [int(x *0.5) for x in types]
    return types,all_starts

def get_onset_type_by_opt(onset_frames,onset_code,end):

    if len(onset_frames) == 0:
        return []
    best_total = 1000000
    best_types = []
    best_speed = 0
    code = parse_rhythm_code(onset_code)
    code = [int(x) for x in code]

    for i in range(len(onset_frames)):
        start_index = i
        end_index = i + 4
        if end_index < len(onset_frames) and end_index < len(code):
            types,spend = get_onset_type_by_part(onset_frames, onset_code, end, start_index)
            if np.abs(np.sum(types) - 8000) < best_total:
                best_types = types
                best_total = np.abs(np.sum(types) - 8000)
                best_speed = spend
    return best_types,onset_frames,best_speed

def get_onset_type_by_part(all_starts,onset_code,end,start_index):
    onset_frames = all_starts.copy()
    onset_frames.append(end)
    if len(onset_frames) == 0:
        return []
    #print("start_index is {},size is {}".format(start_indexs,len(start_indexs)))
    code = parse_rhythm_code(onset_code)
    code = [int(x) for x in code]

    end_index = start_index + 4
    #print("code is {},size is {}".format(code, len(code)))
    if len(onset_frames) > 5:
        total_length_no_last = np.sum(code[start_index:end_index])
        # real_total_length_no_last = onset_frames[-1] - onset_frames[0]
        real_total_length_no_last = onset_frames[end_index] - onset_frames[start_index]

    else:
        total_length_no_last = np.sum(code)
        # real_total_length_no_last = onset_frames[-1] - onset_frames[0]
        real_total_length_no_last = end - onset_frames[0]
    rate = real_total_length_no_last/total_length_no_last
    spend = total_length_no_last/real_total_length_no_last
    code_dict = {}
    for x in code:
        code_dict[x] = int(x * rate)


    types = []
    for x in np.diff(onset_frames):
        best_min = 100000
        best_key = 1
        for key in code_dict:
            value = code_dict.get(key)
            gap = np.abs(value - x)/value
            if gap<best_min:
                best_min = gap
                best_key = key
        types.append(best_key)

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

    #print("types is {},size {}".format(types,len(types)))
    return types,spend

def check_onset_score_from_starts(select_starts,rhythm_code,end):
    onset_types, all_starts = get_onset_type_by_code_dict_without_modify(select_starts, rhythm_code, end, None)
    # print("check_onset_score_from_starts onset_types  is {} ,size {},sum {}".format(onset_types, len(onset_types),sum(onset_types)))
    all_symbols = get_all_symbols(onset_types)
    code = parse_rhythm_code(rhythm_code)
    code = [int(x) for x in code]
    base_symbols = get_all_symbols(code)

    # print(all_symbols)
    threshold_score = 40

    # 修正只有一个节拍错误且误差小于500的场景
    # print("all_symbols  is {} ,all_symbols {}".format(all_symbols, len(all_symbols)))
    # all_symbols = modify_onset_when_small_change(code, onset_types, base_symbols, all_symbols)
    # print("all_symbols  is {} ,all_symbols {}".format(all_symbols, len(all_symbols)))
    # print("base_symbols  is {} ,base_symbols {}".format(base_symbols, len(base_symbols)))
    onset_score, onset_detail,detail_list,raw_positions = calculate_onset_score_from_symbols(base_symbols, all_symbols, threshold_score)
    # print("check_onset_score_from_starts onset_score  is {}".format(onset_score))
    return onset_score,all_starts

def get_miss_change_points_on_base_pitch(starts,change_points):
    mark_change_points = []
    for s in starts:
        offset = [np.abs(s - c) for c in change_points]
        offset_min = np.min(offset)
        if offset_min < 14:
            index = offset.index(offset_min)
            mark_change_points.append(change_points[index])
    miss_change_points = [x for x in change_points if x not in mark_change_points]
    return miss_change_points

def get_miss_starts_on_rms(starts,starts_from_rms_maybe):
    miss_starts = [x for x in starts_from_rms_maybe if x not in starts]
    return miss_starts

def get_miss_starts_on_heighest(starts,starts_on_highest_point):
    mark_change_points = []
    for s in starts:
        offset = [np.abs(s - c) for c in starts_on_highest_point]
        offset_min = np.min(offset)
        if offset_min < 14:
            index = offset.index(offset_min)
            mark_change_points.append(starts_on_highest_point[index])
    miss_starts = [x for x in starts_on_highest_point if x not in mark_change_points]
    return miss_starts

def get_all_miss_points(starts,change_points,starts_from_rms_maybe,starts_on_highest_point):
    miss_change_points = get_miss_change_points_on_base_pitch(starts, change_points)
    miss_starts_on_rms = get_miss_starts_on_rms(starts, starts_from_rms_maybe)
    miss_starts_on_heighest = get_miss_starts_on_heighest(starts, starts_on_highest_point)

    all_miss_points = miss_change_points.copy()
    if len(miss_starts_on_rms) > 0:
        for m in miss_starts_on_rms:
            all_miss_points.append(m)

    if len(miss_starts_on_heighest) > 0:
        for m in miss_starts_on_heighest:
            all_miss_points.append(m)

    all_miss_points.sort()
    return all_miss_points

def add_best_miss_points(starts,base_pitch,change_points,starts_from_rms_maybe,starts_on_highest_point,rhythm_code,pitch_code,start, end,init_score):
    starts_from_rms_maybe_tmp = []
    for s in starts_from_rms_maybe: # 只取音高线上有变化的振幅节拍
        a = s - 7 if s - 7 > start else start
        b = s + 7 if s + 7 < end else end
        base_pitch_tmp = base_pitch[a:b]
        base_pitch_tmp_diff = np.diff(base_pitch_tmp)
        base_pitch_tmp_diff = [x for x in base_pitch_tmp_diff if x != 0]
        if len(base_pitch_tmp_diff) > 0:
            starts_from_rms_maybe_tmp.append(s)
    starts_from_rms_maybe = starts_from_rms_maybe_tmp

    all_miss_points = get_all_miss_points(starts.copy(), change_points, starts_from_rms_maybe, starts_on_highest_point)
    code = parse_rhythm_code(rhythm_code)
    code = [int(x) for x in code]
    if code[0] == 2000:
        all_miss_points = [x for x in all_miss_points if x > start + 30 and x < end]
    elif code[0] == 1000:
        all_miss_points = [x for x in all_miss_points if x > start + 20 and x < end]
    else:
        all_miss_points = [ x for x in all_miss_points if x > start and x < end]
    tmp = []
    tmp.append(all_miss_points[0])
    for a in all_miss_points:
        if a - tmp[-1] > 4:
            tmp.append(a)
    all_miss_points = tmp
    length = end - start
    positions_2000, ranges = find_2000_in_starts_on_highest_point(starts_on_highest_point, rhythm_code, start, end,length)
    tmp = []
    for i in range(0,len(ranges),2):
        a = ranges[i]
        b = ranges[i+1]
        for x in all_miss_points:
            if x > a and x < b:
                tmp.append(x)
    all_miss_points = [ x for x in all_miss_points if x not in tmp]

    best_score = init_score
    best_starts = starts
    best_number = 0
    tests = []
    if len(starts) < len(rhythm_code):
        number = len(code) - len(starts)
        combinations = get_combinations(all_miss_points, number)
        if len(combinations) > 0:
            for cb in combinations:
                for n in range(len(cb)):
                    test_starts = starts.copy()
                    # print("cb is {}".format(cb[n]))
                    # if cb[n][0] == 64 and cb[n][1] == 94 and cb[n][2] == 122:
                    #     print("test")
                    if len(cb[n]) > 1 and np.min(np.diff(cb[n])) <= 10:
                        continue
                    for i in range(number):
                        test_starts.append(cb[n][i])
                    test_starts.sort()
                    onset_score,all_starts = check_onset_score_from_starts(test_starts, rhythm_code, end)
                    # print("add_best_miss_points onset_score is {}".format(onset_score))
                    if onset_score > best_score and len(all_starts) == len(code):
                        best_score = onset_score
                        best_starts = all_starts
                        tests.append(best_starts)
                    elif onset_score == best_score and len(all_starts) == len(code):
                        tests.append(best_starts)
    if len(tests) > 1:
        best_starts = check_total_score(tests,base_pitch, rhythm_code, pitch_code, start, end)
    elif len(tests) == 1:
        best_starts = tests[0]
    else:
        best_starts = starts
    # print("best_starts is {}, size {}".format(best_starts,len(best_starts)))
    # print("add_best_miss_points best_score is {}".format(best_score))
    return best_starts

def get_all_myabe_starts(filename,rhythm_code):
    y, sr = librosa.load(filename)
    # print("time is {}".format(time))
    CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref=np.max)
    cqt_bak = CQT.copy()

    CQT = signal.medfilt(CQT, (5, 5))  # 二维中值滤波
    CQT = np.where(CQT > -35, np.max(CQT), np.min(CQT))
    start, end, length = get_start_and_end(CQT)
    code = parse_rhythm_code(rhythm_code)
    code = [int(x) for x in code]


    base_pitch = get_base_pitch_from_cqt(CQT)
    # print("base_pitch is {},size {}".format(base_pitch[316:], len(base_pitch)))
    base_pitch = modify_base_pitch(base_pitch)  # 修正倍频问题
    # print("base_pitch is {},size {}".format(base_pitch[316:],len(base_pitch)))
    # base_pitch = filter_cqt(cqt_bak)
    base_pitch_bak = base_pitch.copy()

    # base_pitch = signal.medfilt(base_pitch, 11)  # 二维中值滤波

    change_points = get_change_points_on_base_pitch(base_pitch)
    change_points = modify_change_points_on_base_pitch(change_points)
    if code[-1] == 2000:
        change_points = [x for x in change_points if x > start - 5 and x < end - 20]

    threshold = 0.15
    starts_from_rms_maybe = get_starts_from_rms_by_threshold(filename, threshold)
    select_starts_from_rms_maybe = []
    for x in starts_from_rms_maybe:  # 去掉音高为0的伪节拍
        if np.max(base_pitch[x:x + 5]) != 0:
            select_starts_from_rms_maybe.append(x)
    starts_from_rms_maybe = select_starts_from_rms_maybe
    starts_from_rms_maybe_tmp = []
    for s in starts_from_rms_maybe:  # 只取音高线上有变化的振幅节拍
        a = s - 7 if s - 7 > start else start
        b = s + 7 if s + 7 < end else end
        base_pitch_tmp = base_pitch[a:b]
        base_pitch_tmp_diff = np.diff(base_pitch_tmp)
        base_pitch_tmp_diff = [x for x in base_pitch_tmp_diff if x != 0]
        if len(base_pitch_tmp_diff) > 0:
            starts_from_rms_maybe_tmp.append(s)
    starts_from_rms_maybe = starts_from_rms_maybe_tmp

    starts_on_highest_point = get_must_starts_on_highest_point_of_cqt(CQT)

    return change_points,starts_from_rms_maybe,starts_on_highest_point


def check_total_score(tests,base_pitch, rhythm_code, pitch_code, start, end):
    best_score = 0
    best_starts = None
    for all_starts in tests:
        total_score = check_total_score_from_starts_and_base_pitch(all_starts, base_pitch, rhythm_code, pitch_code, start, end)
        if total_score > best_score:
            best_score = total_score
            best_starts = all_starts
    return best_starts

def get_number_cb_in_starts_on_highest_point(cb,starts_on_highest_point):
    number = 0
    for c in cb:
        if c in starts_on_highest_point:
            number += 1
    return number

'''
数组元素组合
https://blog.csdn.net/suibianshen2012/article/details/80772905
'''
def get_combinations(list1, number):
    import itertools
    list2 = []
    for i in range(1, len(list1) + 1):
        iter = itertools.combinations(list1, i)
        tmp = list(iter)
        # print(len(tmp))
        # print(tmp[0])
        # print(len(tmp[0]))
        if len(tmp[0]) == number:
            list2.append(tmp)
    # print(list2)
    return list2

def clear_dir(dis_dir):
    # shutil.rmtree(dis_dir)
    # os.mkdir(dis_dir)
    delList = os.listdir(dis_dir)
    for f in delList:
        filePath = os.path.join(dis_dir, f)
        if os.path.isfile(filePath):
            os.remove(filePath)
            print
            filePath + " was removed!"


def load_and_trim(path):
    audio, sr = librosa.load(path)
    energy = librosa.feature.rmse(audio)
    frames = np.nonzero(energy >= np.max(energy) / 5)
    indices = librosa.core.frames_to_samples(frames)[1]
    audio = audio[indices[0]:indices[-1]] if indices.size else audio[0:0]

    return audio, sr


def load_and_trim_v2(path,offset,duration):
    audio, sr = librosa.load(path, offset=offset, duration=duration)
    energy = librosa.feature.rmse(audio)
    frames = np.nonzero(energy >= np.max(energy) / 5)
    indices = librosa.core.frames_to_samples(frames)[1]
    audio = audio[indices[0]:indices[-1]] if indices.size else audio[0:0]

    return audio, sr

def cqt_split_and_save(filename,onset_frames,savepath):
    y, sr = librosa.load(filename)
    rms = librosa.feature.rmse(y=y)[0]
    time = librosa.get_duration(filename=filename)
    total_frames_number = len(rms)
    # print("time is {}".format(time))
    CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref=np.max)
    w, h = CQT.shape
    for i in range(0, len(onset_frames)):
        half = 30
        start = onset_frames[i] - half
        if start < 0:
            start = 0
        end = onset_frames[i] + half
        if end >= total_frames_number:
            end = total_frames_number - 1
        # y2 = [x if i> start and i<end else 0 for i,x in enumerate(y)]
        CQT_sub = np.zeros(CQT.shape)
        middle = int(h / 2)
        offset = middle - onset_frames[i]
        for j in range(int(start), int(end)):
            CQT_sub[:, j + offset] = CQT[:, j]
        # CQT = CQT_T
        librosa.display.specshow(CQT_sub, y_axis='cqt_note', x_axis='time')
        # y2 = [x for i,x in enumerate(y) if i> start and i<end]
        # y2 = [0.03 if i> start and i<end else 0.02 for i,x in enumerate(y)]
        # y2[int(len(y2) / 2)] = np.max(y)  # 让图片展示归一化
        t = librosa.frames_to_time([middle], sr=sr)
        plt.vlines(t, 0, sr, color='y', linestyle='--')  # 标出节拍位置
        # y2 = np.array(y2)
        # print("len(y2) is {}".format(len(y2)))

        # print("(end - start)*sr is {}".format((end - start) * sr))
        # plt.show()
        # plt.subplot(len(onset_times),1,i+1)
        # y, sr = librosa.load(filename, offset=2.0, duration=3.0)
        # librosa.display.waveplot(y2, sr=sr)
        fig = matplotlib.pyplot.gcf()
        # fig.set_size_inches(4, 4)
        if "." in filename:
            Filename = filename.split(".")[0]
        plt.axis('off')
        plt.axes().get_xaxis().set_visible(False)
        plt.axes().get_yaxis().set_visible(False)
        # plt.savefig(savepath + str(i + 1) + '.jpg', bbox_inches='tight', pad_inches=0)
        plt.savefig(savepath + str(onset_frames[i]) + '.jpg', bbox_inches='tight', pad_inches=0)
        plt.clf()

def init_data(filename, rhythm_code,savepath):
    clear_dir(savepath)
    change_points, starts_from_rms_maybe, starts_on_highest_point = get_all_myabe_starts(filename, rhythm_code)
    tmp = []
    if len(change_points) > 0:
        for c in change_points:
            tmp.append(c)
    change_points = tmp.copy()
    if len(starts_from_rms_maybe) > 0:
        for s in starts_from_rms_maybe:
            change_points.append(s - 2)
            change_points.append(s)
            change_points.append(s + 2)
    if len(starts_on_highest_point) > 0:
        for s in starts_on_highest_point:
            change_points.append(s)
            change_points.append(s+3)
    cqt_split_and_save(filename, change_points, savepath)
    return change_points

def get_base_pitch_by_cqt_and_starts(filename,starts):
    y, sr = librosa.load(filename)
    CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref=np.max)
    h,w = CQT.shape

    base_pitchs = []
    if len(starts) > 0 :
        for i,x in enumerate(starts):
            if i < len(starts)-1:
                s = x
                e = starts[i+1]
            else:
                s = x
                e = w - 5
            cols_cqt = CQT[:,s:e]
            h1,w1 = cols_cqt.shape
            pitchs = np.zeros(h)
            for n in range(h1):
                row_sum = sum(cols_cqt[n,])
                pitchs[n] = row_sum
            # print("pitchs is {}, size {}".format(pitchs,len(pitchs)))
            base_pitchs.append(pitchs)
    base_pitch = []
    if len(base_pitchs) > 0:
        for b in base_pitchs:
            index = list(b).index(np.max(b))
            base_pitch.append(index)
    return base_pitch,base_pitchs

def draw_plt(filename,rhythm_code, pitch_code):
    y, sr = librosa.load(filename)
    rms = librosa.feature.rmse(y=y)[0]
    rms = [x / np.std(rms) for x in rms]
    time = librosa.get_duration(filename=filename)
    # print("time is {}".format(time))
    CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref=np.max)
    cqt_bak = CQT.copy()
    w, h = CQT.shape
    # print("w.h is {},{}".format(w,h))
    # onsets_frames = get_real_onsets_frames_rhythm(y)

    CQT = np.where(CQT > -30, np.max(CQT), np.min(CQT))
    start, end, length = get_start_and_end(CQT)
    plt.subplot(3, 1, 1)
    plt.title(filename)
    plt.xlabel("识别结果示意图")
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    librosa.display.specshow(cqt_bak, x_axis='time')



    plt.subplot(3, 1, 2)
    y, sr = librosa.load(filename)
    rms = librosa.feature.rmse(y=y)[0]
    rms_bak = rms.copy();
    rms = [x / np.std(rms) for x in rms]
    rms = list(np.diff(rms))
    rms.insert(0, 0)
    b, a = signal.butter(8, 0.5, analog=False)
    sig_ff = signal.filtfilt(b, a, rms)

    # Savitzky-Golay filter 平滑
    # from scipy.signal import savgol_filter
    # sig_ff = savgol_filter(rms, 5, 1)  # window size 51, polynomial order 3
    # sig_ff = signal.medfilt(rms, 5)  # 二维中值滤波
    sig_ff = [x / np.std(sig_ff) for x in sig_ff]
    sig_ff = [x if x > 0 else 0 for x in sig_ff]
    # rms = signal.medfilt(rms,3)
    times = librosa.frames_to_time(np.arange(len(rms)))
    plt.plot(times, sig_ff)
    # plt.plot(times, rms)
    plt.xlim(0, np.max(times))

    first_type = pitch_code[1]
    base_pitch = get_base_pitch_from_cqt(CQT)
    all_note_types, all_note_type_position = get_all_note_type(base_pitch, first_type)
    all_note_type_position = check_note_type_position(CQT, base_pitch, all_note_type_position)

    threshold = 0.4
    starts_from_rms_must = get_starts_from_rms_by_threshold(filename, threshold)
    starts_on_highest_point = get_starts_on_highest_point_of_cqt(CQT)  # 最高点变化曲线的波谷点
    select_starts_from_rms_must = []
    for x in starts_from_rms_must:  # 去掉没有音高跳跃点的伪节拍
        offset = [np.abs(x - a) for a in all_note_type_position if a < x + 6]
        offset2 = [np.abs(x - a) for a in starts_on_highest_point if a < x + 2]  # 与最高点变化曲线波谷点的距离
        if ((len(offset) > 0 and np.min(offset) < 12) or (len(offset2) > 0 and np.min(offset2) < 16)) and np.max(
                base_pitch[x + 2:x + 3]) != 0 and (
                np.max(base_pitch[x:x + 5]) != 0 or np.min(base_pitch[x - 5:x]) == 0):
            select_starts_from_rms_must.append(x)
    starts_from_rms_must = select_starts_from_rms_must
    if starts_from_rms_must[0] - start > 15:  # 如果没包括开始点，则需要添加开始点
        starts_from_rms_must.append(start)
    starts_from_rms_must.sort()
    starts_from_rms_must_time = librosa.frames_to_time(starts_from_rms_must)
    plt.vlines(starts_from_rms_must_time, 0, np.max(sig_ff), color='b', linestyle='dashed')
    threshold = 0.15
    starts_from_rms_maybe = get_starts_from_rms_by_threshold(filename, threshold)
    select_starts_from_rms_maybe = []
    for x in starts_from_rms_maybe:  # 去掉音高为0的伪节拍
        offset = [np.abs(x - a) for a in all_note_type_position if a < x + 2]
        if np.max(base_pitch[x:x + 5]) != 0:
            select_starts_from_rms_maybe.append(x)
    starts_from_rms_maybe = select_starts_from_rms_maybe
    starts_from_rms_maybe = [x for x in starts_from_rms_maybe if x not in starts_from_rms_must]
    starts_from_rms_maybe_time = librosa.frames_to_time(starts_from_rms_maybe)
    plt.vlines(starts_from_rms_maybe_time, 0, np.max(sig_ff) / 2, color='r', linestyle='dashed')


    plt.subplot(3, 1, 3)
    librosa.display.specshow(CQT, x_axis='time')
    plt.ylim(0, 84)
    total_score, all_starts, detail = calcalate_total_score(filename, rhythm_code, pitch_code)
    print("总分 is {}".format(total_score))
    print("detail is {}".format(detail))
    all_mins_time = librosa.frames_to_time(all_starts)
    plt.vlines(all_mins_time, 0, 75, color='r', linestyle='dashed')


    return plt,total_score

def draw_detail(filename,rhythm_code,pitch_code):
    y, sr = load_and_trim(filename)
    y, sr = librosa.load(filename)
    rms = librosa.feature.rmse(y=y)[0]
    rms = [x / np.std(rms) for x in rms]
    time = librosa.get_duration(filename=filename)
    # print("time is {}".format(time))
    CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref=np.max)
    cqt_bak = CQT.copy()
    w, h = CQT.shape
    # print("w.h is {},{}".format(w,h))
    # onsets_frames = get_real_onsets_frames_rhythm(y)

    CQT = signal.medfilt(CQT, (5, 5))  # 二维中值滤波
    CQT = np.where(CQT > -35, np.max(CQT), np.min(CQT))
    start, end, length = get_start_and_end(CQT)
    code = parse_rhythm_code(rhythm_code)
    code = [int(x) for x in code]

    plt.subplot(6, 1, 1)
    plt.title(filename)
    plt.xlabel("识别结果示意图")
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    librosa.display.specshow(cqt_bak, x_axis='time')

    plt.subplot(6, 1, 2)
    base_pitch = get_base_pitch_from_cqt(CQT)
    # print("base_pitch is {},size {}".format(base_pitch[316:], len(base_pitch)))
    base_pitch = modify_base_pitch(base_pitch) #修正倍频问题
    # print("base_pitch is {},size {}".format(base_pitch[316:],len(base_pitch)))
    # base_pitch = filter_cqt(cqt_bak)
    base_pitch_bak = base_pitch.copy()

    # base_pitch = signal.medfilt(base_pitch, 11)  # 二维中值滤波
    t = librosa.frames_to_time(np.arange(len(base_pitch)))
    plt.plot(t, base_pitch)
    plt.xlim(0, np.max(t))
    plt.ylim(0, 84)
    # print("all_note_types is {}".format(all_note_types))
    # print("1 all_note_type_position is {} ,size {}".format(all_note_type_position, len(all_note_type_position)))
    # all_note_type_position_time = librosa.frames_to_time(all_note_type_position)
    # plt.vlines(all_note_type_position_time, 0, 84, color='r', linestyle='dashed')
    # change_points = get_change_point_on_pitch(CQT, first_type)
    change_points = get_change_points_on_base_pitch(base_pitch)
    change_points = modify_change_points_on_base_pitch(change_points)
    if code[-1] == 2000:
        change_points = [x for x in change_points if x > start - 5 and x < end - 20]
    # print("change_points is {}, size {}".format(change_points, len(change_points)))
    change_points_time = librosa.frames_to_time(change_points)
    plt.vlines(change_points_time, 0, 40, color='b', linestyle='dashed')

    starts = get_starts_by_change_points_on_base_pitch_and_starts_on_rms(filename, base_pitch,start, end,code)
    starts_time = librosa.frames_to_time(starts)
    plt.vlines(starts_time, 0, 25, color='r', linestyle='dashed')

    start, end, length = get_start_and_end(CQT)
    start_time = librosa.frames_to_time(start)
    end_time = librosa.frames_to_time(end)
    plt.vlines(start_time, 0, 40, color='black', linestyle='solid')
    plt.vlines(end_time, 0, 40, color='black', linestyle='solid')



    plt.subplot(6, 1, 3)
    y, sr = librosa.load(filename)
    rms = librosa.feature.rmse(y=y)[0]
    rms_bak = rms.copy();
    rms = [x / np.std(rms) for x in rms]
    rms = list(np.diff(rms))
    rms.insert(0, 0)

    b, a = signal.butter(8, 0.5, analog=False)
    sig_ff = signal.filtfilt(b, a, rms)

    # Savitzky-Golay filter 平滑
    # from scipy.signal import savgol_filter
    # sig_ff = savgol_filter(rms, 5, 1)  # window size 51, polynomial order 3
    # sig_ff = signal.medfilt(rms, 5)  # 二维中值滤波
    sig_ff = [x / np.std(sig_ff) for x in sig_ff]
    # sig_ff = [x if x > 0 else x - np.min(sig_ff) for x in sig_ff]
    # rms = signal.medfilt(rms,3)
    times = librosa.frames_to_time(np.arange(len(rms)))
    plt.plot(times, sig_ff)
    plt.plot(times, rms)

    starts_from_rms_must = get_must_starts(filename, 2)
    starts_from_rms_must_time = librosa.frames_to_time(starts_from_rms_must)
    plt.vlines(starts_from_rms_must_time, 0, np.max(sig_ff), color='b', linestyle='dashed')
    threshold = 0.15
    starts_from_rms_maybe = get_starts_from_rms_by_threshold(filename, threshold)
    select_starts_from_rms_maybe = []
    for x in starts_from_rms_maybe:  # 去掉音高为0的伪节拍
        if np.max(base_pitch[x:x + 5]) != 0:
            select_starts_from_rms_maybe.append(x)
    starts_from_rms_maybe = select_starts_from_rms_maybe
    starts_from_rms_maybe = [x for x in starts_from_rms_maybe if x not in starts_from_rms_must]
    starts_from_rms_maybe_tmp = []
    for s in starts_from_rms_maybe:  # 只取音高线上有变化的振幅节拍
        a = s - 7 if s - 7 > start else start
        b = s + 7 if s + 7 < end else end
        base_pitch_tmp = base_pitch[a:b]
        base_pitch_tmp_diff = np.diff(base_pitch_tmp)
        base_pitch_tmp_diff = [x for x in base_pitch_tmp_diff if x != 0]
        if len(base_pitch_tmp_diff) > 0:
            starts_from_rms_maybe_tmp.append(s)
    starts_from_rms_maybe = starts_from_rms_maybe_tmp
    starts_from_rms_maybe_time = librosa.frames_to_time(starts_from_rms_maybe)
    plt.vlines(starts_from_rms_maybe_time, 0, np.max(sig_ff) / 2, color='r', linestyle='dashed')
    plt.xlim(0, np.max(t))




    plt.subplot(6, 1, 4)
    # c = np.where(cqt_bak > -15, np.max(CQT), np.min(CQT))
    # librosa.display.specshow(c, x_axis='time')
    librosa.display.specshow(CQT, x_axis='time')
    plt.ylim(0, 84)

    plt.subplot(6, 1, 5)
    gaps = get_highest_point_on_cqt(CQT)
    b, a = signal.butter(20, 0.25, analog=False)
    gaps = signal.filtfilt(b, a, gaps)
    t = librosa.frames_to_time(np.arange(len(gaps)))
    plt.plot(t, gaps)

    starts_on_highest_point = get_must_starts_on_highest_point_of_cqt(CQT)
    starts_on_highest_point_time = librosa.frames_to_time(starts_on_highest_point)
    plt.vlines(starts_on_highest_point_time, 0, np.max(gaps) / 4, color='b', linestyle='dashed')
    plt.xlim(0, np.max(t))
    positions_2000, ranges = find_2000_in_starts_on_highest_point(starts_on_highest_point, rhythm_code, start, end, length)
    positions_2000_time = librosa.frames_to_time(positions_2000)
    plt.vlines(positions_2000_time, 0, np.max(gaps) *0.75, color='r', linestyle='dashed')


    plt.subplot(6, 1, 6)
    onset_types, all_starts,base_pitch = get_all_starts_by_steps(filename, rhythm_code)
    # all_starts.append(113)
    # all_starts.append(320)
    all_starts.sort()
    all_starts_time = librosa.frames_to_time(all_starts)
    plt.vlines(all_starts_time, 0, 0.5, color='b', linestyle='solid')

    onset_score,all_starts = check_onset_score_from_starts(all_starts, rhythm_code, end)
    # print("onset_score is {}".format(onset_score))
    all_starts = add_best_miss_points(all_starts,base_pitch, change_points, starts_from_rms_maybe, starts_on_highest_point, rhythm_code,pitch_code,start, end,onset_score)
    all_starts_time = librosa.frames_to_time(all_starts)
    plt.vlines(all_starts_time, 0.5, 2, color='r', linestyle='dashed')
    plt.xlim(0, np.max(t))
    return plt

def draw_by_alexnet(filename,rhythm_code,pitch_code):
    # savepath = 'E:/t/'  # 保存要测试的目录
    savepath = get_split_pic_save_path()
    init_data(filename, rhythm_code, savepath)  # 切分潜在的节拍点，并且保存切分的结果
    onset_frames, onset_frames_by_overage = get_starts_by_alexnet(filename, rhythm_code, savepath)
    # 如果没包括起始点
    y, sr = librosa.load(filename)
    CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref=np.max)
    CQT = signal.medfilt(CQT, (5, 5))  # 二维中值滤波
    CQT = np.where(CQT > -35, np.max(CQT), np.min(CQT))

    start, end, length = get_start_and_end(CQT)
    if len(onset_frames_by_overage) > 0 and np.abs(onset_frames_by_overage[0] - start) > 15:
        rms = librosa.feature.rmse(y=y)[0]
        rms = [x / np.std(rms) for x in rms]
        rms = list(np.diff(rms))
        rms.insert(0, 0)
        b, a = signal.butter(8, 0.5, analog=False)
        sig_ff = signal.filtfilt(b, a, rms)
        sig_ff = [x / np.std(sig_ff) for x in sig_ff]
        # print("asfdasdfadsfas {}".format(sig_ff[start:start+3]))
        # 如果起始点的振幅大于0.9才算真正的起始点
        s = start - 4 if start - 4 > 0 else 0
        sig_ff_tmp = sig_ff[s:start + 4]
        if np.max(sig_ff_tmp) > 0.9:
            onset_frames_by_overage.append(start)
            onset_frames_by_overage.sort()

    # 判断是否包括了必选节拍
    starts_from_rms_must = get_must_starts(filename, 1.75)
    starts_from_rms_must = [x for x in starts_from_rms_must if x > start - 5 and x < end]
    for s in starts_from_rms_must:
        offset = [np.abs(s - o) for o in onset_frames_by_overage]
        if np.min(offset) >= 8:
            onset_frames_by_overage.append(s)
    onset_frames_by_overage.sort()

    all_starts = onset_frames_by_overage.copy()
    code = parse_rhythm_code(rhythm_code)
    code = [int(x) for x in code]
    onset_types, all_starts, speed = get_onset_type_by_opt(all_starts, rhythm_code, end)
    if code[-1] - onset_types[-1] <= 1000:
        onset_types[-1] = code[-1]  # 最后一个节拍，由于人的习惯不会唱全，所以都识别为标准节拍
    if len(code) > len(all_starts) and len(code) - len(all_starts) <= 2:  # 如果个数小于标准个数2个之内
        for i, a in enumerate(all_starts):
            if onset_types[i] == 500 and code[i] == 250:
                tmp = a + int((all_starts[i + 1] - a) * 0.5)
                all_starts.append(tmp)
                all_starts.sort()
                break
        if len(all_starts) < len(code):
            for i in range(-1, 0 - len(all_starts) + 2, -1):
                if onset_types[i] == 500 and code[i] == 250:
                    a = all_starts[i - 1]
                    tmp = a + int((all_starts[i] - a) * 0.5)
                    all_starts.append(tmp)
                    all_starts.sort()
                    break
    onset_frames = all_starts
    y, sr = librosa.load(filename)
    CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref=np.max)
    plt.subplot(3, 1, 1)
    plt.title(filename)
    plt.xlabel("识别结果示意图")
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    librosa.display.specshow(CQT, x_axis='time')
    w, h = CQT.shape

    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    plt.vlines(onset_times, 0, sr, color='y', linestyle='--')

    plt.subplot(3, 1, 2)
    # print(onset_samples)
    # plt.subplot(len(onset_times),1,1)
    y, sr = librosa.load(filename)
    rms = librosa.feature.rmse(y=y)[0]
    rms_bak = rms.copy();
    rms = [x / np.std(rms) for x in rms]
    rms = list(np.diff(rms))
    rms.insert(0, 0)

    b, a = signal.butter(8, 0.5, analog=False)
    sig_ff = signal.filtfilt(b, a, rms)
    sig_ff = [x / np.std(sig_ff) for x in sig_ff]
    # sig_ff = [x if x > 0 else x - np.min(sig_ff) for x in sig_ff]
    # rms = signal.medfilt(rms,3)
    times = librosa.frames_to_time(np.arange(len(rms)))
    plt.plot(times, sig_ff)
    plt.plot(times, rms)
    plt.xlim(0, np.max(times))
    onset_frames_by_overage_times = librosa.frames_to_time(onset_frames_by_overage, sr=sr)
    plt.vlines(onset_frames_by_overage_times, 0, np.max(sig_ff), color='r', linestyle='--')

    plt.subplot(3, 1, 3)
    CQT = signal.medfilt(CQT, (5, 5))  # 二维中值滤波
    CQT = np.where(CQT > -35, np.max(CQT), np.min(CQT))
    base_pitch = get_base_pitch_from_cqt(CQT)
    # print("base_pitch is {},size {}".format(base_pitch[316:], len(base_pitch)))
    base_pitch = modify_base_pitch(base_pitch)  # 修正倍频问题

    # base_pitch = signal.medfilt(base_pitch, 11)  # 二维中值滤波
    t = librosa.frames_to_time(np.arange(len(base_pitch)))
    plt.plot(t, base_pitch)
    plt.xlim(0, np.max(t))
    plt.ylim(0, 84)
    change_points_time = librosa.frames_to_time(onset_frames)
    plt.vlines(change_points_time, 0, 40, color='b', linestyle='dashed')

    return plt