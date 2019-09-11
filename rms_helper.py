# -*- coding: UTF-8 -*-
import librosa.display
import scipy.signal as signal

from LscHelper import *
from base_helper import *
from create_base import *
from myDtw import *


# 1. Get the file path to the included audio example
# Sonify detected beat events
# 定义加载语音文件并去掉两端静音的函数

def write_txt(content, filename, mode='w'):
    """保存txt数据
    :param content:需要保存的数据,type->list
    :param filename:文件名
    :param mode:读写模式:'w' or 'a'
    :return: void
    """
    with open(filename, mode) as f:
        f.write(content)
def get_rms_max_indexs_for_onset(filename,onset_code,threshold = 0.45):
    y, sr = librosa.load(filename)
    rms = librosa.feature.rmse(y=y)[0]
    rms = [x / np.std(rms) for x in rms]
    rms_bak = rms.copy();
    rms = list(np.diff(rms))
    rms.insert(0,0)

    b, a = signal.butter(8, 0.35, analog=False)
    sig_ff = signal.filtfilt(b, a, rms)

    # Savitzky-Golay filter 平滑
    # from scipy.signal import savgol_filter
    # sig_ff = savgol_filter(rms, 5, 1)  # window size 51, polynomial order 3
    # sig_ff = signal.medfilt(rms, 5)  # 二维中值滤波
    sig_ff = [x/np.std(sig_ff) for x in sig_ff]
    max_indexs = [i for i in range(1,len(sig_ff)-1) if sig_ff[i]>sig_ff[i-1] and sig_ff[i]>sig_ff[i+1] and sig_ff[i] > np.max(sig_ff)*0.15]
    sig_ff_on_max_indexs = [sig_ff[x] for x in max_indexs]
    topN_indexs = find_n_largest(a, 4)
    top_index = sig_ff_on_max_indexs.index(np.max(sig_ff_on_max_indexs))
    hline = np.mean([sig_ff_on_max_indexs[i] for i in range(len(sig_ff_on_max_indexs)) if i in topN_indexs and i != top_index]) * threshold
    # print("1 hline is {}".format(hline))
    max_indexs = [i for i in range(1,len(sig_ff)-1) if sig_ff[i]>sig_ff[i-1] and sig_ff[i]>sig_ff[i+1] and sig_ff[i] > hline]
    sig_ff_on_max_indexs = [sig_ff[x] for x in max_indexs]
    # print("sig_ff_on_max_indexs is {}, size {}".format(sig_ff_on_max_indexs,len(sig_ff_on_max_indexs)))
    # print("-1 max_indexs is {},size is {}".format(max_indexs, len(max_indexs)))

    code = parse_onset_code(onset_code)
    code = [int(x) for x in code]
    if code[-1] >= 2000:
        width_last = len(rms) * code[-1] / np.sum(code)
        max_indexs = [x for x in max_indexs if x < len(rms) - int(width_last * 0.4)]
    else:
        max_indexs = [x for x in max_indexs if x < len(rms) - 5]
    # sig_ff_on_max_indexs = [sig_ff[x] for x in max_indexs]
    # tmp = sig_ff_on_max_indexs.copy()
    # tmp.sort()
    # min_index = sig_ff_on_max_indexs.index(tmp[0])
    # second_index = sig_ff_on_max_indexs.index(tmp[1])
    # if sig_ff_on_max_indexs[second_index] - sig_ff_on_max_indexs[min_index] > np.std(sig_ff_on_max_indexs)*0.8:
    #     max_indexs.remove(max_indexs[min_index])
    return rms_bak,rms,sig_ff,max_indexs

def get_rms_max_indexs_for_find_loss(filename,onset_code,hline= 0.2):
    y, sr = librosa.load(filename)
    rms = librosa.feature.rmse(y=y)[0]
    rms = [x / np.std(rms) for x in rms]
    rms_bak = rms.copy();
    rms = list(np.diff(rms))
    rms.insert(0,0)

    b, a = signal.butter(8, 0.35, analog=False)
    sig_ff = signal.filtfilt(b, a, rms)

    # Savitzky-Golay filter 平滑
    # from scipy.signal import savgol_filter
    # sig_ff = savgol_filter(rms, 5, 1)  # window size 51, polynomial order 3
    # sig_ff = signal.medfilt(rms, 5)  # 二维中值滤波
    sig_ff = [x/np.std(sig_ff) for x in sig_ff]

    # print("1 hline is {}".format(hline))
    max_indexs = [i for i in range(1,len(sig_ff)-1) if sig_ff[i]>sig_ff[i-1] and sig_ff[i]>sig_ff[i+1] and sig_ff[i] > hline]
    sig_ff_on_max_indexs = [sig_ff[x] for x in max_indexs]
    # print("sig_ff_on_max_indexs is {}, size {}".format(sig_ff_on_max_indexs,len(sig_ff_on_max_indexs)))
    # print("-1 max_indexs is {},size is {}".format(max_indexs, len(max_indexs)))

    code = parse_onset_code(onset_code)
    code = [int(x) for x in code]
    if code[-1] >= 2000:
        width_last = len(rms) * code[-1] / np.sum(code)
        max_indexs = [x for x in max_indexs if x < len(rms) - int(width_last * 0.4)]
    else:
        max_indexs = [x for x in max_indexs if x < len(rms) - 5]
    # sig_ff_on_max_indexs = [sig_ff[x] for x in max_indexs]
    # tmp = sig_ff_on_max_indexs.copy()
    # tmp.sort()
    # min_index = sig_ff_on_max_indexs.index(tmp[0])
    # second_index = sig_ff_on_max_indexs.index(tmp[1])
    # if sig_ff_on_max_indexs[second_index] - sig_ff_on_max_indexs[min_index] > np.std(sig_ff_on_max_indexs)*0.8:
    #     max_indexs.remove(max_indexs[min_index])
    return rms_bak,rms,sig_ff,max_indexs

def get_best_max_index(filename,onset_code):
    code = parse_onset_code(onset_code)
    code = [int(x) for x in code]
    base_symbols = get_all_symbols(code)

    rms, rms_diff, sig_ff, max_indexs_first = get_rms_max_indexs_for_onset(filename, onset_code,0.4)
    max_indexs_first_bak = max_indexs_first.copy()
    start, end, total_length = get_start_end_length_by_max_index(max_indexs_first, filename)
    max_indexs_first.append(end if end < len(rms) - 5 else len(rms) - 5)
    types = get_onset_type(max_indexs_first, onset_code,end)
    all_symbols_first = get_all_symbols(types)

    rms, rms_diff, sig_ff, max_indexs_second = get_rms_max_indexs_for_onset(filename, onset_code, 0.55)
    max_indexs_second_bak = max_indexs_second.copy()
    start, end, total_length = get_start_end_length_by_max_index(max_indexs_second, filename)
    max_indexs_second.append(end if end < len(rms) - 5 else len(rms) - 5)
    types = get_onset_type(max_indexs_second, onset_code,end)
    all_symbols_second = get_all_symbols(types)

    lcs_first = find_lcseque(base_symbols, all_symbols_first)
    lcs_second = find_lcseque(base_symbols, all_symbols_second)
    if len(lcs_first) >= len(lcs_second):
        if len(max_indexs_first_bak) < len(code) and len(code) - len(max_indexs_first_bak) == 1:
            lcs, positions, raw_positions = my_find_lcseque(base_symbols, all_symbols_first)
            check_status = [positions[i] for i in range(len(positions)-3) if positions[i] - positions[i-1] == 1 and positions[i+1] - positions[i] == 3 and positions[i+2] - positions[i+1] == 1]
            if len(check_status) == 1:
                start_point = check_status[0] + 1
                rms, rms_diff, sig_ff, max_indexs = get_rms_max_indexs_for_find_loss(filename, onset_code, 0.1)
                select_added = [m for m in max_indexs if m > max_indexs_first[start_point] and m < max_indexs_first[start_point +1]]
                if len(select_added) > 0 :
                    max_indexs_first.append(select_added[0])
                    max_indexs_first.sort()
            return max_indexs_first
        else:
            #print(11)
            return max_indexs_first
    else:
        if len(max_indexs_second_bak) < len(code) and len(code) - len(max_indexs_second_bak) == 1:
            lcs, positions, raw_positions = my_find_lcseque(base_symbols, all_symbols_second)
            check_status = [i for i in range(len(positions)-3) if positions[i] - positions[i-1] == 1 and positions[i+1] - positions[i] == 3 and positions[i+2] - positions[i+1] == 1]
            if len(check_status) == 1:
                start_point = check_status[0] + 1
                rms, rms_diff, sig_ff, max_indexs = get_rms_max_indexs_for_find_loss(filename, onset_code, 0.1)
                select_added = [m for m in max_indexs if m > max_indexs_second[start_point] and m < max_indexs_second[start_point +1]]
                if len(select_added) > 0 :
                    max_indexs_second.append(select_added[0])
                    max_indexs_second.sort()
            return max_indexs_second
        else:
            #print(22)
            return max_indexs_second

def get_start_end_length_by_max_index(max_indexs,filename):
    y, sr = librosa.load(filename)
    rms = librosa.feature.rmse(y=y)[0]
    rms_mean = np.mean(rms)
    end = len(rms) -5
    for i in range(len(rms)-5,20,-1):
        if rms[i] > rms_mean * 0.1:
            end = i
            break
    start = max_indexs[0]
    total_frames_number = end - start
    return start,end,total_frames_number


def get_topN_rms_max_indexs_for_onset(filename,topN):
    rms,rms_diff, sig_ff, max_indexs = get_rms_max_indexs_for_onset(filename)
    rms_on_max_indexs = [rms[x] for x in max_indexs]
    topN_max_indexs = find_n_largest(rms_on_max_indexs, topN)

    result = []
    for x in topN_max_indexs:
        rms_tmp = rms_on_max_indexs[x]
        result.append(rms.index(rms_tmp))
    return result
def find_n_largest(a,topN):
    import heapq

    a = list(a)
    #a = [43, 5, 65, 4, 5, 8, 87]

    re1 = heapq.nlargest(topN, a)  # 求最大的三个元素，并排序
    re1.sort()
    #re2 = map(a.index, heapq.nlargest(total, a))  # 求最大的三个索引    nsmallest与nlargest相反，求最小
    re2 = [i for i,x in enumerate(a) if x in re1]

    #print(re1)
    #print(list(re2))  # 因为re1由map()生成的不是list，直接print不出来，添加list()就行了
    return list(re2)

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


def get_onset_type(onset_frames,onset_code,end):

    if len(onset_frames) == 0:
        return []
    #print("start_index is {},size is {}".format(start_indexs,len(start_indexs)))
    code = parse_onset_code(onset_code)
    code = [int(x) for x in code]

    #print("code is {},size is {}".format(code, len(code)))

    total_length_no_last = np.sum(code)
    # real_total_length_no_last = onset_frames[-1] - onset_frames[0]
    real_total_length_no_last = end - onset_frames[0]
    rate = real_total_length_no_last/total_length_no_last
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
    return types

def get_onset_type_by_opt(onset_frames,onset_code,end):

    if len(onset_frames) == 0:
        return []
    best_total = 1000000
    best_types = []
    best_speed = 0
    code = parse_onset_code(onset_code)
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
    return best_types,best_speed

def get_onset_type_by_part(onset_frames,onset_code,end,start_index):

    if len(onset_frames) == 0:
        return []
    #print("start_index is {},size is {}".format(start_indexs,len(start_indexs)))
    code = parse_onset_code(onset_code)
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

def get_all_symbols(types):
    symbols = ''
    for t in types:
        s = get_type_symbol(t)
        symbols = symbols + s
    return symbols

def calculate_score(max_indexs,onset_code,end):
    types,best_speed = get_onset_type_by_opt(max_indexs, onset_code,end)
    print(types)
    all_symbols = get_all_symbols(types)
    #print(all_symbols)
    code = parse_onset_code(onset_code)
    code = [int(x) for x in code]
    base_symbols = get_all_symbols(code)
    #print(base_symbols)
    # lcs = find_lcseque(base_symbols, all_symbols)
    offset_detail = ''

    offset_threshold = 180
    types, real_types = get_offset_for_each_onsets_by_speed(max_indexs, onset_code,end)
    # types, real_types = get_offset_for_each_onsets_with_speed(max_indexs, onset_code,end,best_speed)
    baseline_offset = [np.abs(types[i] - real_types[i]) for i in range(len(types)) if types[i] == np.min(types)]
    baseline_offset = np.min(baseline_offset) #基准偏差
    # 找出偏差大于125的节拍，判断是要减掉基准偏差
    offset_indexs = [i for i in range(len(types)-1) if np.abs(types[i] - real_types[i]) > baseline_offset * int(types[i]/np.min(types)) and np.abs(types[i] - real_types[i]) - baseline_offset * int(types[i]/np.min(types))  > offset_threshold]
    if len(offset_indexs) > 0:
        str_tmp = list(all_symbols)
        for i in offset_indexs:
            str_tmp[i]  = '0'
        all_symbols = ''.join(str_tmp)
        offset_values = [np.abs(types[i] - real_types[i]) if np.abs(types[i] - real_types[i]) < baseline_offset * int(types[i]/np.min(code)) else np.abs(types[i] - real_types[i]) - baseline_offset * int(types[i]/np.min(code))  for i in range(len(types))]
        offset_detail = "。判定音符类型为 {}，实际音符为 {}，偏差值为 {}，其中大于{}的也都会被视为错误节拍（不包括最后一个节拍）".format(types, real_types, offset_values,offset_threshold)

    lcs, positions,raw_positions = my_find_lcseque(base_symbols, all_symbols)
    each_symbol_score = 100 / len(code)
    total_score = int(len(lcs) * each_symbol_score)

    detail = get_matched_detail(base_symbols, all_symbols, lcs)
    detail = detail + offset_detail

    ex_total = len(all_symbols) - len(lcs) -1
    ex_rate = ex_total / len(base_symbols)
    if len(all_symbols) > len(base_symbols):
        if ex_rate > 0.4:                                # 节奏个数误差超过40%，总分不超过50分
            total_score = total_score if total_score < 50 else 50
            detail = detail + "，多唱节奏个数误差超过40%，总分不得超过50分"
        elif ex_rate > 0.3:                             # 节奏个数误差超过30%，总分不超过65分（超过的）（30-40%）
            total_score = total_score if total_score < 65 else 65
            detail = detail + "，多唱节奏个数误差超过30%，总分不得超过65分"
        elif ex_rate > 0.2:                             # 节奏个数误差超过20%，总分不超过80分（超过的）（20-30%）
            total_score = total_score if total_score < 80 else 80
            detail = detail + "，多唱节奏个数误差超过20%，总分不得超过80分"
        elif ex_rate > 0:                                           # 节奏个数误差不超过20%，总分不超过90分（超过的）（0-20%）
            total_score = total_score if total_score < 90 else 90
            detail = detail + "，多唱节奏个数误差在（1-20%），总分不得超过90分"
    return total_score,detail

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
    return str_detail_list

def get_offset_for_each_onsets_by_speed(max_indexs, onset_code,end):
    code = parse_onset_code(onset_code)
    code = [int(x) for x in code]
    types = get_onset_type(max_indexs, onset_code,end)
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

def get_offset_for_each_onsets_with_speed(max_indexs, onset_code,end,speed):
    code = parse_onset_code(onset_code)
    code = [int(x) for x in code]
    types = get_onset_type(max_indexs, onset_code,end)
    index_diff = np.diff(max_indexs)
    # vs = [int(types[i]) / index_diff[i] for i in range(len(index_diff))]
    real_types = [int(d * speed) for d in index_diff]
    # print("index_diff is {},size is {}".format(index_diff, len(index_diff)))
    # print("vs is {},size is {}".format(vs, len(vs)))
    # print("vs mean is {}".format(np.mean(vs)))
    # print("types is {},size is {}".format(types, len(types)))
    # print("real_types is {},size is {}".format(real_types, len(real_types)))
    # print("code is {},size is {}".format(code, len(code)))
    return types,real_types


def draw_plt(filename,onset_code,rms,sig_ff,max_indexs,start,end):
    # max_indexs = get_topN_rms_max_indexs_for_onset(filename, 10)
    times = librosa.frames_to_time(np.arange(len(rms)))
    plt.clf()
    plt.subplot(2, 1, 1)
    #plt.rcParams['figure.figsize'] = (16, 8)  # 设置figure_size尺寸
    plt.title(filename)
    plt.xlabel("识别结果示意图")
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.plot(times, rms)
    # plt.plot(times, rms_diff)
    plt.plot(times, sig_ff)
    #print(np.std(sig_ff))
    sig_ff = [x if x > 0 else 0 for x in sig_ff]
    sig_ff_on_max_indexs = [sig_ff[x] for x in max_indexs]
    topN_indexs = find_n_largest(sig_ff_on_max_indexs, 4)
    top_index = sig_ff_on_max_indexs.index(np.max(sig_ff_on_max_indexs))
    hline = np.mean([sig_ff_on_max_indexs[i] for i in range(len(sig_ff_on_max_indexs)) if i in topN_indexs and i != top_index]) * 0.25
    # print("2 hline is {}".format(hline))
    #print(hline)
    #print(np.std(sig_ff))
    plt.xlim(0, np.max(times))
    max_index_times = librosa.frames_to_time(max_indexs)
    plt.vlines(max_index_times, 0, np.max(rms), color='r', linestyle='dashed')
    start_time = librosa.frames_to_time(start)
    end_time = librosa.frames_to_time(end)
    plt.vlines(start_time, 0, np.max(rms) / 2, color='black', linestyle='solid')
    plt.vlines(end_time, 0, np.max(rms) / 2, color='black', linestyle='solid')
    plt.hlines(hline, 0, np.max(times), color='r', linestyle='dashed')
    plt.hlines(0, 0, np.max(times), color='black', linestyle='solid')

    plt.subplot(2, 1, 2)
    plt.xlabel("标准节拍示意图")
    base_frames = onsets_base_frames(onset_code, end-start)
    base_frames = [x - (base_frames[0] - max_indexs[0]) for x in base_frames]
    base_frames_times = librosa.frames_to_time(base_frames)
    plt.vlines(base_frames_times, 0, 10, color='r', linestyle='dashed')
    plt.xlim(0, np.max(times))
    return plt