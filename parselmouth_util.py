import parselmouth

import numpy as np
import matplotlib.pyplot as plt
import librosa
from LscHelper import my_find_lcseque
import scipy.signal as signal
from xfyun.wav2pcm import *
from pingfen_uitl import get_lcseque_and_position_with_time_offset,get_all_scores_by_st,get_all_scores_with_5
from xfyun.iat_ws_python3 import get_iat_result

FREQS = [
    ('B0',30.87), ('C1',32.7), ('C#1',34.65),
    ('D1',36.71), ('D#1',38.89), ('E1',41.2),
    ('F1',43.65), ('F#1',46.35), ('G1',49),
    ('G#1',51.91), ('A1',55), ('A#1',58.27),
    ('B1',61.74), ('C2',65.41), ('C#2',69.3),
    ('D2',73.42), ('D#2',77.78), ('E2',82.41),
    ('F2',87.31), ('F#2',92.50), ('G2',98.00),
    ('G#2',103.83), ('A2',110.00), ('A#2',116.54),
    ('B2',123.54), ('C3',130.81), ('C#3',138.59),
    ('D3',146.83), ('D#3',155.56), ('E3',164.81),
    ('F3',174.61), ('F#3',185.00), ('G3',196.00),
    ('G#3',207.65), ('A3',220.00), ('A#3',233.08),
    ('B3',246.94), ('C4',261.63), ('C#4',277.18),
    ('D4',293.66), ('D#4',311.13), ('E4',329.63),
    ('F4',349.23), ('F#4',369.99), ('G4',392.00),
    ('G#4',415.30), ('A4' ,440.00), ('A#4',466.16),
    ('B4',493.88), ('C5',523.25), ('C#5',554.37),
    ('D5',587.33), ('D#5',622.25), ('E5',659.26),
    ('F5',698.46), ('F#5',739.99), ('G5',783.99),
    ('G#5',830.61), ('A5',880,00), ('A#5',932.33),
    ('B5',987.77), ('C6',1046.50), ('C#6',1108.73),
    ('D6',1174.66), ('D#6',1244.51), ('E6',1318.51),
    ('F6',1396.91), ('F#6',1479.98), ('G6',1567.98),
    ('G#6',1661.22), ('A6',1760.00), ('A#6',1864.66),
    ('B6',1975.53), ('C7',2093), ('C#7',2217.46),
    ('D7',2349.32), ('D#7',2489.02), ('E7',2637.03),
    ('F7',2793.83), ('F#7',2959.96), ('G7',3135.44),
    ('G#7',3322.44), ('A7',3520), ('A#7',3729.31),
    ('B7',3951.07)
]

PITCH_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B', 'C', 'C#', 'D3', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def get_freq_by_notation_name(notation_name):
    freq = [tup for tup in FREQS if tup[0] == notation_name]
    if len(freq) > 0:
        return freq[0][1]
    else:
        return None

'''
获取平移后的音高类型
'''
def get_numbered_musical_notation_by_moved(str,step):
    anchor_point = [i for i,p in enumerate(PITCH_NAMES) if i >= 12 and i <= 23 and p.find(str[0]) >= 0]
    anchor_point_moved = anchor_point[0] + step # step > 0 表示向上移动；step < 0 表示向下移动
    notation_on_anchor_point = PITCH_NAMES[anchor_point_moved]
    numbered_notation_on_anchor_point = get_numbered_musical_notation(notation_on_anchor_point)
    return numbered_notation_on_anchor_point


def get_numbered_musical_notation(str):
    if str.find("C") >= 0:
        result = "1"
    elif str.find("D") >= 0:
        result = "2"
    elif str.find("E") >= 0:
        result = "3"
    elif str.find("F") >= 0:
        result = "4"
    elif str.find("G") >= 0:
        result = "5"
    elif str.find("A") >= 0:
        result = "6"
    elif str.find("B") >= 0:
        result = "7"

    if str.find("#") >= 0:
        return result + "#"
    else:
        return result

def get_musical_notation_with_number(str,type):
    if str.find("1") >= 0:
        if type == "capital":
            result = "C"
        else:
            result = "c"
    elif str.find("2") >= 0:
        if type == "capital":
            result = "D"
        else:
            result = "d"
    elif str.find("3") >= 0:
        if type == "capital":
            result = "E"
        else:
            result = "e"
    elif str.find("4") >= 0:
        if type == "capital":
            result = "F"
        else:
            result = "f"
    elif str.find("5") >= 0:
        if type == "capital":
            result = "G"
        else:
            result = "g"
    elif str.find("6") >= 0:
        if type == "capital":
            result = "A"
        else:
            result = "a"
    elif str.find("7") >= 0:
        if type == "capital":
            result = "B"
        else:
            result = "b"

    if str.find("#") >= 0:
        return result + "#"
    else:
        return result

def draw_pitch(pitch,draw_type=1,filename='',notation='',grain_size=0):
    # Extract selected pitch contour, and
    # replace unvoiced samples by NaN to not plot
    p_min = 100
    p_max = 300
    pitch_values = pitch.selected_array['frequency']
    select_pitch_values = [p for p in pitch_values if p != 0]
    pitch_values_max = np.max(select_pitch_values)
    pitch_values_mean = np.mean(select_pitch_values)
    # pitch_values = pitch_values + 60   #平移操作
    p_min = np.min(pitch_values) - 30 if np.min(pitch_values) - 30 > 80 else 80
    p_min = int(p_min)
    p_max = np.max(pitch_values) + 30
    p_max = int(p_max)

    #防止个别偏离现象
    if pitch_values_max - pitch_values_mean > 100:
        p_min = int(pitch_values_mean * 0.5)
        p_max = int(pitch_values_mean * 1.5)
    pitch_values[pitch_values==0] = np.nan
    if draw_type == 1:
        plt.plot(pitch.xs(), pitch_values, 'o', markersize=5, color='w')
        plt.plot(pitch.xs(), pitch_values, 'o', markersize=2)
    else:
        if grain_size == 1:
            freqs = FREQS
        else:
            freqs =  [tup for tup in FREQS if tup[0].find('#') < 0]
        freqs_points = [tup[1] for tup in freqs]
        # freqs_points = [tup[1] for tup in FREQS if tup[0].find('#') < 0]
        pitch_values_candidate = []  # 最靠近的音符
        for p in pitch_values:
            gaps = [np.abs(f - p) for f in freqs_points]
            gap_min = np.min(gaps)
            if np.isnan(gap_min):
                pitch_values_candidate.append(np.nan)
            else:
                p = gaps.index(gap_min)
                pitch_values_candidate.append(freqs_points[p])
        plt.plot(pitch.xs(), pitch_values, 'o', markersize=2)
        # 打印平移后的音高轨迹线
        # pitch_values_moved = pitch_values + 55  # 平移操作
        # pitch_values_candidate_moved = get_pitch_values(pitch_values_moved)
        # plt.plot(pitch.xs(), pitch_values_moved, ':', markersize=2, color="r")

        # 将小缝隙补齐
        pitch_values_candidate = smooth_pitch_values_candidate(pitch_values_candidate)
        # pitch_values_candidate_moved = smooth_pitch_values_candidate(pitch_values_candidate_moved)
        plt.plot(pitch.xs(), pitch_values_candidate, 'o', markersize=2)
        # plt.plot(pitch.xs(), pitch_values_candidate_moved, '*', markersize=4, color="r")
    plt.grid(False)
    plt.title(filename, fontsize=16)
    # plt.ylim(0, pitch.ceiling)
    pitch_all = [p for p in freqs_points if p > p_min and p < p_max]
    plt.hlines(pitch_all, 0, len(pitch_values), color = '0.2', linewidth=1, linestyle=":")
    plt.ylim(p_min, p_max)
    plt.ylabel("fundamental frequency [Hz]")
    plt.xlabel(notation)
    pitch_name = [tup[0] for tup in freqs if tup[1] > p_min and tup[1] < p_max]
    for i,p in enumerate(pitch_all):
        numbered_musical_notation = get_numbered_musical_notation(pitch_name[i])
        plt.text(0.1, p, pitch_name[i] + " - " + numbered_musical_notation,size='8')

    # plt.xlim([snd.xmin, snd.xmax])
    return plt

def draw_pitch_specified (intensity,pitch,pitch_values,draw_type=1,filename='',notation='',grain_size=0):
    # Extract selected pitch contour, and
    # replace unvoiced samples by NaN to not plot
    p_min = 70
    p_max = 300
    # pitch_values = pitch.selected_array['frequency']
    select_pitch_values = [p for p in pitch_values if p != 0]
    pitch_values_max = np.max(select_pitch_values)
    pitch_values_mean = np.mean(select_pitch_values)
    # pitch_values = pitch_values + 60   #平移操作
    p_min = np.min(pitch_values) - 30 if np.min(pitch_values) - 30 > 80 else 80
    p_min = int(p_min)
    p_max = np.max(pitch_values) + 30
    p_max = int(p_max)

    #防止个别偏离现象
    # if pitch_values_max - pitch_values_mean > 100:
    #     p_min = int(pitch_values_mean * 0.5)
    #     p_max = int(pitch_values_mean * 1.5)
    pitch_values[pitch_values==0] = np.nan
    if draw_type == 1:
        plt.plot(pitch.xs(), pitch_values, 'o', markersize=5, color='w')
        plt.plot(pitch.xs(), pitch_values, 'o', markersize=2)
    else:
        if grain_size == 1:
            freqs = FREQS
        else:
            freqs =  [tup for tup in FREQS if tup[0].find('#') < 0]
        freqs_points = [tup[1] for tup in freqs]
        # freqs_points = [tup[1] for tup in FREQS if tup[0].find('#') < 0]
        pitch_values_candidate = []  # 最靠近的音符
        for p in pitch_values:
            gaps = [np.abs(f - p) for f in freqs_points]
            gap_min = np.min(gaps)
            if np.isnan(gap_min):
                pitch_values_candidate.append(np.nan)
            else:
                p = gaps.index(gap_min)
                pitch_values_candidate.append(freqs_points[p])
        plt.plot(pitch.xs(), pitch_values, 'o', markersize=2)
        # 打印平移后的音高轨迹线
        # pitch_values_moved = pitch_values + 55  # 平移操作
        # pitch_values_candidate_moved = get_pitch_values(pitch_values_moved)
        # plt.plot(pitch.xs(), pitch_values_moved, ':', markersize=2, color="r")

        # 将小缝隙补齐
        pitch_values_candidate = smooth_pitch_values_candidate(pitch_values_candidate)
        # pitch_values_candidate_moved = smooth_pitch_values_candidate(pitch_values_candidate_moved)
        plt.plot(pitch.xs(), pitch_values_candidate, 'o', markersize=2)
        # plt.plot(pitch.xs(), pitch_values_candidate_moved, '*', markersize=4, color="r")
    plt.grid(False)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.title(filename, fontsize=16)
    # plt.ylim(0, pitch.ceiling)
    pitch_all = [p for p in freqs_points if p > p_min and p < p_max]
    plt.hlines(pitch_all, 0, len(pitch_values), color = '0.2', linewidth=1, linestyle=":")
    # p_min, p_max = 70,500
    plt.ylim(p_min, p_max)
    plt.ylabel("fundamental frequency [Hz]")
    plt.xlabel(notation)
    # 设置坐标轴刻度
    x_ticks = np.arange(0, pitch.duration, 1)
    plt.xticks(x_ticks)
    pitch_name = [tup[0] for tup in freqs if tup[1] > p_min and tup[1] < p_max]
    for i,p in enumerate(pitch_all):
        numbered_musical_notation = get_numbered_musical_notation(pitch_name[i])
        plt.text(0.1, p, pitch_name[i] + " - " + numbered_musical_notation,size='8')

    # plt.xlim([snd.xmin, snd.xmax])
    plt.twinx()
    draw_intensity(intensity)
    return plt

def draw_spectrogram(spectrogram, dynamic_range=70):
    X, Y = spectrogram.x_grid(), spectrogram.y_grid()
    sg_db = 10 * np.log10(spectrogram.values)
    plt.pcolormesh(X, Y, sg_db, vmin=sg_db.max() - dynamic_range, cmap='afmhot')
    plt.ylim([spectrogram.ymin, spectrogram.ymax])
    plt.xlabel("time [s]")
    plt.ylabel("frequency [Hz]")

def draw_intensity(intensity):
    # plt.plot(intensity.xs(), intensity.values.T, linewidth=3, color='w')
    values = intensity.values.T.copy()
    values = list(values)
    values = [v[0] for v in values]
    values = signal.medfilt(values, 11)
    plt.plot(intensity.xs(), intensity.values.T, linewidth=1)
    plt.plot(intensity.xs(), values, linewidth=1)
    plt.grid(False)
    plt.ylim(0)
    plt.ylabel("intensity [dB]")

def get_mean_pitch(start_frame,end_frame,sr,pitch):
    #将起始帧和结束帧换算成时间点
    onset_times = librosa.frames_to_time([start_frame,end_frame], sr=sr)

    if onset_times[0] > pitch.duration or onset_times[1] > pitch.duration:
        print("the parameter is wrong")
        return None

    #获得总帧数
    frames_total = int(np.floor((pitch.duration - pitch.t1) / pitch.dt) - 1)

    #librosa时间点换算成parselmouth的帧所在位置
    ps_frame = int(onset_times[0] * frames_total / pitch.duration) + 1
    pe_frame = int(onset_times[1] * frames_total / pitch.duration) + 1
    pe_frame = pe_frame if pe_frame < frames_total -1 else frames_total -1 # 防止越界

    pitch_values = pitch.selected_array['frequency']
    pitch_tmp = pitch_values[ps_frame:pe_frame]
    mean_pitch = np.median(pitch_tmp)
    return  mean_pitch

def get_pitch_by_parselmouth(filename):
    snd = parselmouth.Sound(filename)
    pitch = snd.to_pitch()
    return pitch

# def draw_intensity(intensity):
#     # plt.plot(intensity.xs(), intensity.values.T, linewidth=3, color='w')
#     intensity_values_t = intensity.values.T - 50
#     plt.plot(intensity.xs(), intensity_values_t, linewidth=1, color='r')
#     plt.grid(False)
#     plt.ylim(0)
#     plt.ylabel("intensity [dB]")

'''
根据起始帧和结束帧提取特定时段的音高，包括第一判定音高，第二判定音高
'''
def get_pitch_by_start_end(pitch,start_frame,end_frame,sr):
    from collections import Counter
    pitch_values = pitch.selected_array['frequency']
    pitch_values[pitch_values == 0] = np.nan
    # 将起始帧和结束帧换算成时间点
    onset_times = librosa.frames_to_time([start_frame, end_frame], sr=sr)

    if onset_times[0] > pitch.duration or onset_times[1] > pitch.duration:
        print("the parameter is wrong")
        return None

    # 获得总帧数
    frames_total = int(np.floor((pitch.duration - pitch.t1) / pitch.dt) - 1)

    # librosa时间点换算成parselmouth的帧所在位置
    ps_frame = int(onset_times[0] * frames_total / pitch.duration) + 1
    pe_frame = int(onset_times[1] * frames_total / pitch.duration) + 1
    pe_frame = pe_frame if pe_frame < frames_total - 1 else frames_total - 1  # 防止越界

    pitch_tmp = pitch_values[ps_frame:pe_frame]

    freqs = [tup for tup in FREQS if tup[0].find('#') < 0] # 筛选不含半音的标准音高序列
    freqs_points = [tup[1] for tup in freqs]
    # freqs_points = [tup[1] for tup in FREQS if tup[0].find('#') < 0]
    pitch_values_candidate = []  # 最靠近的音符
    for p in pitch_tmp: #遍历该段节奏的音高频率点，找出每个频率最靠近的标准音高，放入pitch_values_candidate中
        gaps = [np.abs(f - p) for f in freqs_points]
        gap_min = np.min(gaps)
        if np.isnan(gap_min):
            pitch_values_candidate.append(np.nan)
        else:
            p = gaps.index(gap_min)
            pitch_values_candidate.append(freqs_points[p])

    # 找出list中出现最多的元素
    res = Counter(pitch_values_candidate)
    if len(pitch_values_candidate) == 0: # 如果整个音高序列为空
        return np.nan,np.nan
    first_candidate = res.most_common(1)[0][0] # 第一判定音高
    pitch_values_candidate_tmp = [p for p in pitch_values_candidate if p != first_candidate]

    if len(pitch_values_candidate_tmp) == 0:
        second_candidate = first_candidate
    else:
        not_nan_list = [p for p in pitch_values_candidate_tmp if not np.isnan(p)] #不为nan的音高序列
        #如果剥除第一判定音高后的序列中包括不为nan的音高,则从不为nan的音高中取第二判定音高，否则第二判定音高为nan
        if len(not_nan_list) != 0:
            res = Counter(not_nan_list)
            second_candidate = res.most_common(1)[0][0]  # 第二判定音高
        else: #不包括其他音高（即非nan音高），第二判定音高为nan
            second_candidate = np.nan
    first_candidate_name = first_candidate if np.isnan(first_candidate) else get_pitch_name(first_candidate)
    second_candidate_name = second_candidate if np.isnan(second_candidate) else get_pitch_name(second_candidate)
    return first_candidate_name,second_candidate_name,first_candidate,second_candidate

def get_all_pitch_candidate(pitch,onset_frames,sr):
    pitch_values = pitch.selected_array['frequency']
    pitch_values[pitch_values == 0] = np.nan
    # 找出最后一个不为0、不为nan的元素的位置
    # last_position = [i for i in range(len(pitch_values)) if pitch_values[i] != 0 and not np.isnan(pitch_values[i])][-1]
    last_position = librosa.time_to_frames(pitch.duration - 0.05, sr=sr)
    onset_frames.append(last_position)
    all_first_candidates = []
    all_second_candidates = []
    all_first_candidate_names = []
    all_second_candidate_names = []
    for i in range(len(onset_frames)-1):
        start_frame = onset_frames[i]
        end_frame = onset_frames[i+1]
        try:
            # 以当前起始点为起点，下一个起始点为终点，获取该段节奏上的音高
            first_candidate_name, second_candidate_name, first_candidate, second_candidate = get_pitch_by_start_end(pitch,start_frame,end_frame,sr)
        except Exception:
            pass
        if np.isnan(first_candidate) and not np.isnan(second_candidate): # 如果第一判定为nan，第二判定不为nan
            first_candidate_name = second_candidate_name
            first_candidate = second_candidate
        if not np.isnan(first_candidate) and not np.isnan(second_candidate):
            all_first_candidates.append(first_candidate)
            all_second_candidates.append(second_candidate)
            all_first_candidate_names.append(first_candidate_name)
            all_second_candidate_names.append(second_candidate_name)
        # print(first_candidate_name, second_candidate_name)
    return all_first_candidate_names,all_second_candidate_names,all_first_candidates,all_second_candidates

def get_pitch_name(freq):
    freqs = [tup for tup in FREQS]
    for tup in freqs:
        if tup[1] == freq:
            return tup[0]
    return np.nan

'''
如果音高序列中部分音高需要变换成带“+”或“-”的音高，则进行相关变换
'''
def change_pitch_seque(all_first_candidate_names,all_first_candidates):
    freqs_points_7 = [tup[1] for tup in FREQS if tup[0].find('B') >= 0]
    freqs_points_1 = [tup[1] for tup in FREQS if tup[0].find('C') >= 0 and tup[0].find('#') < 0]

    # 根据音高序列中最高频率和最低频率是否跨两个八度
    max_freq = np.max(all_first_candidates)
    max_postion = all_first_candidates.index(max_freq)
    max_name = all_first_candidate_names[max_postion]
    numbered_max_name = get_numbered_musical_notation(max_name)
    numbered_max_name = int(numbered_max_name)

    min_freq = np.min(all_first_candidates)
    min_postion = all_first_candidates.index(min_freq)
    min_name = all_first_candidate_names[min_postion]
    numbered_min_name = get_numbered_musical_notation(min_name)
    numbered_min_name = int(numbered_min_name)

    if max_freq == min_freq: #只有同一种音高
        result = [get_numbered_musical_notation(n) for n in all_first_candidate_names]
    elif numbered_max_name <= 7 and  numbered_min_name >= 1 and numbered_max_name > numbered_min_name: # 在同一个八度里面
        result = [get_numbered_musical_notation(n) for n in all_first_candidate_names]
    else:

        #音高为"7"的频率
        freqs_7 = [f for f in freqs_points_7 if f >= min_freq and f <= max_freq]
        check_freq = freqs_7[0]

        more_7 = [f for f in all_first_candidates if f >= check_freq]
        less_7 = [f for f in all_first_candidates if f < check_freq]

        if len(more_7) <= len(less_7):  # 顶部超出 ()
            result = [get_numbered_musical_notation(all_first_candidate_names[i]) if n < check_freq else get_numbered_musical_notation(all_first_candidate_names[i]) + "+" for i,n in enumerate(all_first_candidates)]
        else: # 底部超出
            result = [get_numbered_musical_notation(all_first_candidate_names[i]) if n > check_freq else get_numbered_musical_notation(all_first_candidate_names[i]) + "-" for i,n in enumerate(all_first_candidates)]
    return result

'''
 根据最大公共子序列，计算绝对音高序列的匹配结果
'''
def get_matched_detail_absolute_pitch(base_symbols, all_symbols,threshold_score=60):
    detail_list = np.zeros(len(base_symbols))
    # start_index = 0
    base_symbols_encode = get_encode_pitch_seque(base_symbols) # 编码
    all_symbols_encode = get_encode_pitch_seque(all_symbols) # 编码
    # print("base_symbols_encode is {},size is {}".format(base_symbols_encode, len(base_symbols_encode)))
    # print("all_symbols_encode is {},size is {}".format(all_symbols_encode, len(all_symbols_encode)))
    lcseque, positions,raw_positions = my_find_lcseque(base_symbols_encode, all_symbols_encode)
    for index in positions:
        # index = base_symbols[start_index:].index(l)
        detail_list[index] = 1

    str_detail_list = '旋律识别的结果是：' + str(detail_list)
    str_detail_list = str_detail_list.replace("1", "√")
    str_detail_list = str_detail_list.replace("0", "×")

    ex_total = len(all_symbols) - len(base_symbols)
    each_symbol_score = threshold_score / len(base_symbols)
    total_score = int(len(lcseque) * each_symbol_score)

    if len(all_symbols) > len(base_symbols):
        str_detail_list = str_detail_list + "， 多唱节拍数有：" + str(ex_total) + "个"
    ex_total = len(all_symbols) - len(lcseque) - 1
    ex_rate = ex_total / len(base_symbols)

    detail = str_detail_list
    if len(all_symbols) > len(base_symbols):
        if ex_rate > 0.4:  # 节奏个数误差超过40%，总分不超过50分
            total_score = total_score if total_score < threshold_score * 0.50 else threshold_score * 0.50
            detail = detail + "，多唱节奏个数误差超过40%，总分不得超过50分"
        elif ex_rate > 0.3:  # 节奏个数误差超过30%，总分不超过65分（超过的）（30-40%）
            total_score = total_score if total_score < threshold_score * 0.65 else threshold_score * 0.65
            detail = detail + "，多唱节奏个数误差超过30%，总分不得超过65分"
        elif ex_rate > 0.2:  # 节奏个数误差超过20%，总分不超过80分（超过的）（20-30%）
            total_score = total_score if total_score < threshold_score * 0.80 else threshold_score * 0.80
            detail = detail + "，多唱节奏个数误差超过20%，总分不得超过80分"
        elif ex_rate > 0:  # 节奏个数误差不超过20%，总分不超过90分（超过的）（0-20%）
            total_score = total_score if total_score < threshold_score * 0.90 else threshold_score * 0.90
            detail = detail + "，多唱节奏个数误差在（1-20%），总分不得超过90分"
    return total_score,lcseque,detail,detail_list,raw_positions

'''
从备选音高中找出未匹配的音高
all_second_candidate_names is ['G3', 'G3', 'F3', 'E3', 'F3', 'G3', 'D3', 'D3', 'D3', 'C3', 'G2']
pitch_code_for_absolute_pitch is ['6', '5', '3', '6', '3', '5', '3', '2', '1', '6-']
'''
def check_from_second_candidate_names(note_score_absolute_pitch,str_detail_list,detail_list, all_second_candidate_names,pitch_code_for_absolute_pitch):
    detail_list_bak = detail_list.copy()
    threshold_score = 60
    each_symbol_score = threshold_score / len(pitch_code_for_absolute_pitch)
    if np.sum(detail_list) == len(detail_list):
        return note_score_absolute_pitch,str_detail_list
    else:
        for i,s in enumerate(detail_list):
            if s == 0 and i < len(pitch_code_for_absolute_pitch):
                    pitch_numbered = pitch_code_for_absolute_pitch[i]
                    pitch_numbered = pitch_numbered.replace("-","").replace("+","")
                    pitch_name = get_musical_notation_with_number(pitch_numbered, 'capital')
                    # 从备份音高列表中取出出错音高，记录其所在位置
                    exist_positions = [i for i in range(len(all_second_candidate_names)) if all_second_candidate_names[i][0] == pitch_name]
                    if len(exist_positions) > 0: # 如果存在于备选音高
                        tmp = [e for e in exist_positions if np.abs(e - i) <= 1]
                        if len(tmp) > 0:
                            note_score_absolute_pitch += each_symbol_score
                            detail_list_bak[i] = 1
    tmp = '旋律识别的结果是：' + str(detail_list_bak)
    tmp = tmp.replace("1", "√")
    tmp = tmp.replace("0", "×")
    str_detail_list = tmp + str_detail_list.split("]")[1]
    return note_score_absolute_pitch, str_detail_list

'''
编码规则(方便两个字符串匹配)：
1、如果不带“+”或“-”，则不转换；
2、如果带“+”，则转换为大写字母；
3、如果带“-”，则转换为小写字母；
例如：1,2,3,4,5,-6,+1，编码后为：1,2,3,4,5,a,C
'''
def get_encode_pitch_seque(raw_seque):
    # tmp = raw_seque.split(',')
    tmp = raw_seque
    list = []
    for t in tmp:
        if t.find("-") >= 0:
            t = t[0]
            c = get_musical_notation_with_number(t, "small") # 小写字母
        elif t.find("+") >= 0:
            t = t[0]
            c = get_musical_notation_with_number(t, "capital") # 大写字母
        else:
            c = str(t) # 数字转字符
        list.append(c)
    result = ''.join(list)
    return result

def get_all_absolute_pitchs_for_filename(filename,onset_frames,sr,move_gap = 0):
    pitch = get_pitch_by_parselmouth(filename)
    if move_gap != 0:
        pitch.selected_array['frequency'] = pitch.selected_array['frequency'] + move_gap
    all_first_candidate_names,all_second_candidate_names,all_first_candidates,all_second_candidates = get_all_pitch_candidate(pitch,onset_frames.copy(),sr)
    # print("all_second_candidate_names is {},size is {}".format(all_second_candidate_names,len(all_second_candidate_names)))
    result = change_pitch_seque(all_first_candidate_names, all_first_candidates)
    return all_first_candidate_names,result,all_second_candidate_names

def get_all_absolute_pitchs(pitch,onset_frames,sr,move_gap = 0):
    if move_gap != 0:
        pitch.selected_array['frequency'] = pitch.selected_array['frequency'] + move_gap
    all_first_candidate_names,all_second_candidate_names,all_first_candidates,all_second_candidates = get_all_pitch_candidate(pitch,onset_frames.copy(),sr)
    result = change_pitch_seque(all_first_candidate_names, all_first_candidates)
    return all_first_candidate_names,result

def parse_rhythm_code_for_absolute_pitch(rhythm_code):
    code = rhythm_code
    indexs = []
    code = code.replace(";", ',')
    code = code.replace("[", '')
    code = code.replace("]", '')
    tmp = [x for x in code.split(',')]
    return tmp

'''
    绝对音高对节奏起始点进行去重
    1、如果标准音高中没有相临音高没有出现相同的情况，则可以直接删除识别结果中相临重复的音高之一；
    2、如果标准音高中没有相临音高存在相同的情况，则需要对识别结果进行判断，可以不是标准中相临重复的音高删除之一；
'''
def del_the_same_with_absolute_pitch(pitch, all_starts,sr):
    from collections import Counter
    # 获取绝对音高
    # all_first_candidate_names, result = get_all_absolute_pitchs_for_filename(filename, all_starts, sr)
    #
    # #判断绝对音高是否出现相临音高相同的情况
    # check_same_result_absolute_pitch = [i for i in range(len(result)-1) if result[i] == result[i+1]] # 找出绝对音高中相临相同的位置
    # if len(check_same_result_absolute_pitch) == 0:
    #     return onset_types, all_starts
    # else:
    #     base_pitchs =  parse_rhythm_code_for_absolute_pitch(pitch_code)
    #     # 判断标准音高是否出现相临音高相同的情况
    #     check_same_result_base_pitchs = [i for i in range(len(base_pitchs) - 1) if base_pitchs[i] == base_pitchs[i + 1]]
    #     tmp = [base_pitchs[i] for i in check_same_result_base_pitchs]
    #     del_indexs = []
    #     for i in check_same_result_absolute_pitch:
    #         if result[i] not in tmp:
    #             del_indexs.append(i+1)
    #             # del all_starts[i]
    #             # del onset_types[i]
    #     starts_result = [all_starts[i] for i in range(len(all_starts)) if i not in del_indexs]
    #     types_result = [onset_types[i] for i in range(len(onset_types)) if i not in del_indexs]

    # 将起始帧和结束帧换算成时间点
    onset_times = librosa.frames_to_time(all_starts, sr=sr)
    # librosa时间点换算成parselmouth的帧所在位置
    all_starts_parselmouth = [int(o * pitch.n_frames / pitch.duration) for o in onset_times]
    pitch_values = pitch.selected_array['frequency']
    pitch_values_candidate = get_pitch_values(pitch_values)
    del_indexs = []
    for i,s in enumerate(all_starts_parselmouth):
        start = s - 10 if s - 10 > 0 else 0
        end = s + 10 if s + 10 < len(pitch_values_candidate)-1 else len(pitch_values_candidate)-1
        tmp = pitch_values_candidate[start:end]
        if np.std(tmp) == 0:  # 如果该区间音高未有变化，则不是起始点，需要删除
            del_indexs.append(i)
        # if i < len(all_starts_parselmouth) -1:
        #     e = all_starts_parselmouth[i+1]
        #     if Counter(pitch_values_candidate[s:e]).most_common(1)[0][0] < 70:
        #         del_indexs.append(i)
    starts_result = [a for i,a in enumerate(all_starts) if i not in del_indexs]
    return starts_result

def get_pitch_values(pitch_values,check_type='big'):
    if check_type == 'big':
        freqs = [tup for tup in FREQS if tup[0].find('#') < 0]
    elif check_type == 'small':
        freqs = [tup for tup in FREQS ]
    freqs_points = [tup[1] for tup in freqs]
    # freqs_points = [tup[1] for tup in FREQS if tup[0].find('#') < 0]
    pitch_values_candidate = []  # 最靠近的音符
    for p in pitch_values:
        gaps = [np.abs(f - p) for f in freqs_points]
        gap_min = np.min(gaps)
        if np.isnan(gap_min):
            pitch_values_candidate.append(np.nan)
        else:
            p = gaps.index(gap_min)
            pitch_values_candidate.append(freqs_points[p])
    return pitch_values_candidate

'''
根据绝对音高，找出大概率为音符起始点的位置
'''
def get_starts_by_absolute_pitch(pitch,small_or_big,move_gap = 0):
    from collections import Counter
    import scipy.signal as signal
    pitch_values = pitch.selected_array['frequency']
    if move_gap != 0:
        pitch_values = pitch_values + 0
    # b, a = signal.butter(8, 0.2, analog=False)
    # pitch_values = signal.filtfilt(b, a, pitch_values)
    pitch_values = signal.medfilt(pitch_values, 35)
    pitch_values_candidate = get_pitch_values(pitch_values,small_or_big)
    # 将小缝隙补齐
    pitch_values_candidate = smooth_pitch_values_candidate(pitch_values_candidate)

    # 获取连续段的起始点及长度
    starts, lens = get_starts_and_length_for_section(pitch_values_candidate)
    start_frames = [starts[i] for i in range(len(lens)) if lens[i] >= 30]
    # start_frames = [i for i in range(len(pitch_values) - 30) if np.abs(pitch_values_candidate[i] - pitch_values_candidate[i+1]) > 5
    #                and Counter(pitch_values_candidate[i+1:i+30]).most_common(1)[0][1] > 23
    #                and Counter(pitch_values_candidate[i+1:i+30]).most_common(1)[0][0] > 75]
    # print("test_frames is {},size is {}".format(start_frames,len(start_frames)))
    if len(start_frames) == 0:
        return [],[]
    first_frame = start_frames[0]
    start_frames = [start_frames[i] for i in range(1,len(start_frames)) if start_frames[i] - start_frames[i-1] > 20]
    start_frames.append(first_frame)
    start_frames.sort()
    # print("test_frames is {},size is {}".format(start_frames, len(start_frames)))
    onset_times = [pitch.duration * t / pitch.n_frames for t in start_frames]
    return start_frames,onset_times

'''
根据绝对音高，找出大概率为音符起始点的位置(短节奏的，例如：250)
'''
def get_short_starts_by_absolute_pitch(pitch,small_or_big,move_gap = 0):
    from collections import Counter
    import scipy.signal as signal
    pitch_values = pitch.selected_array['frequency']
    if move_gap != 0:
        pitch_values = pitch_values + 0
    # b, a = signal.butter(8, 0.2, analog=False)
    # pitch_values = signal.filtfilt(b, a, pitch_values)
    pitch_values = signal.medfilt(pitch_values, 35)
    pitch_values_candidate = get_pitch_values(pitch_values,small_or_big)
    # 将小缝隙补齐
    pitch_values_candidate = smooth_pitch_values_candidate(pitch_values_candidate)

    # 获取连续段的起始点及长度
    starts,lens = get_starts_and_length_for_section(pitch_values_candidate)
    start_frames = [starts[i] for i in range(len(lens)) if lens[i] < 30 and lens[i] > 10]
    # start_frames = [i for i in range(len(pitch_values) - 30) if np.abs(pitch_values_candidate[i] - pitch_values_candidate[i+1]) > 5
    #                and Counter(pitch_values_candidate[i+1:i+30]).most_common(1)[0][1] > 15 and Counter(pitch_values_candidate[i+1:i+30]).most_common(1)[0][1] <= 20
    #                and Counter(pitch_values_candidate[i+1:i+30]).most_common(1)[0][0] > 75]
    # print("test_frames is {},size is {}".format(start_frames,len(start_frames)))
    if len(start_frames) > 0:
        first_frame = start_frames[0]
        start_frames = [start_frames[i] for i in range(1,len(start_frames)) if start_frames[i] - start_frames[i-1] > 10]
        start_frames.append(first_frame)
        start_frames.sort()
        # print("test_frames is {},size is {}".format(start_frames, len(start_frames)))
        onset_times = [pitch.duration * t / pitch.n_frames for t in start_frames]
    else:
        onset_times = []
    return start_frames,onset_times

def get_all_starts_by_absolute_pitch(pitch,small_or_big = 'big'):
    start_frames, onset_times = get_starts_by_absolute_pitch(pitch,small_or_big)
    short_start_frames, short_onset_times = get_short_starts_by_absolute_pitch(pitch,small_or_big)
    all_start_frames = start_frames + short_start_frames
    all_onset_times = onset_times + short_onset_times
    gap_start_frames, gap_onset_times = get_gap_by_diff_on_pitch(pitch,small_or_big)
    for i,g in enumerate(gap_start_frames):
        tmp = [np.abs(g - o) for o in all_start_frames]
        if np.min(tmp) > 15:
            all_start_frames.append(g)
            all_onset_times.append(gap_onset_times[i])
    all_start_frames.sort()
    all_onset_times.sort()
    return all_start_frames,all_onset_times

def get_high_believe_starts_by_absolute_pitch(pitch,small_or_big = 'big'):
    all_start_frames, all_onset_times = get_starts_by_absolute_pitch(pitch,small_or_big)
    gap_start_frames, gap_onset_times = get_gap_by_diff_on_pitch(pitch,small_or_big)
    for i,g in enumerate(gap_start_frames):
        tmp = [np.abs(g - o) for o in all_start_frames]
        if np.min(tmp) > 15:
            all_start_frames.append(g)
            all_onset_times.append(gap_onset_times[i])
    all_start_frames.sort()
    all_onset_times.sort()
    import scipy.signal as signal
    pitch_values = pitch.selected_array['frequency']
    pitch_values = signal.medfilt(pitch_values, 45)
    # 找出第一个不为0、不为nan的元素的位置
    first_position = [i for i in range(len(pitch_values)) if pitch_values[i] >= 50 and not np.isnan(pitch_values[i])][0]
    if all_start_frames[0] - first_position > 12:
        all_start_frames.append(first_position)
        all_onset_times.append(round(first_position/100,2))
    all_start_frames.sort()
    all_onset_times.sort()
    return all_start_frames,all_onset_times

'''
    音高轨迹线前后两个点间隔大于8，则判定为起始点
'''
def get_gap_by_diff_on_pitch(pitch,small_or_big):
    from collections import Counter
    import scipy.signal as signal
    pitch_values = pitch.selected_array['frequency']
    pitch_values = signal.medfilt(pitch_values, 35)
    pitch_values_candidate = get_pitch_values(pitch_values, small_or_big)
    # 将小缝隙补齐
    pitch_values_candidate = smooth_pitch_values_candidate(pitch_values_candidate)
    tmp = [i for i in range(len(pitch_values)-1) if np.abs(pitch_values[i] - pitch_values[i+1]) > 8 and pitch_values[i] > 70 and pitch_values[i+1] > 70]
    #判断每个起始点之后的音高轨迹线连续长度,连续长度大于15记为正常的起始点。
    start_frames = []
    for s in tmp:
        if Counter(pitch_values_candidate[s:s + 20]).most_common(1)[0][1] > 15 and Counter(pitch_values_candidate[s:s + 20]).most_common(1)[0][0] > 50 :
            start_frames.append(s)
    onset_times = [pitch.duration * t / pitch.n_frames for t in start_frames]
    return start_frames,onset_times

'''
获取连续段的起始点及长度
'''
def get_starts_and_length_for_section(pitch_values_candidate):
    starts =[]
    lens = []
    jump_point = 1
    end = 1
    for i in range(len(pitch_values_candidate)-10):
        tmp = pitch_values_candidate[i]
        start = i
        if i > jump_point and tmp > 70:
            for j in range(i+1,len(pitch_values_candidate)):
                if pitch_values_candidate[j] == tmp:
                    end = j
                else:
                    if end > start:
                        starts.append(start)
                        lens.append(end - start)
                        jump_point = end
                    break
    return starts,lens

def get_starts_by_absolute_pitch_with_filename(filename,small_or_big='big'):
    pitch = get_pitch_by_parselmouth(filename)
    start_frames, start_times = get_starts_by_absolute_pitch(pitch,small_or_big)
    return start_frames, start_times

def smooth_pitch_values_candidate(pitch_values_candidate):
    # 将小缝隙补齐
    for i in range(len(pitch_values_candidate) - 30):
        if np.std(pitch_values_candidate[i:i + 10]) > 1 and pitch_values_candidate[i] == \
                pitch_values_candidate[i + 10] and pitch_values_candidate[i] == pitch_values_candidate[i + 11]:
            # print(pitch_values_candidate[i+1:i+6])
            p = pitch_values_candidate[i]
            pitch_values_candidate[i + 1:i + 11] = [p for i in range(i + 1, i + 11)]
        if np.std(pitch_values_candidate[i:i + 5]) > 1 and pitch_values_candidate[i] == pitch_values_candidate[i + 5]:
            # print(pitch_values_candidate[i+1:i+6])
            p = pitch_values_candidate[i]
            pitch_values_candidate[i + 1:i + 5] = [p for i in range(i + 1, i + 5)]
    # # 获取连续段的起始点及长度
    # starts, lens = get_starts_and_length_for_section(pitch_values_candidate)
    # jump_point = 1
    # for i in range(len(pitch_values_candidate) - 30):
    #     tmp = pitch_values_candidate[i]
    #     if i > jump_point and tmp > 70 and pitch_values_candidate[i+1] != pitch_values_candidate[i] and pitch_values_candidate[i+1] > 70 : # 如果相临两个前后不相同，即为该小缝隙的开始点
    #         for j in range(i + 1, i+25):
    #             if pitch_values_candidate[j] == tmp: #如果有相同的点，即为该小缝隙的结束点
    #                 end = j
    #                 p = pitch_values_candidate[i]
    #                 pitch_values_candidate[i + 1:end] = [p for i in range(i + 1,end)]
    #                 jump_point = j
    #                 break
    #             # else:
    #             #     if j == i+24: #如果循环到最后一个点还没有该小缝隙的结束点，则跳过
    #             #         jump_point = j


    return pitch_values_candidate

'''
平移算法获取匹配度（即最大公共子序列长度最大）最高的相对音高
'''
def get_best_relative_pitch(pitch,pitch_code):
    pitch_values = pitch.selected_array['frequency']
    pitch_values_candidate = get_pitch_values(pitch_values,'big')
    # 将小缝隙补齐
    pitch_values_candidate = smooth_pitch_values_candidate(pitch_values_candidate)

    # 音符起始点的位置
    start_frames, start_onset_times = get_all_starts_by_absolute_pitch(pitch)
    # 根据音高轨迹线找出每个音符
    pitch_names, freqs = get_pitch_names_and_freqs_on_starts(pitch_values_candidate, start_frames)
    # 音符序列“+”“-”编码
    result = change_pitch_seque(pitch_names, freqs)
    print("result is {},size is {}".format(result, len(result)))
    pitch_code_for_absolute_pitch = parse_rhythm_code_for_absolute_pitch(pitch_code)
    note_score_absolute_pitch, lcseque, str_detail_list, detail_list, raw_positions = get_matched_detail_absolute_pitch(pitch_code_for_absolute_pitch, result)
    print("lcseque is {},size is {}".format(lcseque, len(lcseque)))

def get_pitch_names_and_freqs_on_starts(pitch_values_candidate,start_frames):
    freqs = []
    pitch_names = []
    for s in start_frames:
        freq = pitch_values_candidate[s+1]
        pitch_name = get_pitch_name(freq)
        pitch_names.append(pitch_name)
        freqs.append(freq)
    return pitch_names,freqs

def get_start_and_end_with_parselmouth(filename):
    pitch = get_pitch_by_parselmouth(filename)
    pitch_values = pitch.selected_array['frequency']
    import scipy.signal as signal
    pitch_values = signal.medfilt(pitch_values, 35)
    # 找出第一个不为0、不为nan的元素的位置
    first_position = [i for i in range(len(pitch_values)) if pitch_values[i] >= 50 and not np.isnan(pitch_values[i])][0]
    # 找出最后一个不为0、不为nan的元素的位置
    last_position = [i for i in range(len(pitch_values)) if pitch_values[i] >= 50 and not np.isnan(pitch_values[i])][-1]
    first_time = first_position * pitch.dt + pitch.t1
    last_time = last_position * pitch.dt + pitch.t1
    return first_time,last_time,last_position - first_position

'''
获取所有平移后的音高类型
'''
def get_all_numbered_musical_notation_by_moved(first_base_numbered_notation,all_notations,test_times,end_time=None):

    # all_times = librosa.frames_to_time(all_frames)
    # all_times = list(all_times)
    all_times = test_times.copy()
    if end_time is None:
        all_times.append(all_times[-1] + 0.2) # 添加一个结束点
    else:
        all_times.append(end_time)
    result = []
    first_notation = ''
    first_freq = 0
    for a in all_notations:
        if a is not None:
            first_notation = a
            first_freq = get_freq_by_notation_name(a)
            break
    if first_notation == '':
        return result
    #将标准序列中首个数字音高转换为字母
    first_base_numbered_notation_str = str(first_base_numbered_notation)
    first_base_notation = get_musical_notation_with_number(first_base_numbered_notation_str,'capital')
    #获取识别首音高字母的位置
    anchor_point = [i for i, p in enumerate(PITCH_NAMES) if i >= 12 and i <= 23 and p.find(first_notation[0]) >= 0]
    #计算平移步长
    steps = [i-anchor_point[0] for i,s in enumerate(PITCH_NAMES) if s.find(first_base_notation) >= 0]
    #获取最小平移步长
    min_steps = [s for s in steps if np.abs(s) == np.min(np.abs(steps))]
    step = min_steps[0]

    detail = []
    for i,a in enumerate(all_notations):
        if a is None:
            result.append(None)
            detail.append((None,all_times[i],all_times[i+1]))
        else:
            numbered_notation = get_numbered_musical_notation_by_moved(a,step)
            if numbered_notation.find("#") >= 0:
                numbered_notation = numbered_notation[0]
            a_freq = get_freq_by_notation_name(a)
            if int(numbered_notation) == first_base_numbered_notation and a_freq > first_freq: # 如果当前频率大于基准频率，且音高相同，则需要上调一个音分
                numbered_notation = str((int(numbered_notation) + 1)%7)
            elif int(numbered_notation) == first_base_numbered_notation and a_freq < first_freq: # 如果当前频率大于基准频率，且音高相同，则需要下调一个音分
                numbered_notation = str(int(numbered_notation) - 1) if int(numbered_notation) - 1 != 0 else '7'
            result.append(numbered_notation)
            detail.append((numbered_notation, all_times[i], all_times[i + 1]))
    return result,detail

def check_moved_step(standard_notations,standard_notation_times,all_first_candidate_names,test_times,end_time=None):
    if end_time is None:
        test_times.append(test_times[-1] + 0.2) # 添加一个结束点
    else:
        test_times.append(end_time)
    first_standard_notation = int(standard_notations.split(",")[0][0])
    numbered_notations, numbered_notations_detail = get_all_numbered_musical_notation_by_moved(first_standard_notation,
                                                                                               all_first_candidate_names,
                                                                                               test_times)
    lcseque, standard_positions, test_positions = get_lcseque_and_position_with_time_offset(standard_notations, numbered_notations, standard_notation_times, test_times)
    return lcseque,numbered_notations

def find_the_cut_point(standard_notations,standard_notation_times,all_first_candidate_names,test_times,end_time=None):
    best_lcseque_len = 0
    best_numbered_notations = None
    type = 0

    # 第一种情况：多2个音符（标准序列的第1个与测试序列的第3个对比）
    #pass

    # 第二种情况：多1个音符（标准序列的第1个与测试序列的第2个对比）
    all_first_candidate_names_modified = all_first_candidate_names[1:]
    test_times_modified = test_times[1:]
    lcseque_second, numbered_notations_second = check_moved_step(standard_notations, standard_notation_times, all_first_candidate_names_modified, test_times_modified, end_time=None)
    if len(lcseque_second) > best_lcseque_len:
        best_numbered_notations = ['1'] + numbered_notations_second
        best_lcseque_len = len(lcseque_second)
        type = 2

    # 第三种情况：对齐
    lcseque_third, numbered_notations_third = check_moved_step(standard_notations, standard_notation_times, all_first_candidate_names, test_times, end_time=None)
    if len(lcseque_third) > best_lcseque_len:
        best_numbered_notations = numbered_notations_third
        best_lcseque_len = len(lcseque_third)
        type = 3

    # 第四种情况：少1个音符（标准序列的第2个与测试序列的第1个对比）
    surplus_note = standard_notations[0]
    surplus_time = standard_notation_times[0]
    standard_notations_modified = standard_notations[1:]
    standard_notation_times_modified = standard_notation_times[1:]
    lcseque_fourth, numbered_notations_fourth = check_moved_step(standard_notations_modified, standard_notation_times_modified, all_first_candidate_names, test_times, end_time=None)
    if len(lcseque_fourth) > best_lcseque_len:
        best_numbered_notations = [surplus_note] + numbered_notations_fourth
        best_lcseque_len = len(lcseque_fourth)
        type = 4

        # 第五种情况：少2个音符（标准序列的第3个与测试序列的第1个对比）
    # pass
    return best_numbered_notations,type,best_lcseque_len

def find_best_numbered_notations(standard_notations,standard_notation_times,all_first_candidate_names,test_times,end_time=None):
    best_lcseque_len = 0
    best_numbered_notations = None
    type = 0

    if end_time is None:
        test_times.append(test_times[-1] + 0.2) # 添加一个结束点
    else:
        test_times.append(end_time)

    for step in range(1,8):
        first_standard_notation = step
        numbered_notations, numbered_notations_detail = get_all_numbered_musical_notation_by_moved(first_standard_notation, all_first_candidate_names, test_times)
        lcseque, standard_positions, test_positions = get_lcseque_and_position_with_time_offset(standard_notations, numbered_notations, standard_notation_times, test_times)
        if len(lcseque) > best_lcseque_len:
            best_numbered_notations = numbered_notations
            best_lcseque_len = len(lcseque)
            type = step
    return best_numbered_notations,type,best_lcseque_len

def get_all_numbered_notation_and_offset(pitch,onset_frames,sr=22050):
    from collections import Counter
    # 将起始帧和结束帧换算成时间点
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    # librosa时间点换算成parselmouth的帧所在位置
    all_starts_parselmouth = [int(o * pitch.n_frames / pitch.duration) for o in onset_times]
    onset_frames = all_starts_parselmouth
    pitch_values = pitch.selected_array['frequency']
    pitch_values = signal.medfilt(pitch_values, 35)
    small_or_big = 'small'
    pitch_values_candidate = get_pitch_values(pitch_values, small_or_big)
    # 将小缝隙补齐
    pitch_values_candidate = smooth_pitch_values_candidate(pitch_values_candidate)
    last_position = int((pitch.duration - 0.05) * 100)
    onset_frames.append(last_position)
    all_first_candidates = []
    all_first_candidate_names = []
    all_offset_types = []
    # 以当前起始点为起点，下一个起始点为终点，获取该段节奏上的音高
    freqs = [tup for tup in FREQS]  # 筛选标准音高序列
    notations = [tup[0] for tup in freqs]
    for i in range(len(onset_frames) - 1):
        start_frame = onset_frames[i]
        end_frame = onset_frames[i + 1]
        try:
            candidate_notation_tmp = pitch_values_candidate[start_frame:end_frame]
            candidate_notation_tmp = [p for p in candidate_notation_tmp if p > 50]
            pitch_tmp = pitch_values[start_frame:end_frame]
            if len(candidate_notation_tmp) == 0:
                all_first_candidate_names.append(None)
                all_first_candidates.append(None)
                all_offset_types.append(None)
                continue
            # 找出list中出现最多的元素
            res = Counter(candidate_notation_tmp)
            if len(pitch_values_candidate) == 0:  # 如果整个音高序列为空
                return np.nan, np.nan,np.nan
            most_notation_freq = res.most_common(1)[0][0]  # 第一判定音高
            first_candidate_name = most_notation_freq if np.isnan(most_notation_freq) else get_pitch_name(most_notation_freq)
            small_list = [i for i,p in enumerate(pitch_tmp) if p < most_notation_freq and p > 50]
            more_list = [i for i,p in enumerate(pitch_tmp) if p > most_notation_freq and p > 50]
            if len(small_list) < len(more_list):
                offset = 'more'
            else:
                offset = 'less'
            all_first_candidate_names.append(first_candidate_name)
            all_first_candidates.append(most_notation_freq)
            all_offset_types.append(offset)
        except Exception:
            print(str(Exception))
            # print(str(e))
            # print(repr(e))
            # print( e.message)
            pass
    return all_first_candidate_names, all_first_candidates, all_offset_types

'''
如果前后两个音高数字相差不大，且频率相差也不大，可以大概率认为是伪起始点
'''
def check_all_starts(pitch,onset_frames,test_onset_times):
    pitch_values = pitch.selected_array['frequency']
    pitch_values = signal.medfilt(pitch_values, 35)
    # librosa时间点换算成parselmouth的帧所在位置
    all_starts_parselmouth = [int(o * pitch.n_frames / pitch.duration) for o in test_onset_times]
    all_first_candidate_names, all_first_candidates, all_offset_types = get_all_numbered_notation_and_offset(pitch,onset_frames)
    print("check_all_starts all_first_candidate_names is {},size is {}".format(all_first_candidate_names,len(all_first_candidate_names)))
    numbered_musical_notations = []
    for a in all_first_candidate_names:
        if a[1] == "#":
            s = a[0:2]
        else:
            s = a[0]
        anchor_point = [i for i, p in enumerate(PITCH_NAMES) if i >= 12 and i <= 23 and p.find(s) >= 0]
        numbered_musical_notations.append(anchor_point[0])

    pitch_values_on_test_frames = [np.abs(round(pitch_values[t - 2] - pitch_values[t + 2], 2)) for t in all_starts_parselmouth]
    print("check_all_starts pitch_values_on_test_frames is {},size is {}".format(pitch_values_on_test_frames, len(pitch_values_on_test_frames)))
    result = [onset_frames[0]] # 添加第一个
    onset_times = [test_onset_times[0]]  # 添加第一个
    for i,s in enumerate(numbered_musical_notations):
        if i > 0:
            s1 = int(numbered_musical_notations[i-1]) #前一个音高数字
            s2 = int(numbered_musical_notations[i]) #当前音高数字
            if np.abs(s1 -s2) <= 1 and np.abs(pitch_values_on_test_frames[i]) < 9: # 如果前后两个音高数字相差不大，且频率相差也不大，可以大概率认为是伪起始点
                continue
            else:
                result.append(onset_frames[i])
                onset_times.append(test_onset_times[i])
    return result,onset_times

'''
通过科大讯飞语音识别接口获取歌词信息
'''
def get_result_from_xfyun(filename):

    all_message, all_detail = get_iat_result(filename)
    return all_message, all_detail

'''
    根据幅度较大的波谷判断为大概率的起始点
'''
def get_starts_by_parselmouth_rms(intensity,pitch,standard_notation_time_diff_min):
    values = intensity.values.T.copy()
    values = list(values)
    values = [v[0] for v in values]  #原始幅度
    values_medfilt = signal.medfilt(values, 11)  #滤波后幅度
    values_gap = [values_medfilt[i] - values[i] for i in range(len(values_medfilt))]
    values_len = len(values)

    pitch_values = pitch.selected_array['frequency']
    pitch_values = signal.medfilt(pitch_values, 35)

    #处理异常
    if len([i for i in range(len(pitch_values)) if pitch_values[i] >= 50 and not np.isnan(pitch_values[i])]) < 1:
        return [],[]

    # 找出第一个不为0、不为nan的元素的位置
    first_position = [i for i in range(len(pitch_values)) if pitch_values[i] >= 50 and not np.isnan(pitch_values[i])][0]
    first_position = int(first_position*values_len/len(pitch_values))
    # 找出最后一个不为0、不为nan的元素的位置
    last_position = [i for i in range(len(pitch_values)) if pitch_values[i] >= 50 and not np.isnan(pitch_values[i])][-1]
    last_position = int(last_position * values_len / len(pitch_values))

    #判断条件：1、幅度差值大于1.5；  2、位置开始点和结束点之间；  3、起始点之后15之后要有音高线；
    must_starts = [i for i,v in enumerate(values_medfilt) if values_gap[i] > 1.6 and i > first_position and i < last_position - 40]
    # must_starts = [s for s in must_starts if np.mean(pitch_values[s:s+15]) > 70]

    #处理异常
    if len(must_starts) < 1:
        return [],[]

    #相临太密的取幅度减值较大的
    result = [must_starts[0]]
    if standard_notation_time_diff_min > 0.3:
        threshold = 30
    else:
        threshold = int(standard_notation_time_diff_min * 100 / 2)
    for i in range(1,len(must_starts)):
        c = must_starts[i]
        c_on_pitch_values = int(c *len(pitch_values) / values_len)
        if np.mean(pitch_values[c_on_pitch_values:c_on_pitch_values+15]) > 70:
            if c - result[-1] < threshold:
                selected = c if values_gap[c] > values_gap[result[-1]] else result[-1]
                result[-1] = selected
            else:
                result.append(c)
    onset_times = [pitch.duration * t / values_len for t in result]
    result = [int(o * 100) for o in onset_times]
    return result,onset_times


def merge_candidate(pitch_values,pitch_values_candidate):
    pass

def merge_times_from_iat_plm_rms(iat_times,plm_times,rms_times,standard_notation_time_diff_min):
    tmp = plm_times + rms_times
    tmp = [round(t,2) for t in tmp]
    tmp.sort()
    result = iat_times

    if standard_notation_time_diff_min > 0.3:
        threshold = 0.2
    else:
        threshold = round(standard_notation_time_diff_min / 2, 2)
    for s in tmp:
        offset = [np.abs(t - s) for t in result]
        if np.min(offset) > threshold:
            result.append(s)
    result.sort()
    return result

def get_notation_detail_by_times(numbered_notations_detail,start_time,end_time):
    selected_numbered_notations_detail = [tup for tup in numbered_notations_detail if tup[2] > start_time and tup[1] < end_time] # 开始点和结束点落在区间内的都算
    return selected_numbered_notations_detail

def score_all(filename, standard_kc,standard_kc_time, standard_notations, standard_notation_time):
    # ###### 语音识别===========
    if filename.find(".wav") >= 0:
        wav2pcm(filename)
    pcmfile = filename.split(".wav")[0] + ".pcm"
    test_kc, kc_detail = get_result_from_xfyun(pcmfile)
    if len(kc_detail) == 0:
        return 0,0,0,0,0,0,'语音识别未能正常返回结果','语音识别未能正常返回结果','语音识别未能正常返回结果','语音识别未能正常返回结果','语音识别未能正常返回结果'
    detail_time = [round((value) / 100, 2) for value in kc_detail.keys() if value > 0]
    pitch = get_pitch_by_parselmouth(filename)
    end_time = pitch.duration
    small_or_big = 'small'
    test_frames, test_onset_times = get_all_starts_by_absolute_pitch(pitch, small_or_big)

    #处理异常
    if len(test_frames) < 1 or len(test_onset_times) < 1:
        return 0, 0, 0, 0, 0, 0, '音高识别未能正常返回结果，请检查音频数据是否正常', '音高识别未能正常返回结果，请检查音频数据是否正常', '音高识别未能正常返回结果，请检查音频数据是否正常', '音高识别未能正常返回结果，请检查音频数据是否正常', '音高识别未能正常返回结果，请检查音频数据是否正常'

    snd = parselmouth.Sound(filename)
    intensity = snd.to_intensity()

    standard_notation_time_diff_min = np.min(np.diff(standard_notation_time))
    starts_by_parselmouth_rms, starts_by_parselmouth_rms_times = get_starts_by_parselmouth_rms(intensity, pitch,standard_notation_time_diff_min)
    # 平移语音识别的时间点
    detail_time = [t - (detail_time[0] - test_onset_times[0]) for t in detail_time]
    test_times = merge_times_from_iat_plm_rms(detail_time, test_onset_times, starts_by_parselmouth_rms_times,standard_notation_time_diff_min)
    merge_frames = librosa.time_to_frames(test_times)
    all_first_candidate_names, all_first_candidates, all_offset_types = get_all_numbered_notation_and_offset(pitch,merge_frames)
    first_standard_notation = int(standard_notations.split(",")[0][0])
    # print("========== all_first_candidate_names size is {},detail is {}".format(len(all_first_candidate_names),all_first_candidate_names))
    # print("========== standard_notations size is {},detail is {}".format(len(standard_notations),standard_notations))
    # print("========== standard_notation_time size is {},detail is {}".format(len(standard_notation_time),standard_notation_time))
    # print("========== test_times size is {},detail is {}".format(len(test_times), test_times))
    # numbered_notations,numbered_notations_detail = get_all_numbered_musical_notation_by_moved(first_standard_notation,all_first_candidate_names,test_times)
    standard_notations_list = standard_notations.split(',')
    standard_notations_list = [s[0] for s in standard_notations_list]
    standard_notations_list = ''.join(standard_notations_list)
    best_numbered_notations, type, best_lcseque_len = find_best_numbered_notations(standard_notations_list, standard_notation_time,
                                                                         all_first_candidate_names, test_times,
                                                                         end_time=None)
    numbered_notations = best_numbered_notations
    # print("========== numbered_notations size is {},detail is {}".format(len(numbered_notations), numbered_notations))
    kc_express_total_score, fluency_total_score,kc_express_sscore_detail,fluency_sscore_detail = 0,0,'',''
    # total_score, pitch_total_score, notation_duration_total_score, kc_duration_total_score, pitch_score_detail, notation_duration_score_detail, kc_rhythm_sscore_detail = get_all_scores(standard_kc,
    #                                                                                                                     standard_kc_time,
    #                                                                                                                     test_kc,
    #                                                                                                                     standard_notations,
    #                                                                                                                     numbered_notations,
    #                                                                                                                     standard_notation_time,
    #                                                                                                                     test_times,
    #                                                                                                                     kc_detail,
    #                                                                                                                     end_time)
    # return total_score, pitch_total_score, notation_duration_total_score, kc_duration_total_score,pitch_score_detail, notation_duration_score_detail, kc_rhythm_sscore_detail,
    total_score, pitch_total_score, notation_duration_total_score, kc_duration_total_score,kc_express_total_score,fluency_total_score, pitch_score_detail, notation_duration_score_detail, kc_rhythm_sscore_detail,kc_express_sscore_detail,fluency_sscore_detail = get_all_scores_with_5(
        standard_kc,
        standard_kc_time,
        test_kc,
        standard_notations,
        numbered_notations,
        standard_notation_time,
        test_times,
        kc_detail, end_time,intensity)
    return total_score, pitch_total_score, notation_duration_total_score, kc_duration_total_score,kc_express_total_score,fluency_total_score, pitch_score_detail, notation_duration_score_detail, kc_rhythm_sscore_detail,kc_express_sscore_detail,fluency_sscore_detail
def score_all_by_st(filename, standard_kc,standard_kc_time, standard_notations, standard_notation_time):
    # ###### 语音识别===========
    if filename.find(".wav") >= 0:
        wav2pcm(filename)
    pcmfile = filename.split(".wav")[0] + ".pcm"
    test_kc, kc_detail = get_result_from_xfyun(pcmfile)
    if len(kc_detail) == 0:
        return 0,0,0,0,'语音识别未能正常返回结果','语音识别未能正常返回结果','语音识别未能正常返回结果'
    kyes = list(kc_detail.keys())
    detail_time = [round((value) / 100, 2) for value in kyes if value > 0]
    pitch = get_pitch_by_parselmouth(filename)
    end_time = pitch.duration
    small_or_big = 'small'
    test_frames, test_onset_times = get_all_starts_by_absolute_pitch(pitch, small_or_big)

    snd = parselmouth.Sound(filename)
    intensity = snd.to_intensity()
    starts_by_parselmouth_rms, starts_by_parselmouth_rms_times = get_starts_by_parselmouth_rms(intensity, pitch)
    test_times = merge_times_from_iat_plm_rms(detail_time, test_onset_times, starts_by_parselmouth_rms_times)
    merge_frames = librosa.time_to_frames(test_times)
    all_first_candidate_names, all_first_candidates, all_offset_types = get_all_numbered_notation_and_offset(pitch,merge_frames)
    print("3 all_first_candidate_names is {},size is {}".format(all_first_candidate_names,len(all_first_candidate_names)))
    numbered_notations,numbered_notations_detail = get_all_numbered_musical_notation_by_moved(3,all_first_candidate_names,test_times,end_time)
    total_score, pitch_total_score, notation_duration_total_score, kc_duration_total_score, pitch_score_detail, notation_duration_score_detail, kc_rhythm_sscore_detail = get_all_scores_by_st(standard_kc, standard_kc_time, standard_notations, numbered_notations, standard_notation_time,
                         test_times, kc_detail, end_time)
    return total_score, pitch_total_score, notation_duration_total_score, kc_duration_total_score, pitch_score_detail, notation_duration_score_detail, kc_rhythm_sscore_detail

def get_troughs_on_intensity(intensity):
    troughs = [i for i in range(10,len(intensity)-10) if intensity[i-1] > intensity[i] and intensity[i+1] >= intensity[i]] # 获取振幅上的波谷点
    values = signal.medfilt(intensity, 15)
    indexs = [i for i in range(10,len(intensity)-10) if values[i] - intensity[i] > 1.5]
    troughs = [t for t in troughs if t in indexs]
    tmp = troughs[0]
    for t in troughs:
        en = intensity[t]
        if en > 42:
            tmp = t
            break
    troughs = [t for t in troughs if t >= tmp]

    #左右跨点
    gap = 8e-1
    offtake_points = [i for i in range(10,len(intensity)-10) if (np.abs(values[i] - intensity[i]) < gap and values[i+1] > intensity[i+1]) or (np.abs(values[i] - intensity[i]) < gap and values[i+1] <= intensity[i+1])]
    offtake_points_before = [[o for o in offtake_points if o<t][-1] for t in troughs]
    offtake_points_after = [[o for o in offtake_points if o > t][0] for t in troughs]
    selected_offtake_points = offtake_points_before + offtake_points_after
    selected_offtake_points.sort()
    durations = []
    for t in troughs:
        start = [o for o in offtake_points if o<t][-1]
        end = [o for o in offtake_points if o > t][0]
        duration = end - start
        durations.append(duration)

    # offtake_points = [i for i in range(10,len(intensity)-10) if np.abs(values[i] - intensity[i]) < gap and values[i+1] > intensity[i+1]]
    # offtake_points = [i for i in range(10,len(intensity)-10) if np.abs(values[i] - intensity[i]) < gap and values[i+1] <= intensity[i+1]]
    # offtake_points = [i for i in range(10,len(intensity)-10) if values[i] - intensity[i] == 0]
    return troughs,selected_offtake_points,durations
if __name__ == "__main__":
    raw_seque = ['1','1+','1-','2','2+','2-','3','3+','3-','4','4+','4-','5','5+','5-','6','6+','6-','7','7+','7-']
    test = get_encode_pitch_seque(raw_seque)
    print(test)
    freqs_names = [tup[0] for tup in FREQS]
    print(freqs_names)
    filename, notation = 'F:/项目/花城音乐项目/样式数据/20.04.08MP3/2段词-标准1648.wav', '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
    # filename,notation = 'F:/项目/花城音乐项目/样式数据/20.04.08MP3/2段词-标准4882.wav','3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
    # filename,notation = 'F:/项目/花城音乐项目/样式数据/20.04.08MP3/2段词-标准1681.wav','3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
    # filename,notation = 'F:/项目/花城音乐项目/样式数据/20.04.08MP3/2段词-不标准.wav','3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
    filename, notation = 'F:/项目/花城音乐项目/样式数据/20.04.08MP3/2段词-不标准1648.wav', '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
    # filename, notation = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋3罗（80）.wav', '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
    # filename, notation = 'F:/项目/花城音乐项目/样式数据/12.05MP3/wav/旋律/H-5.wav', '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
    # filename, notation = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/两只老虎20190624-1089.wav', '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
    # filename, notation = 'F:/项目/花城音乐项目/样式数据/20.03.26MP3/wav/CI1.wav', '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
    # filename, notation = 'F:/项目/花城音乐项目/样式数据/20.04.08MP3/2段词-标准1450.wav', '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
    # filename, notation = 'F:/项目/花城音乐项目/样式数据/20.04.08MP3/2段词-标准1050.wav', '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
    # filename, notation = 'F:/项目/花城音乐项目/样式数据/20.04.08MP3/2段词-不标准1050.wav', '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
    # filename, notation = 'F:/项目/花城音乐项目/样式数据/20.04.29MP3/wav/CI1.wav', '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
    # filename, notation = 'F:/项目/花城音乐项目/样式数据/20.04.29MP3/wav/CI2.wav', '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
    filename, notation = 'F:/项目/花城音乐项目/样式数据/20.04.29MP3/wav/test1-1547.wav', '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
    filename, notation = 'F:/项目/花城音乐项目/样式数据/20.04.29MP3/wav/test2-1548.wav', '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
    filename, notation = 'F:/项目/花城音乐项目/样式数据/20.04.29MP3/wav/test3-1547.wav', '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
    # filename, notation = 'F:/项目/花城音乐项目/样式数据/20.05.01MP3/wav/3927-1548.wav', '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
    # filename, notation = 'F:/tmp/m-1648.wav', '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
    # filename, notation = 'F:/tmp/m-4880.wav', '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
    standard_kc = '喜爱春天的人儿是心地纯洁的人像紫罗兰花儿一样是我知心朋友'
    # standard_kc = '喜爱夏天的人儿是意志坚强的人像冲打岩礁的波浪一样是我敬爱的父亲'
    standard_kc_time = [0, 1, 2, 3, 3.5, 4, 5, 6, 8, 9, 10, 11, 11.5, 12, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,26.5, 27, 28, 32]
    # standard_kc_time = [0, 1, 2, 3, 3.5, 4, 5, 6, 8, 9, 10, 11, 11.5, 12, 16, 17, 17.5,18, 19,19.5, 20, 21, 22, 23, 24, 25, 26,26.5,26.75, 27, 28, 32]
    standard_notations = '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
    standard_notation_time = [0, 1, 1.5, 2, 3, 3.5, 4, 5, 6, 8, 9, 9.5, 10, 10.5, 11, 11.5, 12, 16, 17, 17.5, 18, 19,19.5, 20, 21, 21.5, 22, 23, 24, 25, 26, 26.5, 27, 27.5, 28, 32]
    # standard_notation_time = [0,0.6818181818181817,1.0227272727272734,1.363636363636365,2.0454545454545467,2.3863636363636385,2.72727272727273,3.409090909090912,4.0909090909090935,5.454545454545459,6.136363636363642,6.477272727272732,6.818181818181822,7.159090909090912,7.500000000000002,7.840909090909092,8.181818181818182,10.909090909090908,11.590909090909092,11.931818181818182,12.272727272727272,12.954545454545455,13.295454545454545,13.636363636363635,14.318181818181818,14.659090909090908,14.999999999999998,15.681818181818182,16.363636363636367,17.045454545454547,17.727272727272734,18.06818181818182,18.409090909090914,18.75,19.090909090909093,21.81818181818182]
    # # ========================= 2020.05.09 1701 ===================
    # filename = 'F:/项目/花城音乐项目/样式数据/20.05.01MP3/dbg/1701/seg1.wav'
    # # filename = 'F:/项目/花城音乐项目/样式数据/20.05.01MP3/dbg/1701/seg1-1732.wav'
    # standard_kc = '喜爱春天的人儿是心地纯洁的人像紫罗兰花儿一样是我知心朋友'
    # # standard_kc = '喜爱夏天的人儿是意志坚强的人像冲打岩礁的波浪一样是我敬爱的父亲'
    # standard_kc_time = [0,1,2,3,3.5,4,5,6,8,9,10,11,11.5,12,16,17,18,19,20,21,22,23,24,25,26,26.5,27,28,32]
    # standard_notations = '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6- '
    # standard_notation_time = [0,1,1.5,2,3,3.5,4,5,6,8,9,9.5,10,10.5,11,11.5,12,16,17,17.5,18,19,19.5,20,21,21.5,22,23,24,25,26,26.5,27,27.5,28,32]
    # # ========================= end ===================
    #
    # # ========================= 2020.05.09 1701 ===================
    # filename = 'F:/项目/花城音乐项目/样式数据/20.05.01MP3/dbg/3141/seg1.wav'
    # standard_kc = '喜爱春天的人儿是心地纯洁的人像紫罗兰花儿一样是我知心朋友'
    # # standard_kc = '喜爱夏天的人儿是意志坚强的人像冲打岩礁的波浪一样是我敬爱的父亲'
    # standard_kc_time = [0, 1, 2, 3, 3.5, 4, 5, 6, 8, 9, 10, 11, 11.5, 12, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,26.5, 27, 28, 32]
    # standard_notations = '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6- '
    # standard_notation_time = [0, 1, 1.5, 2, 3, 3.5, 4, 5, 6, 8, 9, 9.5, 10, 10.5, 11, 11.5, 12, 16, 17, 17.5, 18, 19,19.5, 20, 21, 21.5, 22, 23, 24, 25, 26, 26.5, 27, 27.5, 28, 32]
    #
    # # ========================= end ===================
    #
    # # ========================= 2020.05.09 6749 ===================
    # filename = 'F:/项目/花城音乐项目/样式数据/20.05.01MP3/wav/6749-1133.wav'
    # standard_kc = '喜爱春天的人儿是心地纯洁的人像紫罗兰花儿一样是我知心朋友'
    # # standard_kc = '喜爱夏天的人儿是意志坚强的人像冲打岩礁的波浪一样是我敬爱的父亲'
    # standard_kc_time = [0,0.6818181818181817,1.363636363636365,2.0454545454545467,2.3863636363636385,2.72727272727273,3.409090909090912,4.0909090909090935,5.454545454545459,6.136363636363642,6.818181818181822,7.500000000000002,7.840909090909092,8.181818181818182,10.909090909090908,11.590909090909092,12.272727272727272,12.954545454545455,13.636363636363635,14.318181818181818,14.999999999999998,15.681818181818182,16.363636363636367,17.045454545454547,17.727272727272734,18.06818181818182,18.409090909090914,19.090909090909093,21.81818181818182]
    # standard_notations = '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6- '
    # standard_notation_time = [0,0.6818181818181817,1.0227272727272734,1.363636363636365,2.0454545454545467,2.3863636363636385,2.72727272727273,3.409090909090912,4.0909090909090935,5.454545454545459,6.136363636363642,6.477272727272732,6.818181818181822,7.159090909090912,7.500000000000002,7.840909090909092,8.181818181818182,10.909090909090908,11.590909090909092,11.931818181818182,12.272727272727272,12.954545454545455,13.295454545454545,13.636363636363635,14.318181818181818,14.659090909090908,14.999999999999998,15.681818181818182,16.363636363636367,17.045454545454547,17.727272727272734,18.06818181818182,18.409090909090914,18.75,19.090909090909093,21.81818181818182]
    # # ========================= end ===================

    # ========================= 2020.05.12 ===================
    # filename = 'F:/项目/花城音乐项目/样式数据/20.05.12MP3/wav/200508-4710-1548.wav'
    # # filename = 'F:/项目/花城音乐项目/样式数据/20.05.12MP3/wav/200508-8312-1548.wav'
    # # filename = 'F:/项目/花城音乐项目/样式数据/20.05.12MP3/wav/200508-8312-4881.wav'
    # standard_kc = '喜爱春天的人儿是心地纯洁的人像紫罗兰花儿一样是我知心朋友'
    # # standard_kc = '喜爱夏天的人儿是意志坚强的人像冲打岩礁的波浪一样是我敬爱的父亲'
    # standard_kc_time = [0, 1, 2, 3, 3.5, 4, 5, 6, 8, 9, 10, 11, 11.5, 12, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,26.5, 27, 28, 32]
    # standard_notations = '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6- '
    # standard_notation_time = [0, 1, 1.5, 2, 3, 3.5, 4, 5, 6, 8, 9, 9.5, 10, 10.5, 11, 11.5, 12, 16, 17, 17.5, 18, 19, 19.5, 20, 21, 21.5, 22, 23, 24, 25, 26, 26.5, 27, 27.5, 28, 32]
    # ========================= end ===================

    # ========================= 2020.05.19 ===================
    # filename = 'F:/项目/花城音乐项目/样式数据/20.05.12MP3/wav/20200518-8354-1132.wav'
    # standard_kc = '喜爱春天的人儿是心地纯洁的人像紫罗兰花儿一样是我知心朋友'
    # # standard_kc = '喜爱夏天的人儿是意志坚强的人像冲打岩礁的波浪一样是我敬爱的父亲'
    # standard_kc_time = [0,0.6818181818181817,1.363636363636365,2.0454545454545467,2.3863636363636385,2.72727272727273,3.409090909090912,4.0909090909090935,5.454545454545459,6.136363636363642,6.818181818181822,7.500000000000002,7.840909090909092,8.181818181818182,10.909090909090908,11.590909090909092,12.272727272727272,12.954545454545455,13.636363636363635,14.318181818181818,14.999999999999998,15.681818181818182,16.363636363636367,17.045454545454547,17.727272727272734,18.06818181818182,18.409090909090914,19.090909090909093,21.81818181818182]
    # standard_notations = '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6- '
    # standard_notation_time = [0,0.6818181818181817,1.0227272727272734,1.363636363636365,2.0454545454545467,2.3863636363636385,2.72727272727273,3.409090909090912,4.0909090909090935,5.454545454545459,6.136363636363642,6.477272727272732,6.818181818181822,7.159090909090912,7.500000000000002,7.840909090909092,8.181818181818182,10.909090909090908,11.590909090909092,11.931818181818182,12.272727272727272,12.954545454545455,13.295454545454545,13.636363636363635,14.318181818181818,14.659090909090908,14.999999999999998,15.681818181818182,16.363636363636367,17.045454545454547,17.727272727272734,18.06818181818182,18.409090909090914,18.75,19.090909090909093,21.81818181818182]
    # ========================= end ===================

    # ========================= 2020.05.21 ===================
    # filename = 'F:/项目/花城音乐项目/样式数据/20.05.20MP3/wav/20200520-2360-1548.wav'
    # standard_kc = '喜爱春天的人儿是心地纯洁的人像紫罗兰花儿一样是我知心朋友'
    # # standard_kc = '喜爱夏天的人儿是意志坚强的人像冲打岩礁的波浪一样是我敬爱的父亲'
    # standard_kc_time = [0, 1, 2, 3, 3.5, 4, 5, 6, 8, 9, 10, 11, 11.5, 12, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 26.5, 27, 28, 32]
    # standard_notations = '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6- '
    # standard_notation_time = [0, 1, 1.5, 2, 3, 3.5, 4, 5, 6, 8, 9, 9.5, 10, 10.5, 11, 11.5, 12, 16, 17, 17.5, 18, 19, 19.5, 20, 21, 21.5, 22, 23, 24, 25, 26, 26.5, 27, 27.5, 28, 32]
    # ========================= end ===================

    # ========================= 2020.05.27 ===================
    # filename = 'F:/项目/花城音乐项目/样式数据/20.05.26MP3/20200526-8406-seg0.wav'
    # filename = 'F:/tmp/20200526-8406-1133.wav'
    # standard_kc = '喜爱春天的人儿是心地纯洁的人像紫罗兰花儿一样是我知心朋友'
    # # standard_kc = '喜爱夏天的人儿是意志坚强的人像冲打岩礁的波浪一样是我敬爱的父亲'
    # standard_kc_time = [0,0.6818181818181817,1.363636363636365,2.0454545454545467,2.3863636363636385,2.72727272727273,3.409090909090912,4.0909090909090935,5.454545454545459,6.136363636363642,6.818181818181822,7.500000000000002,7.840909090909092,8.181818181818182,10.909090909090908,11.590909090909092,12.272727272727272,12.954545454545455,13.636363636363635,14.318181818181818,14.999999999999998,15.681818181818182,16.363636363636367,17.045454545454547,17.727272727272734,18.06818181818182,18.409090909090914,19.090909090909093,21.81818181818182]
    # standard_notations = '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6- '
    # standard_notation_time = [0,0.6818181818181817,1.0227272727272734,1.363636363636365,2.0454545454545467,2.3863636363636385,2.72727272727273,3.409090909090912,4.0909090909090935,5.454545454545459,6.136363636363642,6.477272727272732,6.818181818181822,7.159090909090912,7.500000000000002,7.840909090909092,8.181818181818182,10.909090909090908,11.590909090909092,11.931818181818182,12.272727272727272,12.954545454545455,13.295454545454545,13.636363636363635,14.318181818181818,14.659090909090908,14.999999999999998,15.681818181818182,16.363636363636367,17.045454545454547,17.727272727272734,18.06818181818182,18.409090909090914,18.75,19.090909090909093,21.81818181818182]
    # ========================= end ===================

    # ========================= 2020.05.27 ===================
    # filename = 'F:/tmp/20200530-2202-1133.wav'
    # standard_kc = '喜爱春天的人儿是心地纯洁的人像紫罗兰花儿一样是我知心朋友'
    # # standard_kc = '喜爱夏天的人儿是意志坚强的人像冲打岩礁的波浪一样是我敬爱的父亲'
    # standard_kc_time = [0,0.6818181818181817,1.363636363636365,2.0454545454545467,2.3863636363636385,2.72727272727273,3.409090909090912,4.0909090909090935,5.454545454545459,6.136363636363642,6.818181818181822,7.500000000000002,7.840909090909092,8.181818181818182,10.909090909090908,11.590909090909092,12.272727272727272,12.954545454545455,13.636363636363635,14.318181818181818,14.999999999999998,15.681818181818182,16.363636363636367,17.045454545454547,17.727272727272734,18.06818181818182,18.409090909090914,19.090909090909093,21.81818181818182]
    # standard_notations = '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6- '
    # standard_notation_time = [0,0.6818181818181817,1.0227272727272734,1.363636363636365,2.0454545454545467,2.3863636363636385,2.72727272727273,3.409090909090912,4.0909090909090935,5.454545454545459,6.136363636363642,6.477272727272732,6.818181818181822,7.159090909090912,7.500000000000002,7.840909090909092,8.181818181818182,10.909090909090908,11.590909090909092,11.931818181818182,12.272727272727272,12.954545454545455,13.295454545454545,13.636363636363635,14.318181818181818,14.659090909090908,14.999999999999998,15.681818181818182,16.363636363636367,17.045454545454547,17.727272727272734,18.06818181818182,18.409090909090914,18.75,19.090909090909093,21.81818181818182]
    # ========================= end ===================

    # ========================= 2020.05.31 ===================
    # filename = 'F:/tmp/20200530-2202-1133.wav'
    # filename = 'F:/项目/花城音乐项目/人声分离数据库/新建文件夹/录音31.wav'
    # filename = 'F:/项目/花城音乐项目/人声分离数据库/新建文件夹/录音30.wav'
    # standard_kc = '喜爱春天的人儿是心地纯洁的人像紫罗兰花儿一样是我知心朋友'
    # # standard_kc = '喜爱夏天的人儿是意志坚强的人像冲打岩礁的波浪一样是我敬爱的父亲'
    # standard_kc_time = [0,0.6818181818181817,1.363636363636365,2.0454545454545467,2.3863636363636385,2.72727272727273,3.409090909090912,4.0909090909090935,5.454545454545459,6.136363636363642,6.818181818181822,7.500000000000002,7.840909090909092,8.181818181818182,10.909090909090908,11.590909090909092,12.272727272727272,12.954545454545455,13.636363636363635,14.318181818181818,14.999999999999998,15.681818181818182,16.363636363636367,17.045454545454547,17.727272727272734,18.06818181818182,18.409090909090914,19.090909090909093,21.81818181818182]
    # standard_notations = '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6- '
    # standard_notation_time = [0,0.6818181818181817,1.0227272727272734,1.363636363636365,2.0454545454545467,2.3863636363636385,2.72727272727273,3.409090909090912,4.0909090909090935,5.454545454545459,6.136363636363642,6.477272727272732,6.818181818181822,7.159090909090912,7.500000000000002,7.840909090909092,8.181818181818182,10.909090909090908,11.590909090909092,11.931818181818182,12.272727272727272,12.954545454545455,13.295454545454545,13.636363636363635,14.318181818181818,14.659090909090908,14.999999999999998,15.681818181818182,16.363636363636367,17.045454545454547,17.727272727272734,18.06818181818182,18.409090909090914,18.75,19.090909090909093,21.81818181818182]
    # ========================= end ===================

    # ========================= 2020.05.31 ===================
    filename = 'F:/项目/花城音乐项目/样式数据/20.05.30MP3/test-c5.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/20.05.30MP3/test-29.wav'
    standard_kc = '喜爱春天的人儿是心地纯洁的人像紫罗兰花儿一样是我知心朋友'
    # standard_kc = '喜爱夏天的人儿是意志坚强的人像冲打岩礁的波浪一样是我敬爱的父亲'
    standard_kc_time = [0, 1, 2, 3, 3.5, 4, 5, 6, 8, 9, 10, 11, 11.5, 12, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                        26.5, 27, 28, 32]
    # standard_kc_time = [0, 1, 2, 3, 3.5, 4, 5, 6, 8, 9, 10, 11, 11.5, 12, 16, 17, 17.5,18, 19,19.5, 20, 21, 22, 23, 24, 25, 26,26.5,26.75, 27, 28, 32]
    standard_notations = '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
    standard_notation_time = [0, 1, 1.5, 2, 3, 3.5, 4, 5, 6, 8, 9, 9.5, 10, 10.5, 11, 11.5, 12, 16, 17, 17.5, 18, 19,
                              19.5, 20, 21, 21.5, 22, 23, 24, 25, 26, 26.5, 27, 27.5, 28, 32]
    # ========================= end ===================
    # total_score, pitch_total_score, notation_duration_total_score, kc_duration_total_score,kc_express_total_score,fluency_total_score, pitch_score_detail, notation_duration_score_detail, kc_rhythm_sscore_detail,kc_express_sscore_detail,fluency_sscore_detail = score_all(filename, standard_kc,standard_kc_time, standard_notations, standard_notation_time)
    # print("total_score is {}".format(total_score))
    # score_detail = "音高评分结果为{}，{}，音符节奏评分结果为{}，{}，歌词节奏评分结果为{}，{}，歌词表达评分结果为{}，{}，流畅度评分结果为{}，{}".format(pitch_total_score,
    #                                                                        pitch_score_detail,
    #                                                                        notation_duration_total_score,
    #                                                                        notation_duration_score_detail,
    #                                                                        kc_duration_total_score,
    #                                                                        kc_rhythm_sscore_detail,
    #                                                                        kc_express_total_score,
    #                                                                        kc_express_sscore_detail,
    #                                                                        fluency_total_score,
    #                                                                        fluency_sscore_detail)
    # print("score detail is {}".format(score_detail))
    print("+++++++++++++++++++++++++++++++++++~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~++++++++++++++++++++++++++++")
    # total_score, pitch_total_score, notation_duration_total_score, kc_duration_total_score, pitch_score_detail, notation_duration_score_detail, kc_rhythm_sscore_detail = score_all_by_st(filename, standard_kc, standard_kc_time, standard_notations, standard_notation_time)
    # print("total_score is {}".format(total_score))
    # score_detail = "音高评分结果为{}，{}，音符节奏评分结果为{}，{}，歌词节奏评分结果为{}，{}".format(pitch_total_score,
    #                                                                    pitch_score_detail,
    #                                                                    notation_duration_total_score,
    #                                                                    notation_duration_score_detail,
    #                                                                    kc_duration_total_score,
    #                                                                    kc_rhythm_sscore_detail)
    # print("score detail is {}".format(score_detail))

    # standard_notations = '5,5,5,5,3,5,1,6,5,5,5,5,5,3,2,1,3,2,2,3,5,5,3,3,2,1,1,2,3,1,1,6-,5-,6-,5-'
    # standard_notation_times = [0, 0.17045454545454586, 0.5113636363636367, 0.6818181818181825, 1.0227272727272734, 1.1931818181818192, 1.363636363636365, 1.704545454545456, 2.0454545454545467, 2.7272727272727284, 2.8977272727272734, 3.238636363636365, 3.40909090909091, 3.7500000000000018, 3.9204545454545467, 4.090909090909092, 4.431818181818183, 4.772727272727275, 5.454545454545457, 5.7954545454545485, 5.9659090909090935, 6.306818181818185, 6.47727272727273, 6.647727272727275, 6.988636363636367, 7.159090909090912, 7.329545454545457, 7.6704545454545485, 8.01136363636364, 8.352272727272732, 8.522727272727277, 8.863636363636369, 9.034090909090914, 9.204545454545459, 9.54545454545455, 9.886363636363642, 10.568181818181824]
    # all_first_candidate_names = ['B3', 'B3', 'B3', 'G#3', 'G#3', 'D#3', 'D3', 'E3', 'C4', 'B3', 'B3', 'B3', 'B3', 'G3', 'F#3', 'E3', 'D#3', 'D#3', 'G3', 'F#3', 'F3', 'F#3', 'G3', 'A#3', 'A#3', 'B3', 'B3', 'G3', 'G3', 'G3', 'F#3', 'D#3', 'E3', 'D#3', 'F3', 'F3', 'E3', 'D#3', 'C3', 'B2', 'A#2', 'C3', None]
    # test_times = [0.26074144486692014, 0.7407414448669202, 0.9407414448669201, 1.23, 1.3807414448669202, 1.5807414448669204, 1.78074144486692, 1.89, 2.05, 2.24, 2.4607414448669203, 2.9407414448669202, 3.44, 3.9407414448669202, 4.14, 4.30074144486692, 4.42074144486692, 4.62074144486692, 4.88, 5.02, 5.68, 5.78074144486692, 6.21, 6.39, 6.54074144486692, 6.71, 6.9, 7.0607414448669195, 7.22074144486692, 7.41, 7.55, 7.73, 7.90074144486692, 8.14074144486692, 8.58074144486692, 8.92, 9.06, 9.37, 9.54, 9.74074144486692, 9.90074144486692, 10.08, 10.54074144486692]
    # standard_notations = standard_notations.split(',')
    # standard_notations = [s[0] for s in standard_notations]
    # standard_notations = ''.join(standard_notations)
    # lcseque = check_moved_step(standard_notations, standard_notation_times, all_first_candidate_names, test_times)
    # print("lcseque size is {} ,detail {}".format(len(lcseque),lcseque))
    # best_numbered_notations,type,best_lcseque_len = find_the_cut_point(standard_notations, standard_notation_times, all_first_candidate_names, test_times, end_time=None)
    # print("best_numbered_notations size is {} ,detail {}".format(len(best_numbered_notations), best_numbered_notations))
    # print("tpye is {}".format(type))
    # print("best_lcseque_len is {}".format(best_lcseque_len))

    standard_notations = '5,5,5,5,3,5,1,6,5,5,5,5,5,3,2,1,3,2,2,3,5,5,3,3,2,1,1,2,3,1,1,6-,5-,6-,5-'
    standard_notation_times = [0, 0.17045454545454675, 0.5113636363636367, 0.6818181818181834, 1.0227272727272734, 1.1931818181818201, 1.3636363636363669, 1.7045454545454568, 2.0454545454545467, 2.72727272727273, 2.897727272727277, 3.238636363636367, 3.4090909090909136, 3.7500000000000036, 3.9204545454545503, 4.090909090909097, 4.431818181818187, 4.772727272727277, 5.45454545454546, 5.79545454545455, 5.965909090909097, 6.306818181818187, 6.647727272727277, 6.988636363636367, 7.159090909090914, 7.32954545454546, 7.67045454545455, 8.01136363636364, 8.35227272727273, 8.522727272727277, 8.863636363636367, 9.034090909090914, 9.20454545454546, 9.54545454545455, 9.88636363636364, 10.568181818181824]
    all_first_candidate_names = ['A#2', None, 'B3', 'B3', 'B3', 'B3', 'B3', 'G#3', 'G#3', 'D#3', 'E3', 'C#4', 'B3', 'B3', 'B3', 'B3', 'B3', 'A#3', 'A#3', 'A3', 'F#3', 'F#3', 'D#3', 'D#3', 'D#3', 'G#3', 'F#3', 'F3', 'F3', 'F#3', 'F#3', 'F#3', 'F#3', 'A#3', 'B3', 'B3', 'B3', 'G3', 'G3', 'G3', 'G3', 'F3', 'E3', 'D#3', 'E3', 'D#3', 'D#3', 'F3', 'F3', 'D#3', 'D#3', 'D#3', 'A#2']
    test_times = [0.020056764427625356, 0.26, 0.66, 0.8600567644276254, 1.05, 1.3400567644276253, 1.5000567644276253, 1.7000567644276252, 1.8, 1.91, 2.28, 2.3800567644276254, 2.62, 2.9000567644276254, 3.42, 3.5800567644276255, 3.78, 3.97, 4.100056764427626, 4.33, 4.420056764427626, 4.53, 4.63, 4.740056764427625, 4.97, 5.0600567644276255, 5.35, 5.740056764427625, 5.99, 6.15, 6.25, 6.420056764427626, 6.52, 6.72, 6.93, 7.05, 7.140056764427626, 7.340056764427626, 7.580056764427625, 7.73, 7.820056764427625, 7.940056764427626, 8.11, 8.21, 8.300056764427625, 8.48, 8.78, 8.9, 9.180056764427624, 9.4, 9.7, 9.860056764427625, 10.11]
    standard_notations = standard_notations.split(',')
    standard_notations = [s[0] for s in standard_notations]
    standard_notations = ''.join(standard_notations)
    lcseque = check_moved_step(standard_notations, standard_notation_times, all_first_candidate_names, test_times)
    print("lcseque size is {} ,detail {}".format(len(lcseque),lcseque))
    # best_numbered_notations,type,best_lcseque_len = find_the_cut_point(standard_notations, standard_notation_times, all_first_candidate_names, test_times, end_time=None)
    best_numbered_notations,type,best_lcseque_len = find_best_numbered_notations(standard_notations, standard_notation_times, all_first_candidate_names, test_times, end_time=None)
    print("best_numbered_notations size is {} ,detail {}".format(len(best_numbered_notations), best_numbered_notations))
    print("tpye is {}".format(type))
    print("best_lcseque_len is {}".format(best_lcseque_len))