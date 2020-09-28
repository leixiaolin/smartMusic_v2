import parselmouth
import matplotlib.pyplot as plt
import numpy as np
import librosa

FREQS = [
    ('B0', 30.87), ('C1', 32.7), ('C#1', 34.65),
    ('D1', 36.71), ('D#1', 38.89), ('E1', 41.2),
    ('F1', 43.65), ('F#1', 46.35), ('G1', 49),
    ('G#1', 51.91), ('A1', 55), ('A#1', 58.27),
    ('B1', 61.74), ('C2', 65.41), ('C#2', 69.3),
    ('D2', 73.42), ('D#2', 77.78), ('E2', 82.41),
    ('F2', 87.31), ('F#2', 92.50), ('G2', 98.00),
    ('G#2', 103.83), ('A2', 110.00), ('A#2', 116.54),
    ('B2', 123.54), ('C3', 130.81), ('C#3', 138.59),
    ('D3', 146.83), ('D#3', 155.56), ('E3', 164.81),
    ('F3', 174.61), ('F#3', 185.00), ('G3', 196.00),
    ('G#3', 207.65), ('A3', 220.00), ('A#3', 233.08),
    ('B3', 246.94), ('C4', 261.63), ('C#4', 277.18),
    ('D4', 293.66), ('D#4', 311.13), ('E4', 329.63),
    ('F4', 349.23), ('F#4', 369.99), ('G4', 392.00),
    ('G#4', 415.30), ('A4', 440.00), ('A#4', 466.16),
    ('B4', 493.88), ('C5', 523.25), ('C#5', 554.37),
    ('D5', 587.33), ('D#5', 622.25), ('E5', 659.26),
    ('F5', 698.46), ('F#5', 739.99), ('G5', 783.99),
    ('G#5', 830.61), ('A5', 880, 00), ('A#5', 932.33),
    ('B5', 987.77), ('C6', 1046.50), ('C#6', 1108.73),
    ('D6', 1174.66), ('D#6', 1244.51), ('E6', 1318.51),
    ('F6', 1396.91), ('F#6', 1479.98), ('G6', 1567.98),
    ('G#6', 1661.22), ('A6', 1760.00), ('A#6', 1864.66),
    ('B6', 1975.53), ('C7', 2093), ('C#7', 2217.46),
    ('D7', 2349.32), ('D#7', 2489.02), ('E7', 2637.03),
    ('F7', 2793.83), ('F#7', 2959.96), ('G7', 3135.44),
    ('G#7', 3322.44), ('A7', 3520), ('A#7', 3729.31),
    ('B7', 3951.07)
]

PITCH_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B', 'C', 'C#', 'D3', 'D#', 'E', 'F', 'F#',
               'G', 'G#', 'A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

'''
获取单个音高符号的序号
'''
def get_index_by_notation_name(notation_name):
    indexs = [i for i,tup in enumerate(FREQS) if tup[0] == notation_name]
    if len(indexs) > 0:
        return indexs[0]
    else:
        return -1

def get_notation_name_by_index(index):
    notation_name = [tup[0] for i,tup in enumerate(FREQS) if i == index]
    if len(notation_name) > 0:
        return notation_name[0]
    else:
        return None

'''
获取每个音高符号的序号，主要用于音高平移
notations_names = ['C4', 'C3', 'C3', 'C4', 'C4', 'C4', 'C4', 'C4', None, 'C4', 'D4', 'G#3', 'G#3', 'A#3', 'C4', 'C4']
'''
def get_all_indexs_for_notation_names(notation_names):
    result = []
    for notation_name in notation_names:
        index = get_index_by_notation_name(notation_name)
        result.append(index)
    return result

def get_all_notation_names_for_indexs(indexs):
    result = []
    for i in indexs:
        notation_name = get_notation_name_by_index(i)
        result.append(notation_name)
    return result

'''
平移音高，找到最匹配的音高
standard_notations = '5,5,5,5,3,5,1,6,5,5,5,5,5,3,2,1,3,2,2,3,5,5,3,3,2,1,1,2,3,1,1,6-,5-,6-,5-'
test_notations = ['C4', 'C3', 'C3', 'C4', 'C4', 'C4', 'C4', 'C4', None, 'C4', 'D4', 'G#3', 'G#3', 'A#3', 'C4', 'C4']
'''
def get_best_candidate_names_by_moved(standard_notations, test_notations):
    standard_notations = standard_notations.replace("[", '')
    standard_notations = standard_notations.replace("]", '')
    standard_notations = standard_notations.split(',')
    standard_notations = [s[0] for s in standard_notations]
    standard_notations = ''.join(standard_notations)

    test_notations = [n for n in test_notations if n is not None]
    numbered_test_notations = [get_numbered_musical_notation(n)[0] for n in test_notations]
    numbered_test_notations = ''.join(numbered_test_notations)
    # print("numbered_test_notations_moved is {},size is {}".format(numbered_test_notations,len(numbered_test_notations)))
    lcseque,d = find_lcseque(standard_notations, numbered_test_notations)
    # print("lcseque is {},size is {}".format(lcseque, len(lcseque)))
    best_lcseque_len = len(lcseque)
    best_numbered_notations = test_notations
    best_d = d
    for step in range(1,8):
        all_indexs = get_all_indexs_for_notation_names(test_notations)
        all_indexs_moved = [a + step for a in all_indexs]
        test_notations_moved = get_all_notation_names_for_indexs(all_indexs_moved)
        numbered_test_notations_moved = [get_numbered_musical_notation(n)[0] for n in test_notations_moved]
        numbered_test_notations_moved = ''.join(numbered_test_notations_moved)
        lcseque,d = find_lcseque(standard_notations, numbered_test_notations_moved)
        # print("numbered_test_notations_moved is {},size is {}".format(numbered_test_notations_moved, len(numbered_test_notations_moved)))
        # print("lcseque is {},size is {}".format(lcseque, len(lcseque)))
        if len(lcseque) > best_lcseque_len:
            best_lcseque_len = len(lcseque)
            best_numbered_notations = test_notations_moved
            best_d = d
    return best_numbered_notations,best_d

'''
https://blog.csdn.net/ggdhs/article/details/90713154
https://blog.csdn.net/miner_zhu/article/details/81159902
'''
def find_lcseque(s1, s2):
    # 生成字符串长度加1的0矩阵，m用来保存对应位置匹配的结果
    m = [[0 for x in range(len(s2) + 1)] for y in range(len(s1) + 1)]
    # d用来记录转移方向
    d = [[None for x in range(len(s2) + 1)] for y in range(len(s1) + 1)]

    for p1 in range(len(s1)):
        for p2 in range(len(s2)):
            if s1[p1] == s2[p2]:  # 字符匹配成功，则该位置的值为左上方的值加1
                m[p1 + 1][p2 + 1] = m[p1][p2] + 1
                d[p1 + 1][p2 + 1] = 'ok'
            elif m[p1 + 1][p2] > m[p1][p2 + 1]:  # 左值大于上值，则该位置的值为左值，并标记回溯时的方向
                m[p1 + 1][p2 + 1] = m[p1 + 1][p2]
                d[p1 + 1][p2 + 1] = 'left'
            else:  # 上值大于左值，则该位置的值为上值，并标记方向up
                m[p1 + 1][p2 + 1] = m[p1][p2 + 1]
                d[p1 + 1][p2 + 1] = 'up'
    (p1, p2) = (len(s1), len(s2))
    # print(numpy.array(d))
    s = []
    positions = []
    while m[p1][p2]:  # 不为None时
        c = d[p1][p2]
        if c == 'ok':  # 匹配成功，插入该字符，并向左上角找下一个
            s.append(s1[p1 - 1])
            positions.append((p1,p2))
            p1 -= 1
            p2 -= 1
        if c == 'left':  # 根据标记，向左找下一个
            p2 -= 1
        if c == 'up':  # 根据标记，向上找下一个
            p1 -= 1
    s.reverse()
    positions.reverse()
    return ''.join(s),positions

def get_matched_positions(d):
    standard_positions = []
    test_positions = []
    for p in d:
        standard_positions.append(p[0])
        test_positions.append(p[1])
    return standard_positions,test_positions

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


def get_pitch_by_parselmouth(filename):
    snd = parselmouth.Sound(filename)
    pitch = snd.to_pitch()
    return pitch


def get_pitch_diff(pitch):
    pitch_values = pitch.selected_array['frequency']
    pitch_values[pitch_values == np.nan] = 0
    pitch_diff = np.diff(pitch_values)
    return pitch_diff


def get_pitch_derivative(pitch, n=2):
    pitch_values = pitch.selected_array['frequency']
    pitch_values[pitch_values == np.nan] = 0
    pitch_diff = [pitch_values[i + n] - pitch_values[i - n] for i in range(n, len(pitch_values) - n)]
    padding = np.zeros(n)
    pitch_diff = np.hstack((padding, pitch_diff))
    pitch_diff = np.hstack((pitch_diff, padding))
    pitch_diff = [5 if p >= 5 else p for p in pitch_diff]
    return pitch_diff


def get_pitch_derivative_from_file(filename):
    pitch = get_pitch_by_parselmouth(filename)
    pitch_derivative = get_pitch_derivative(pitch, n=1)
    # pitch_derivative = signal.medfilt(pitch_derivative, 5)  # 中值滤波
    pitch_derivative = [p if np.abs(p) >= 3 else 0 for p in pitch_derivative]
    return pitch_derivative


def draw_pitch(pitch):
    # Extract selected pitch contour, and
    # replace unvoiced samples by NaN to not plot
    pitch_values = pitch.selected_array['frequency']
    pitch_values[pitch_values == 0] = np.nan
    # plt.plot(pitch.xs(), pitch_values, 'o', markersize=5, color='w')
    plt.plot(pitch.xs(), pitch_values, 'o', markersize=1)
    plt.grid(False)
    plt.ylim(0, pitch.ceiling)
    plt.ylabel("fundamental frequency [Hz]")


def draw_spectrogram(spectrogram, dynamic_range=70):
    X, Y = spectrogram.x_grid(), spectrogram.y_grid()
    sg_db = 10 * np.log10(spectrogram.values)
    plt.pcolormesh(X, Y, sg_db, vmin=sg_db.max() - dynamic_range, cmap='afmhot')
    plt.ylim([spectrogram.ymin, spectrogram.ymax])
    plt.xlabel("time [s]")
    plt.ylabel("frequency [Hz]")


def draw_intensity(intensity):
    # plt.plot(intensity.xs(), intensity.values.T, linewidth=3, color='w')
    plt.plot(intensity.xs(), intensity.values.T, linewidth=1)
    plt.grid(False)
    plt.ylim(0)
    plt.ylabel("intensity [dB]")


def get_pitch_values(pitch_values, check_type='big'):
    if check_type == 'big':
        freqs = [tup for tup in FREQS if tup[0].find('#') < 0]
    elif check_type == 'small':
        freqs = [tup for tup in FREQS]
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


def get_pitch_name(freq):
    freqs = [tup for tup in FREQS]
    for tup in freqs:
        if tup[1] == freq:
            return tup[0]
    return np.nan


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
    return pitch_values_candidate


def get_all_numbered_notation_and_offset(pitch, onset_frames, sr=44100):
    from collections import Counter
    import scipy.signal as signal
    # 将起始帧和结束帧换算成时间点
    # onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    # # librosa时间点换算成parselmouth的帧所在位置
    # all_starts_parselmouth = [int(o * pitch.n_frames / pitch.duration) for o in onset_times]
    # onset_frames = all_starts_parselmouth
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
            if Counter(candidate_notation_tmp).most_common(2)[0][0] < 50 and Counter(candidate_notation_tmp).most_common(2)[1][1] < 20 and end_frame != onset_frames[-1]: # 如果该区间的音高为空
                all_first_candidate_names.append(None)
                all_first_candidates.append(None)
                all_offset_types.append(None)
                continue
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
                return np.nan, np.nan, np.nan
            most_notation_freq = res.most_common(1)[0][0]  # 第一判定音高
            first_candidate_name = most_notation_freq if np.isnan(most_notation_freq) else get_pitch_name(most_notation_freq)
            small_list = [i for i, p in enumerate(pitch_tmp) if p < most_notation_freq and p > 50]
            more_list = [i for i, p in enumerate(pitch_tmp) if p > most_notation_freq and p > 50]
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

'''
获取每个起始点后面的音高长度
'''
def get_pitch_length_for_each_onset(pitch,onset_frames,grain_size=0):
    from collections import Counter
    pitch_values = pitch.selected_array['frequency']
    pitch_values[pitch_values == 0] = np.nan
    if grain_size == 1:
        freqs = FREQS
    else:
        freqs = [tup for tup in FREQS if tup[0].find('#') < 0]
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

    # 将小缝隙补齐
    pitch_values_candidate = smooth_pitch_values_candidate(pitch_values_candidate)
    # 获取连续段的起始点及长度
    starts, lens = get_starts_and_length_for_section(pitch_values_candidate)

    result = []
    for i,o in enumerate(onset_frames):
        if o != onset_frames[-1]:
            tmp = pitch_values_candidate[o:onset_frames[i+1]]
        else:
            tmp = pitch_values_candidate[o:]
        # 找出tmp中出现最多的元素
        res = Counter(tmp)
        sorted(res)
        most = res.most_common(1)[0][0]  # 就是相应的最高频元素
        length = res.most_common(1)[0][1]  # 就是相应的最高频元素的频次
        if most is np.NAN:  # 如果最长的音高是空白项
            if len(res.most_common()) > 1 and res.most_common(2)[1][0] > 70 and res.most_common(2)[1][1] > 10:  # 如果第二长的音高是有效音高
                result.append(res.most_common(2)[1][1])
            else:
                result.append(0)
        else:
            result.append(length)
    return result,starts, lens

'''
删除过短音高或空白的起始点
'''
def del_onset_frames_for_too_short(pitch,onset_frames,grain_size=0):
    lengths,starts, lens = get_pitch_length_for_each_onset(pitch, onset_frames)
    dels = []
    for i,o in enumerate(lengths):
        if o < 5:

            if o == lengths[-1]: # 最后一个
                dels.append(i)
            else:
                onset_fs = onset_frames[i]
                onset_fe = onset_frames[i+1]
                offset_fs = [np.abs(n-onset_fs) for n in starts]
                offset_fe = [np.abs(n - onset_fe) for n in starts]
                if np.min(offset_fs) < np.min(offset_fe):
                    dels.append(i+1)
                else:
                    dels.append(i)
    result = [o for i,o in enumerate(onset_frames) if i not in dels]
    return result

def draw_pitch_specified(intensity, pitch, pitch_values, draw_type=1, filename='', notation='', grain_size=0):
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

    # 防止个别偏离现象
    # if pitch_values_max - pitch_values_mean > 100:
    #     p_min = int(pitch_values_mean * 0.5)
    #     p_max = int(pitch_values_mean * 1.5)
    pitch_values[pitch_values == 0] = np.nan
    if draw_type == 1:
        plt.plot(pitch.xs(), pitch_values, 'o', markersize=5, color='w')
        plt.plot(pitch.xs(), pitch_values, 'o', markersize=2)
    else:
        if grain_size == 1:
            freqs = FREQS
        else:
            freqs = [tup for tup in FREQS if tup[0].find('#') < 0]
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

        # 获取连续段的起始点及长度
        # starts, lens = get_starts_and_length_for_section(pitch_values_candidate)
        # pitch_values_candidate_moved = smooth_pitch_values_candidate(pitch_values_candidate_moved)
        plt.plot(pitch.xs(), pitch_values_candidate, 'o', markersize=2)
        # plt.plot(pitch.xs(), pitch_values_candidate_moved, '*', markersize=4, color="r")
    plt.grid(False)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.title(filename, fontsize=16)
    # plt.ylim(0, pitch.ceiling)
    pitch_all = [p for p in freqs_points if p > p_min and p < p_max]
    plt.hlines(pitch_all, 0, len(pitch_values)*10e-3, color='0.2', linewidth=1, linestyle=":")
    # p_min, p_max = 70,500
    plt.ylim(p_min, p_max)
    plt.ylabel("fundamental frequency [Hz]")
    plt.xlabel(notation)
    # 设置坐标轴刻度
    x_ticks = np.arange(0, pitch.duration, 1)
    plt.xticks(x_ticks)
    pitch_name = [tup[0] for tup in freqs if tup[1] > p_min and tup[1] < p_max]
    for i, p in enumerate(pitch_all):
        numbered_musical_notation = get_numbered_musical_notation(pitch_name[i])
        plt.text(0.1, p, pitch_name[i] + " - " + numbered_musical_notation, size='8')

    # plt.xlim([snd.xmin, snd.xmax])
    plt.twinx()
    draw_intensity(intensity)
    return plt
