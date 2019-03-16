import  numpy as np
import librosa
import matplotlib.pyplot as plt
import re
import math
from viterbi import *
from note_frequency import *

codes = np.array(['[1000,1000;2000;1000,500,500;2000]',
                  '[2000;1000,1000;500,500,1000;2000]',
                  '[1000,1000;500,500,1000;1000,1000;2000]',
                  '[1000,--(1000);1000,--(1000);500,250,250,1000;--(1000),1000]',
                  '[500;1000,500,1000,500;500,500,500,250,250,500,500;250,250,500,500,1000]',
                  '[1000,--(1000);1000,--(1000);1000,-(500),500;1000,1000]',
                  '[750,250,500,500,500,-(500);500,1000,500,500,-(500);750,250,500,500,500,-(500)]',
                  '[500,1000,500,500,250,250;1000,500,750,250,500;3000]',
                  '[500,500,500;1000,500;500,500,500;1500;500,500,500;1000,500;500;1000;1500]',
                  '[500,500,1000;500,500;1000;375,125,250,250,375,125,250,250;500,500,1000]'])
note_codes = np.array(['[3,3,3,3,3,3,3,5,1,2,3]',
                       '[5,5,3,2,1,2,5,3,2]',
                       '[5,5,3,2,1,2,2,3,2,6,5]',
                       '[5,1,7,1,2,1,7,6,5,2,4,3,6,5]',
                       '[3,6,7,1,2,1,7,6,3]',
                       '[1,7,1,2,3,2,1,7,6,7,1,2,7,1,7,1,12,1]',
                       '[5,6,1,6,2,3,1,6,5]',
                       '[5,5,6,5,6,5,1,3,0,2,2,5,2,1]',
                       '[3,2,1,2,1,1,2,3,4,5,3,6,5,5,3]',
                       '[3,4,5,1,7,6,5]'])
rhythm_codes = np.array(['[500,500,1000;500,500,1000;500,500,750,250;2000]',
                        '[1000,1000;500,500,1000;1000,500,500; 2000]',
                        '[1000,1000;500,500,1000;500,250,250,250;2000]',
                        '[500,1000,500;250,250,250,250,500,500;500,500,500,500;2000]',
                        '[1000;500,500,1000;500,500,500,500;2000]',
                        '[500;500,500,500,500;500,500,500,500;500,500,500,500;250,250,250,250,500]',
                        '[1000,750,250,2000;500,500,500,500,2000]',
                        '[1000,1000,1000,500,500;1000,1000,1000,--(1000);250,750,1000,1000;1000,4000]',
                        '[1500,500,500,500;2500,500;1000,500,500,500,500;2500,500]',
                        '[500,500;1500,500,500,500;2000]'])
pitch_base = ['C','D','E','F','G','A','B']
pitch_number = ['1','2','3','4','5','6','7']
pitch_v = [0,2,4,5,7,9,11]

def load_and_trim(path):
    audio, sr = librosa.load(path)
    energy = librosa.feature.rmse(audio)
    frames = np.nonzero(energy >= np.max(energy) / 5)
    indices = librosa.core.frames_to_samples(frames)[1]
    audio = audio[indices[0]:indices[-1]] if indices.size else audio[0:0]

    return audio, sr


def get_basetime(s):
    if s is None or len(s) < 1:
        print("input is empty")

    s = s.replace('[','').replace(']','')
    tmp = s.split(';')
    print(tmp)
    result = []
    for c in tmp:
        if c.find(","):  # 包括","的情况，即有多个数值
            cc = c.split(",")
            for ccc in cc:
                if ccc.find("(") > 0: # 空音的情况
                    score = re.sub("\D", "", ccc)  # 筛选数字
                    score = "-" + score
                    result.append(score)
                else:
                    result.append(ccc)
        else: # 不包括","的情况，即只有一个数值
            if c.find("(") > 0:  # 空音的情况
                score = re.sub("\D", "", c)  # 筛选数字
                score = -1 * score
                result.append(score)
            else:
                result.append(c)
    return result

def get_chroma_pitch(pitch_code):
    result = []
    s = pitch_code.replace('[','').replace(']','')
    tmp = s.split(',')
    for x in tmp:
        result.append(pitch_v[int(x)-1])
    return result

def onsets_base(code,time,start_point):
    result = get_basetime(code)
    print(result)
    total = 0
    for r in result:
        if int(r) > 0:  # 不是空音
            total += int(r)
        else:
            total -= int(r)

    off = 0  # 累积时长，用于计算后面每个节拍点的位置
    ds = []
    for i, r in enumerate(result):
        if int(r) > 0:  # 不是空音
            ds.append(start_point + time * off / total)
            off += int(r)
        else:
            off -= int(r)

    ds.append(time)
    return ds
def onsets_base_frames(code,frames_number):
    result = get_basetime(code)
    print(result)
    total = 0
    for r in result:
        if int(r) > 0:  # 不是空音
            total += int(r)
        else:
            total -= int(r)

    off = 0  # 累积时长，用于计算后面每个节拍点的位置
    ds = []
    for i, r in enumerate(result):
        if int(r) > 0:  # 不是空音
            ds.append(math.ceil(frames_number * off / total))
            off += int(r)
        else:
            off -= int(r)

    #ds.append(frames_number)
    return ds

def onsets_base_frames_rhythm(index,frames_number):
    result = get_basetime(rhythm_codes[index])
    print(result)
    total = 0
    for r in result:
        if int(r) > 0:  # 不是空音
            total += int(r)
        else:
            total -= int(r)

    off = 0  # 累积时长，用于计算后面每个节拍点的位置
    ds = []
    for i, r in enumerate(result):
        if int(r) > 0:  # 不是空音
            ds.append(math.ceil(frames_number * off / total))
            off += int(r)
        else:
            off -= int(r)

    #ds.append(frames_number)
    return ds

'''
找波峰
'''
def get_next_peak(y):
    index = -1
    y_diff = np.diff(y)
    for i in range(len(y_diff)-1):
        if y_diff[i] >= 0 and y_diff[i+1] < 0:
            index = i + 1
            break
    return index

'''
找波谷
'''
def get_next_trough(y):
    index = -1
    y_diff = np.diff(y)
    for i in range(len(y_diff)-1):
        if y_diff[i] <= 0 and y_diff[i+1] > 0:
            index = i + 1
            break
    return index

'''
找所有波峰
'''
def get_all_peak(y):
    points = []
    start = 0
    next_peak = get_next_peak(y)
    if next_peak < 0:
        return points
    while start < len(y):
        start += next_peak
        points.append(start)
        next_peak = get_next_peak(y[start + 1:])
        if next_peak < 0:
            break
    return points

'''
根据波峰找出所有的节拍起始点
'''
def get_all_onsets_starts_for_beat(rms,gap):
    points = []
    peak_points = get_all_peak(rms)
    if peak_points:
        peak_points_time = librosa.frames_to_time(peak_points)
        # plt.vlines(peak_points_time, 0,np.max(rms), color='r', linestyle='dashed')
    trough_points = get_all_trough(rms)
    # trough_points.sort()
    # trough_points = np.sort(trough_points, axis=None)
    if trough_points:
        trough_points_time = librosa.frames_to_time(trough_points)
        # plt.vlines(trough_points_time, 0,np.max(rms), color='b', linestyle='dashed')
    all_points = np.hstack((peak_points, trough_points))
    all_points = list(set(all_points))
    all_points.sort()
    print("all_points is {}".format(all_points))
    peak_trough_rms = [rms[x] for x in all_points]
    peak_trough_rms_diff = np.diff(peak_trough_rms)
    print("peak_trough_rms_diff is {}".format(peak_trough_rms_diff))
    # want_all_points = [x for i,x in enumerate(all_points) if i < len(all_points)-1 and (peak_trough_rms_diff[i]>1 or peak_trough_rms_diff[i]<-1)]
    want_all_points = []
    for i in range(len(all_points) - 1):
        if peak_trough_rms_diff[i] > gap:
            want_all_points.append(all_points[i])
        # if peak_trough_rms_diff[i]<-1:
        #     want_all_points.append(all_points[i+1])

    first_max = np.max(rms[0:want_all_points[0]])
    if first_max - rms[want_all_points[0]] > 0.8:
        tmp = rms[0:want_all_points[0]]
        tmp_diff = np.diff(tmp)
        index = [i for i,x in enumerate(tmp_diff) if x>0.3 or x == np.max(tmp_diff)]
        if index[0] == 0:
            index[0] = 1
        want_all_points.insert(0, index[0])
    want_all_points = get_local_best_for_beat(rms, want_all_points, 18)
    return want_all_points

'''
根据波峰找出所有的节拍起始点
'''
def get_all_onsets_starts(rms,gap):
    points = []
    peak_points = get_all_peak(rms)
    if peak_points:
        peak_points_time = librosa.frames_to_time(peak_points)
        # plt.vlines(peak_points_time, 0,np.max(rms), color='r', linestyle='dashed')
    trough_points = get_all_trough(rms)
    # trough_points.sort()
    # trough_points = np.sort(trough_points, axis=None)
    if trough_points:
        trough_points_time = librosa.frames_to_time(trough_points)
        # plt.vlines(trough_points_time, 0,np.max(rms), color='b', linestyle='dashed')
    all_points = np.hstack((peak_points, trough_points))
    all_points = list(set(all_points))
    all_points.sort()
    print("all_points is {}".format(all_points))
    peak_trough_rms = [rms[x] for x in all_points]
    peak_trough_rms_diff = np.diff(peak_trough_rms)
    print("peak_trough_rms_diff is {}".format(peak_trough_rms_diff))
    # want_all_points = [x for i,x in enumerate(all_points) if i < len(all_points)-1 and (peak_trough_rms_diff[i]>1 or peak_trough_rms_diff[i]<-1)]
    want_all_points = []
    for i in range(len(all_points) - 1):
        if peak_trough_rms_diff[i] > gap:
            want_all_points.append(all_points[i])
        # if peak_trough_rms_diff[i]<-1:
        #     want_all_points.append(all_points[i+1])

    first_max = np.max(rms[0:want_all_points[0]])
    if first_max - rms[want_all_points[0]] > 0.8:
        tmp = rms[0:want_all_points[0]]
        tmp_diff = np.diff(tmp)
        index = [i for i,x in enumerate(tmp_diff) if x>0.3 or x == np.max(tmp_diff)]
        if index[0] == 0:
            index[0] = 1
        want_all_points.insert(0, index[0])
    want_all_points = get_local_min(rms, want_all_points, 4)
    return want_all_points
'''
根据波谷找出所有的节拍结束点
'''
def get_all_onsets_ends(rms,gap):
    points = []
    peak_points = get_all_peak(rms)
    if peak_points:
        peak_points_time = librosa.frames_to_time(peak_points)
        # plt.vlines(peak_points_time, 0,np.max(rms), color='r', linestyle='dashed')
    trough_points = get_all_trough(rms)
    # trough_points.sort()
    # trough_points = np.sort(trough_points, axis=None)
    if trough_points:
        trough_points_time = librosa.frames_to_time(trough_points)
        # plt.vlines(trough_points_time, 0,np.max(rms), color='b', linestyle='dashed')
    all_points = np.hstack((peak_points, trough_points))
    all_points = list(set(all_points))
    all_points.sort()
    print("all_points is {}".format(all_points))
    peak_trough_rms = [rms[x] for x in all_points]
    peak_trough_rms_diff = np.diff(peak_trough_rms)
    print("peak_trough_rms_diff is {}".format(peak_trough_rms_diff))
    # want_all_points = [x for i,x in enumerate(all_points) if i < len(all_points)-1 and (peak_trough_rms_diff[i]>1 or peak_trough_rms_diff[i]<-1)]
    want_all_points = []
    for i in range(len(all_points) - 1):
        if peak_trough_rms_diff[i]<gap:
            want_all_points.append(all_points[i+1])
    want_all_points = get_local_min(rms,want_all_points,4)
    return want_all_points

def get_local_min(rms,want_all_points,offset):
    result = []
    for i, x in enumerate(want_all_points):
        if i < len(want_all_points) - 1 and np.max(rms[want_all_points[i]:want_all_points[i+1]]) < 1.5 or rms[want_all_points[i]]>2.5:
            continue

        start = x - offset
        end = x + offset
        if start < 0:
            start = 0
        if end >=len(rms):
            end = len(rms) - 1
        local_rms = rms[start:end]
        if np.min(local_rms) < rms[x]:
            index = np.where(local_rms==np.min(local_rms))
            tmp = start + index[0][0]
            if tmp == 0:
                tmp += 1
            result.append(tmp)
        else:
            result.append(x)
    return result

def get_local_best_for_beat(rms,want_all_points,offset):
    result = []
    for i, x in enumerate(want_all_points):
        if i < len(want_all_points) - 1 and np.max(rms[want_all_points[i]:want_all_points[i+1]]) < 1.0 or rms[want_all_points[i]]>2.5:
            continue

        start = x
        end = x + offset
        if start < 0:
            start = 0
        if end >=len(rms):
            end = len(rms) - 1
        local_rms = rms[start:end]
        local_rms_diff = np.diff(local_rms)
        local_rms_diff = [local_rms_diff[i] + local_rms_diff[i+1] for i in range(len(local_rms_diff)-1)]
        index = np.where(local_rms_diff == np.max(local_rms_diff))

        tmp = start + index[0][0]
        if tmp == 0:
            tmp += 1
        result.append(tmp)
    return result

def get_all_peak(y):
    points = []
    start = 0
    next_peak = get_next_peak(y)
    if next_peak < 0:
        return points
    while start < len(y):
        start += next_peak
        points.append(start)
        next_peak = get_next_peak(y[start + 1:])
        if next_peak < 0:
            break
    return points

'''
找所有波谷
'''
def get_all_trough(y):
    points = []
    start = 0
    next_trough = get_next_trough(y)
    if next_trough < 0:
        return points
    while start < len(y):
        start += next_trough
        points.append(start)
        next_trough = get_next_trough(y[start + 1:])
        if next_trough < 0:
            break
    return points

'''
找所有上升沿的起点
'''


def get_min_max_total(s):
    if s is None or len(s) < 1:
        print("input is empty")

    s = s.replace('[','').replace(']','')
    tmp = s.split(';')
    print(tmp)
    result = []
    for c in tmp:
        if c.find(","):  # 包括","的情况，即有多个数值
            cc = c.split(",")
            for ccc in cc:
                if ccc.find("(") > 0: # 空音的情况
                    score = re.sub("\D", "", ccc)  # 筛选数字
                    score = "-" + score
                    result.append(score)
                else:
                    result.append(ccc)
        else: # 不包括","的情况，即只有一个数值
            if c.find("(") > 0:  # 空音的情况
                score = re.sub("\D", "", c)  # 筛选数字
                score = -1 * score
                result.append(score)
            else:
                result.append(c)
    result = [int(x) for x in result]
    min = np.min(result)
    max = np.max(result)
    total = np.sum(result)
    last = result[-1]
    return min,max,last,total

def get_real_onsets_frames(y):
    y_max = max(y)
    # y = np.array([x if x > y_max*0.01 else y_max*0.01 for x in y])
    # 获取每个帧的能量
    energy = librosa.feature.rmse(y)
    print(np.mean(energy))
    energy_diff = np.diff(energy)
    #print(energy_diff)
    onsets_frames = librosa.onset.onset_detect(y)

    print(onsets_frames)
    print(np.diff(onsets_frames))

    some_y = [energy[0][x] for x in onsets_frames]
    print("some_y is {}".format(some_y)) # 节拍点对应帧的能量
    energy_mean = (np.sum(some_y) - np.max(some_y))/(len(some_y)-1)  # 获取能量均值
    print("energy_mean for some_y is {}".format(energy_mean))
    energy_gap = energy_mean * 0.3
    some_energy_diff = [energy_diff[0][x] if x < len(energy_diff) else energy_diff[0][x-1]  for x in onsets_frames]
    energy_diff_mean = np.mean(some_energy_diff)
    print("some_energy_diff is {}".format(some_energy_diff))
    print("energy_diff_meanis {}".format(energy_diff_mean))
    onsets_frames = [x for x in onsets_frames if energy[0][x] > energy_gap]  # 筛选能量过低的伪节拍点

    r,c = energy_diff.shape
    if onsets_frames[-1] >= c:
        first = onsets_frames[0]
        last = onsets_frames[-1]
        onsets_frames = [x for x in onsets_frames[1:-1] if energy[0][x] > energy[0][x - 1] and energy[0][x + 1] > energy[0][x]]  # 只选择上升沿的节拍点
        onsets_frames.append(last)
        onsets_frames.insert(0,first)
    else:
        first = onsets_frames[0]
        onsets_frames = [x for x in onsets_frames[1:] if energy[0][x] > energy[0][x -1] and energy[0][x + 1] > energy[0][x] ]  # 只选择上升沿的节拍点
        onsets_frames.insert(0,first)


    # 筛选过密的节拍点
    onsets_frames_new = []
    for i in range(0, len(onsets_frames)):
        if i == 0:
            onsets_frames_new.append(onsets_frames[i])
            continue
        if onsets_frames[i] - onsets_frames[i - 1] <= 3:
            middle = int((onsets_frames[i] + onsets_frames[i - 1]) / 2)
            # middle = onsets_frames[i]
            onsets_frames_new.pop()
            onsets_frames_new.append(middle)
        else:
            onsets_frames_new.append(onsets_frames[i])
    onsets_frames = onsets_frames_new
    return onsets_frames

def get_bigin(y,onsets_first):
    y_max = max(y)
    # y = np.array([x if x > y_max*0.01 else y_max*0.01 for x in y])
    # 获取每个帧的能量
    energy = librosa.feature.rmse(y)
    print(np.mean(energy))
    energy_diff = np.diff(energy)
    #print(energy_diff)
    onsets_frames = librosa.onset.onset_detect(y)

    print(onsets_frames)
    print(np.diff(onsets_frames))

    some_energy = energy[0][0:onsets_first-1]
    if np.max(some_energy) > energy[0][onsets_first] * 1.3:
        return np.argmax(some_energy)
    else:
        return onsets_first

def get_real_onsets_frames_by_strength(y,sr):
    onset_env = librosa.onset.onset_strength(y=y, sr=sr,
                                             aggregate=np.median,
                                             fmax=8000, n_mels=512)

    max_onset_env = [i for i, x in enumerate(onset_env[1:-1])
                     if onset_env[i] > onset_env[i - 1]
                     and onset_env[i] > onset_env[i + 1]
                     and onset_env[i] > np.max(onset_env) * 0.5]
    return max_onset_env

'''
  结合三种特征（onset_strength、onset_strength_median、CQT）,通过阀值来提取节拍点,
'''
def get_onsets_by_all(y,sr):
    all_onset = []

    gap1 = 0.5
    gap2 = 0.5
    gap3 = 0.75
    gap4 = 5
    onset_env_v1 = librosa.onset.onset_strength(y=y, sr=sr)
    max_onset_env_v1 = [x if onset_env_v1[i] > onset_env_v1[i - 1] and onset_env_v1[i] > onset_env_v1[i + 1] and onset_env_v1[i] > np.max(
        onset_env_v1) * gap1 else 0 for i, x in enumerate(onset_env_v1[1:-1])]
    max_onset_env_v1.append(0)
    max_onset_env_v1.insert(0, 0)
    max_onset_env_index = [i for i, x in enumerate(onset_env_v1[1:-1]) if
                           onset_env_v1[i] > onset_env_v1[i - 1] and onset_env_v1[i] > onset_env_v1[i + 1] and onset_env_v1[
                               i] > np.max(onset_env_v1) * gap1]
    print("max_onset_env_index is {}".format(max_onset_env_index))

    all_onset = np.hstack((all_onset, max_onset_env_index))


    onset_env_v2 = librosa.onset.onset_strength(y=y, sr=sr,
                                             aggregate=np.median,
                                             fmax=8000, n_mels=512)
    # print("onset_env is {}".format(onset_env))
    max_onset_env_v2 = [x if onset_env_v2[i] > onset_env_v2[i - 1] and onset_env_v2[i] > onset_env_v2[i + 1] and onset_env_v2[i] > np.max(
        onset_env_v2) * gap2 else 0 for i, x in enumerate(onset_env_v2[1:-1])]
    max_onset_env_v2.append(0)
    max_onset_env_v2.insert(0, 0)
    max_onset_env_index = [i for i, x in enumerate(onset_env_v2[1:-1]) if
                           onset_env_v2[i] > onset_env_v2[i - 1] and onset_env_v2[i] > onset_env_v2[i + 1] and onset_env_v2[
                               i] > np.max(onset_env_v2) * gap2]
    print("max_onset_env_index is {}".format(max_onset_env_index))
    all_onset = np.hstack((all_onset, max_onset_env_index))


    onset_env_v3 = librosa.onset.onset_strength(y=y, sr=sr,
                                             feature=librosa.cqt)

    max_onset_env_v3 = [x if onset_env_v3[i] > onset_env_v3[i - 1] and onset_env_v3[i] > onset_env_v3[i + 1] and onset_env_v3[i] > np.max(
        onset_env_v3) * gap3 else 0 for i, x in enumerate(onset_env_v3[1:-1])]
    max_onset_env_v3.append(0)
    max_onset_env_v3.insert(0, 0)
    max_onset_env_index = [i for i, x in enumerate(onset_env_v3[1:-1]) if
                           onset_env_v3[i] > onset_env_v3[i - 1] and onset_env_v3[i] > onset_env_v3[i + 1] and onset_env_v3[
                               i] > np.max(onset_env_v3) * gap3]
    print("max_onset_env_index is {}".format(max_onset_env_index))

    all_onset = np.hstack((all_onset, max_onset_env_index))
    news_ids = []
    for id in all_onset:
        if id not in news_ids:
            news_ids.append(int(id))
    all_onset = news_ids
    all_onset.sort()
    all_onset_diff = np.diff(all_onset)

    result = [all_onset[0]]
    for i,v in enumerate(all_onset_diff):
        if v > gap4:
            result.append(all_onset[i+1])
        else:
            max1 = np.max([max_onset_env_v1[i],max_onset_env_v2[i],max_onset_env_v3[i]])
            max2 = np.max([max_onset_env_v1[i+1], max_onset_env_v2[i+1], max_onset_env_v3[i+1]])
            if max1 >= max2:
                continue
            else:
                tmp = result.pop()
                #result.append(tmp + int((all_onset[i+1] - tmp)/2))
                result.append(all_onset[i+1])
    print("all_onset is {}".format(result))
    # 获取起始点
    first_frame = get_bigin(y, result[0])
    if first_frame < result[0]:
        if first_frame == 0:
            first_frame = 1
        result.insert(0, first_frame)
    result_strength  ={}

    for x in result:
        max1 = np.max([onset_env_v1[x]/np.max(onset_env_v1), onset_env_v2[x]/np.max(onset_env_v2), onset_env_v3[x]/np.max(onset_env_v3)])
        result_strength[x] = max1
    return  result,result_strength

'''
  结合三种特征（onset_strength、onset_strength_median、CQT）,通过阀值来提取节拍点,
'''
def get_onsets_by_all_v2(y,sr,onsets_total):
    all_onset = []

    gap4 = 10
    onset_env_v1 = librosa.onset.onset_strength(y=y, sr=sr)
    max_onset_env_index = find_n_largest(onset_env_v1,onsets_total)
    max_onset_env_v1 = [onset_env_v1[i] if i in max_onset_env_index else 0   for i in range(0,len(onset_env_v1))]

    print("max_onset_env_index is {}".format(max_onset_env_index))

    all_onset = np.hstack((all_onset, max_onset_env_index))


    onset_env_v2 = librosa.onset.onset_strength(y=y, sr=sr,
                                             aggregate=np.median,
                                             fmax=8000, n_mels=512)
    # print("onset_env is {}".format(onset_env))
    max_onset_env_index = find_n_largest(onset_env_v2, onsets_total)
    max_onset_env_v2 = [onset_env_v2[i] if i in max_onset_env_index else 0 for i in range(0, len(onset_env_v2))]

    print("max_onset_env_index is {}".format(max_onset_env_index))
    all_onset = np.hstack((all_onset, max_onset_env_index))


    onset_env_v3 = librosa.onset.onset_strength(y=y, sr=sr,
                                             feature=librosa.cqt)

    max_onset_env_index = find_n_largest(onset_env_v3, onsets_total)
    max_onset_env_v3 = [onset_env_v3[i] if i in max_onset_env_index else 0 for i in range(0, len(onset_env_v3))]

    print("max_onset_env_index is {}".format(max_onset_env_index))

    all_onset = np.hstack((all_onset, max_onset_env_index))
    news_ids = []
    for id in all_onset:
        if id not in news_ids:
            news_ids.append(int(id))
    all_onset = news_ids
    all_onset.sort()
    all_onset_diff = np.diff(all_onset)

    result = [all_onset[0]]
    for i,v in enumerate(all_onset_diff):
        if v > gap4:
            result.append(all_onset[i+1])
        else:
            max1 = np.max([max_onset_env_v1[i],max_onset_env_v2[i],max_onset_env_v3[i]])
            max2 = np.max([max_onset_env_v1[i+1], max_onset_env_v2[i+1], max_onset_env_v3[i+1]])
            if max1 >= max2:
                continue
            else:
                result.pop()
                result.append(all_onset[i+1])
    print("all_onset is {}".format(result))
    # 获取起始点
    first_frame = get_bigin(y, result[0])
    if first_frame < result[0]:
        if first_frame == 0:
            first_frame = 1
        result.insert(0, first_frame)
    return  result

def find_n_largest(a,total):
    import heapq

    a = list(a)
    #a = [43, 5, 65, 4, 5, 8, 87]

    re1 = heapq.nlargest(total, a)  # 求最大的三个元素，并排序
    re1.sort()
    #re2 = map(a.index, heapq.nlargest(total, a))  # 求最大的三个索引    nsmallest与nlargest相反，求最小
    re2 = [i for i,x in enumerate(a) if x in re1]

    print(re1)
    print(list(re2))  # 因为re1由map()生成的不是list，直接print不出来，添加list()就行了
    return list(re2)
def max_min(x, y, z):
    max = min = x
    if y > max:
        max = y
    else:
        min = y
    if z > max:
        max = z
    else:
        min = z
    return (max, min)

def get_real_onsets_frames_rhythm(y):
    y_max = max(y)
    # y = np.array([x if x > y_max*0.01 else y_max*0.01 for x in y])
    # 获取每个帧的能量
    energy = librosa.feature.rmse(y)
    print(np.mean(energy))
    energy_diff = np.diff(energy)
    #print(energy_diff)
    onsets_frames = librosa.onset.onset_detect(y)

    print(onsets_frames)
    print(np.diff(onsets_frames))

    some_y = [energy[0][x] for x in onsets_frames]
    print("some_y is {}".format(some_y)) # 节拍点对应帧的能量
    energy_mean = (np.sum(some_y) - np.max(some_y))/(len(some_y)-1)  # 获取能量均值
    print("energy_mean for some_y is {}".format(energy_mean))
    energy_gap = energy_mean * 0.3
    some_energy_diff = [energy_diff[0][x] if x < len(energy_diff) else energy_diff[0][x-1]  for x in onsets_frames]
    energy_diff_mean = np.mean(some_energy_diff)
    print("some_energy_diff is {}".format(some_energy_diff))
    print("energy_diff_meanis {}".format(energy_diff_mean))
    onsets_frames = [x for x in onsets_frames if energy[0][x] > energy_gap]  # 筛选能量过低的伪节拍点

    # 筛选过密的节拍点
    onsets_frames_new = []
    for i in range(0, len(onsets_frames)):
        if i == 0:
            onsets_frames_new.append(onsets_frames[i])
            continue
        if onsets_frames[i] - onsets_frames[i - 1] <= 7:
            middle = int((onsets_frames[i] + onsets_frames[i - 1]) / 2)
            # middle = onsets_frames[i]
            onsets_frames_new.pop()
            onsets_frames_new.append(middle)
        else:
            onsets_frames_new.append(onsets_frames[i])
    onsets_frames = onsets_frames_new
    return onsets_frames

def get_onsets_frames_by_cqt_for_rhythm(y,sr):
    gap4 = 15
    cqt_gap = 0.3
    onset_env = librosa.onset.onset_strength(y=y, sr=sr,
                                             aggregate=np.median,
                                             fmax=8000, n_mels=512)
    # print("onset_env is {}".format(onset_env))
    max_onset_env = [x if onset_env[i] > onset_env[i - 1] and onset_env[i] > onset_env[i + 1] and onset_env[i] > np.max(
        onset_env) * cqt_gap else 0 for i, x in enumerate(onset_env[1:-1])]
    max_onset_env.append(0)
    max_onset_env.insert(0, 0)
    max_onset_env_index = [i for i, x in enumerate(onset_env[1:-1]) if
                           onset_env[i] > onset_env[i - 1] and onset_env[i] > onset_env[i + 1] and onset_env[
                               i] > np.max(onset_env) * cqt_gap]
    print("max_onset_env_index is {}".format(max_onset_env_index))

    all_onset = []
    all_onset = np.hstack((all_onset, max_onset_env_index))
    news_ids = []
    for id in all_onset:
        if id not in news_ids:
            news_ids.append(int(id))
    all_onset = news_ids
    all_onset.sort()
    all_onset_diff = np.diff(all_onset)

    result = [all_onset[0]]
    for i, v in enumerate(all_onset_diff):
        if v > gap4:
            result.append(all_onset[i + 1])
        else:
            pass
            # max1 = max_onset_env[i]
            # max2 = max_onset_env[i + 1]
            # if max1 >= max2:
            #     continue
            # else:
            #     result.pop()
            #     result.append(all_onset[i + 1])
    print("all_onset is {}".format(result))
    # 获取起始点
    first_frame = get_bigin(y, result[0])
    if first_frame < result[0]:
        if first_frame == 0:
            first_frame = 1
        result.insert(0, first_frame)
    return result


def get_onsets_index_by_filename(filename):
    if filename.find("节奏1") >= 0 or filename.find("节奏一") >= 0 or filename.find("节奏题一") >= 0 or filename.find("节奏题1") >= 0:
        return 0
    elif filename.find("节奏2") >= 0 or filename.find("节奏二") >= 0 or filename.find("节奏题二") >= 0 or filename.find("节奏题2") >= 0:
        return 1
    elif filename.find("节奏3") >= 0 or filename.find("节奏三") >= 0 or filename.find("节奏题三") >= 0 or filename.find("节奏题3") >= 0:
        return 2
    elif filename.find("节奏4") >= 0 or filename.find("节奏四") >= 0 or filename.find("节奏题四") >= 0 or filename.find("节奏题4") >= 0:
        return 3
    elif filename.find("节奏5") >= 0 or filename.find("节奏五") >= 0 or filename.find("节奏题五") >= 0 or filename.find("节奏题5") >= 0:
        return 4
    elif filename.find("节奏6") >= 0 or filename.find("节奏六") >= 0 or filename.find("节奏题六") >= 0 or filename.find("节奏题6") >= 0:
        return 5
    elif filename.find("节奏7") >= 0 or filename.find("节奏七") >= 0 or filename.find("节奏题七") >= 0 or filename.find("节奏题7") >= 0:
        return 6
    elif filename.find("节奏8") >= 0 or filename.find("节奏八") >= 0 or filename.find("节奏题八") >= 0 or filename.find("节奏题8") >= 0:
        return 7
    elif filename.find("节奏9") >= 0 or filename.find("节奏九") >= 0 or filename.find("节奏题九") >= 0 or filename.find("节奏题9") >= 0:
        return 8
    elif filename.find("节奏10") >= 0 or filename.find("节奏十") >= 0 or filename.find("节奏题十") >= 0 or filename.find("节奏题10") >= 0:
        return 9
    else:
        return -1

def get_onsets_index_by_filename_rhythm(filename):
    if filename.find("旋律1") >= 0 or filename.find("旋律一") >= 0 or filename.find("视唱一") >= 0 or filename.find("视唱1") >= 0:
        return 0
    elif filename.find("旋律2") >= 0 or filename.find("旋律二") >= 0 or filename.find("视唱二") >= 0 or filename.find("旋律题2") >= 0:
        return 1
    elif filename.find("旋律3") >= 0 or filename.find("旋律三") >= 0 or filename.find("视唱三") >= 0 or filename.find("旋律题3") >= 0:
        return 2
    elif filename.find("旋律4") >= 0 or filename.find("旋律四") >= 0 or filename.find("视唱四") >= 0 or filename.find("视唱4") >= 0:
        return 3
    elif filename.find("旋律5") >= 0 or filename.find("旋律五") >= 0 or filename.find("视唱五") >= 0 or filename.find("视唱5") >= 0:
        return 4
    elif filename.find("旋律6") >= 0 or filename.find("旋律六") >= 0 or filename.find("视唱六") >= 0 or filename.find("视唱6") >= 0:
        return 5
    elif filename.find("旋律7") >= 0 or filename.find("旋律七") >= 0 or filename.find("视唱七") >= 0 or filename.find("视唱7") >= 0:
        return 6
    elif filename.find("旋律8") >= 0 or filename.find("旋律八") >= 0 or filename.find("视唱八") >= 0 or filename.find("视唱8") >= 0:
        return 7
    elif filename.find("旋律9") >= 0 or filename.find("旋律九") >= 0 or filename.find("视唱九") >= 0 or filename.find("视唱9") >= 0:
        return 8
    elif filename.find("旋律10") >= 0 or filename.find("旋律十") >= 0 or filename.find("视唱十") >= 0 or filename.find("视唱10") >= 0:
        return 9
    else:
        return -1

def get_total_frames_number(path):
    audio, sr = librosa.load(path)
    energy = librosa.feature.rmse(audio)
    frames = np.nonzero(energy >= np.max(energy) / 5)

    total = frames[1][-1]

    return total

def get_onset_rmse_viterbi(y,silence_threshold):
    times, states = get_viterbi_state(y, silence_threshold)
    states_diff = np.diff(states)
    result = [i for i, x in enumerate(states_diff) if x == 1]
    return times,result

def get_nearly_note(input,step):
    for x in range(0,len(input),step):
        min_gap = 10000
        if np.abs(input[x] - const.do) < min_gap:
            min = const.do
            min_gap = np.abs(input[x] - const.do)
        if np.abs(input[x] - const.do_up) < min_gap:
            min = const.do_up
            min_gap = np.abs(input[x] - const.do_up)
        if np.abs(input[x] - const.re) < min_gap:
            min = const.re
            min_gap = np.abs(input[x] - const.re)
        if np.abs(input[x] - const.re_up) < min_gap:
            min = const.re_up
            min_gap = np.abs(input[x] - const.re_up)
        if np.abs(input[x] - const.mi) < min_gap:
            min = const.mi
            min_gap = np.abs(input[x] - const.mi)
        if np.abs(input[x] - const.fa) < min_gap:
            min = const.fa
            min_gap = np.abs(input[x] - const.fa)
        if np.abs(input[x] - const.fa_up) < min_gap:
            min = const.fa_up
            min_gap = np.abs(input[x] - const.fa_up)
        if np.abs(input[x] - const.so) < min_gap:
            min = const.so
            min_gap = np.abs(input[x] - const.so)
        if np.abs(input[x] - const.so_up) < min_gap:
            min = const.so_up
            min_gap = np.abs(input[x] - const.so_up)
        if np.abs(input[x] - const.la) < min_gap:
            min = const.la
            min_gap = np.abs(input[x] - const.la)
        if np.abs(input[x] - const.la_up) < min_gap:
            min = const.la_up
            min_gap = np.abs(input[x] - const.la_up)
        if np.abs(input[x] - const.xi) < min_gap:
            min = const.xi
            min_gap = np.abs(input[x] - const.xi)
        start = x
        end = np.min([x + step,len(input)])

        input[start:end] = min
    return input
if __name__ == '__main__':
    start_point = 0.2
    time = 6.45
    #code = '[0500,0500;1000;1500;0500;0250,0250,0250,0250;1000]'
    code = '[500,500,1000;500,500,1000;500,500,750,250;2000]'
    #code = '[2000;1000,1000;500,500,1000;2000]'
    #code = '[1000,1000;500,500,1000;1000,1000;2000]'
    #code = '[1000,--(1000);1000,--(1000);500,250,250,1000;--(1000),1000]'
    #code = '[500;1000,500,1000,500;500,500,500,250,250,500,500;250,250,500,500,1000]'
    #code = '[1000,--(1000);1000,--(1000);1000,-(500),500;1000,1000]'
    #code = '[750,250,500,500,500,-(500);500,1000,500,500,-(500);750,250,500,500,500,-(500)]'
    ds = onsets_base(code,time,start_point)


    plt.vlines(ds[:-1], 0, 2400, color='black', linestyle='dashed')
    plt.vlines(ds[-1], 0, 2400, color='white', linestyle='dashed')
    #plt.vlines(time, 0, 2400, color='white', linestyle='dashed')

    pitch_code = '[3,3,3,3,3,3,3,5,1,2,3]'
    chroma_pitch = get_chroma_pitch(pitch_code)
    print(chroma_pitch)
    plt.show()