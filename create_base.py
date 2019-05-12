# -*- coding:utf-8 -*-
import threading

import  numpy as np
import librosa
import matplotlib.pyplot as plt
import re
import math
from viterbi import *
from note_frequency import *
from filters import *
import os
from multiprocessing import Pool
import multiprocessing
import time
import threading


pitch_base = ['C','D','E','F','G','A','B']
pitch_number = ['1','2','3','4','5','6','7']
pitch_v = [0,2,4,5,7,9,11]

def load_and_trim(path,threshold=5):
    audio, sr = librosa.load(path)
    energy = librosa.feature.rmse(audio)
    frames = np.nonzero(energy >= np.max(energy) / threshold)
    indices = librosa.core.frames_to_samples(frames)[1]
    audio = audio[indices[0]:indices[-1]] if indices.size else audio[0:0]

    return audio, sr



def get_basetime(s):
    if s is None or len(s) < 1:
        print("input is empty")

    s = s.replace('[','').replace(']','')
    tmp = s.split(';')
    #print(tmp)
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
        if str(x).find("+") > 0:
            result.append(pitch_v[int(x[0]) - 1] + 12)
        elif str(x).find("-") > 0:
            result.append(pitch_v[int(x[0]) - 1] - 12)
        elif str(x).find("0") > 0:
            result.append(-1)
        else:
            result.append(pitch_v[int(x[0])-1])
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
    #print(result)
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


def onsets_base_frames_rhythm(rhythm_code,frames_number):
    result = get_basetime(rhythm_code)
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
    #ds = [x - math.ceil(frames_number * int(result[0]) / total) for x in ds]
    #ds.append(frames_number-5)
    return ds

def onsets_base_frames_for_note(filename,rhythm_code):
    #frames_total = get_total_frames_number_for_note(filename)
    #frames_total = get_total_frames_number(filename)
    y, sr = librosa.load(filename)
    start, end = get_start_and_end_for_note(y, sr)
    frames_total = end - start
    print("frames_total is {}".format(frames_total))
    #type_index = get_onsets_index_by_filename_rhythm(filename)
    base_frames = onsets_base_frames_rhythm(rhythm_code, frames_total)
    return base_frames,frames_total

def base_note(filename,pitch_code):
    #type_index = get_onsets_index_by_filename_rhythm(filename)
    #pitch_code = note_codes[type_index]
    chroma_pitch = get_chroma_pitch(pitch_code)
    #print(chroma_pitch)
    return chroma_pitch

def add_base_note_to_cqt(cqt,base_notes,base_frames,end,filename,rhythm_code):
    #type_index = get_onsets_index_by_filename_rhythm(filename)
    codes = get_basetime(rhythm_code)
    for i in range(len(base_frames)-1):
        start_frame = base_frames[i]
        if codes[i+1].find("-") >= 0:
            last = int(codes[i-1])
            current = int(re.sub("\D", "", codes[i+1]))  # 筛选数字
            end_frame = base_frames[i + 1] - (base_frames[i + 1] - base_frames[i])*(current/(current + last))
            end_frame = int(end_frame)
        else:
            end_frame = base_frames[i+1]
        note = base_notes[i]
        #print("note,start_frame,end_frame is {},{},{}".format(note,start_frame,end_frame))
        if note != -1:
            cqt[note,start_frame:end_frame] = -20
    cqt[base_notes[-1],base_frames[-1]:end] = -20
    return cqt

def add_base_note_to_cqt_for_filename(filename,rhythm_code,first_frame=[],CQT=[]):
    y, sr = librosa.load(filename)
    start, end = get_start_and_end_for_note(y, sr)
    base_frames = onsets_base_frames_for_note(filename)
    if first_frame:
        base_frames = [x + first_frame - base_frames[0] for x in base_frames]
    else:
        base_frames = [x + start - base_frames[0] for x in base_frames]

    if len(CQT) < 1:
        CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref=np.max)
    w, h = CQT.shape
    CQT[0:20, :] = np.min(CQT)
    base_notes = base_note(filename)
    base_notes = [x + 5 - np.min(base_notes) for x in base_notes]
    #type_index = get_onsets_index_by_filename_rhythm(filename)
    codes = get_basetime(rhythm_code)
    for i in range(len(base_frames)-1):
        start_frame = base_frames[i]
        if codes[i+1].find("-") >= 0:
            last = int(codes[i-1])
            current = int(re.sub("\D", "", codes[i+1]))  # 筛选数字
            end_frame = base_frames[i + 1] - (base_frames[i + 1] - base_frames[i])*(current/(current + last))
            end_frame = int(end_frame)
        else:
            end_frame = base_frames[i+1]
        note = base_notes[i]
        #print("note,start_frame,end_frame is {},{},{}".format(note,start_frame,end_frame))
        if note != -1:
            CQT[note,start_frame:end_frame] = -20
    CQT[base_notes[-1],base_frames[-1]:end] = -20
    return CQT,base_notes

def add_base_note_to_cqt_for_filename_by_base_notes(filename,base_frames,first_frame=[],CQT=[],base_notes=[]):
    y, sr = librosa.load(filename)
    start, end = get_start_and_end_for_note(y, sr)

    if len(CQT) < 1:
        CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref=np.max)
    w, h = CQT.shape
    CQT[0:10, :] = np.min(CQT)
    if len(base_notes) < 1:
        base_notes = base_note(filename)
    base_notes = [x + 2 - np.min(base_notes) for x in base_notes]
    # type_index = get_onsets_index_by_filename_rhythm(filename)
    # codes = get_basetime(rhythm_codes[type_index])
    length = len(base_notes) if len(base_notes) < len(base_frames) else len(base_frames)
    for i in range(length -1):
        start_frame = base_frames[i]
        end_frame = base_frames[i+1]
        note = base_notes[i]
        #print("note,start_frame,end_frame is {},{},{}".format(note,start_frame,end_frame))
        if note != -1:
            CQT[note,start_frame:end_frame] = -20
    CQT[base_notes[-1],base_frames[-1]:end] = -20
    return CQT,base_notes

'''
根据当前位置获取最小帧距
'''
def get_min_range_frames_rhythm(frame_numbers,current_frames_number):
    #所有节拍的起点
    #frame_numbers =onsets_base_frames_rhythm(index, frames_total)

    frame_numbers_diff = np.diff(frame_numbers)

    #判断当前位置位于哪一个节拍中
    index = [i for i in range(0,len(frame_numbers)-1) if frame_numbers[i] <= current_frames_number and frame_numbers[i+1] >= current_frames_number ]

    if len(index)<1:
        return np.min(frame_numbers)

    start = index[0] - 1
    if start < 0:
        start = 0
    end = index[0] + 2
    if end > len(frame_numbers_diff):
        end = len(frame_numbers_diff)
    sub_frame_numbers_diff = frame_numbers_diff[start:end]
    min_frame_width = int(np.min(sub_frame_numbers_diff) * 0.4)
    return min_frame_width

'''
获取最小帧距
'''
def get_min_width_rhythm(filename,rhythm_code):
    total_frames_number = get_total_frames_number(filename)
    base_frames = onsets_base_frames_rhythm(rhythm_code, total_frames_number)
    base_frames_diff = np.diff(base_frames)
    result = np.min(base_frames_diff)
    return result

'''
获取最小帧距
'''
def get_min_width_onsets(filename,code):
    total_frames_number = get_total_frames_number(filename)
    base_frames = onsets_base_frames(code, total_frames_number)
    base_frames_diff = np.diff(base_frames)
    result = np.min(base_frames_diff)
    return result

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
        if len(index)>0:
            if index[0] == 0:
                index[0] = 1
            want_all_points.insert(0, index[0])
    want_all_points = get_local_best_for_beat(rms, want_all_points, 20)
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

def get_onsets_by_cqt_rms(y, sr,base_frames,threshold):
    CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=sr), ref=np.max)
    w, h = CQT.shape
    CQT[50:w, :] = -100
    CQT[0:20, :] = -100
    rms = librosa.feature.rmse(y=y)[0]
    rms = [x / np.std(rms) for x in rms]
    rms = [x / np.std(rms) if x / np.std(rms) > np.max(rms)*0.45 else 0 for x in rms]
    #rms = rms / np.std(rms)
    all_peak_points = get_all_onsets_starts(rms, threshold)
    c_max = np.argmax(CQT, axis=0)
    c_max = deburring(c_max, 10)
    note_min_step = 7 #最小节拍长度
    all_notes_start = find_all_notes_start(c_max,base_frames, note_min_step)
    for x in all_notes_start:
        note_start,note_end,_ = x
        if rms[note_start] > rms[note_start +1]: #下降沿的大概率不是节拍起点
            continue
        indexs = [1  for x in all_peak_points if np.abs(x - note_start) <10]
        if len(indexs)<1 and rms[note_start]>0.7 and rms[note_start]<3:
            all_peak_points.append(note_start)
    all_peak_points = list(set(all_peak_points))
    all_peak_points.sort()
    return all_peak_points

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
        if i < len(want_all_points) - 1 and (np.max(rms[want_all_points[i]:want_all_points[i+1]]) < 1.0 or rms[want_all_points[i]]>2.5):
            continue
        if i < len(want_all_points) - 1:
            sub_rms = rms[want_all_points[i]:want_all_points[i+1]]
            if np.max(sub_rms)- want_all_points[i]>1.5:
                offset = np.where(sub_rms == np.max(sub_rms))[0][0]

        start = x
        end = x + offset
        if start < 0:
            start = 0
        if end >=len(rms):
            end = len(rms) - 1
        local_rms = rms[start:end]
        local_rms_diff = np.diff(local_rms)
        if len(local_rms_diff)>1:
            local_rms_diff = [local_rms_diff[i] + local_rms_diff[i+1] for i in range(len(local_rms_diff)-1)]
            index = np.where(local_rms_diff == np.max(local_rms_diff))
            tmp = start + index[0][0]
        else:
            tmp = start
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

def find_note_number_by_range(cqt_max,cerrent,next):
    cqt_max_sub = cqt_max[cerrent:next]
    if len(cqt_max_sub)<2:
        return -1,-1,-1
    a_diff = np.diff(cqt_max_sub)
    # a_find = [i for i in range(1, len(a_diff)) if
    #           (a_diff[i - 1] != 0 and a_diff[i] == 0) or (a_diff[i - 1] == 0 and a_diff[i] != 0)]
    starts = [i for i in range(1, len(a_diff)) if (a_diff[i - 1] != 0 and a_diff[i] == 0)]
    ends = [i for i in range(0, len(a_diff) - 1) if (a_diff[i] == 0 and a_diff[i + 1] != 0 and i> 0)]
    if a_diff[0] == 0 and a_diff[1] == 0:
        starts.insert(0,0)
    # starts_ends = zip(starts,ends)
    #print(a_diff)
    #print(a_find)
    max = -1
    max_x = -1
    max_y = -1
    for x, y in zip(starts, ends):
        #print(x, y)
        if y - x > max:
            max = y - x
            max_x = x
            max_y = y
    #print(max_x, max_y)
    if -1 == max_x:
        print("note number no found,param is {},{}".format(cerrent,next))
        return -1,-1,-1
    else:
        return cerrent + max_x,cerrent + max_y,cqt_max[cerrent + max_x]
def find_all_notes_start(cqt_max,base_frames,note_min_step):
    result = []
    cqt_max_sub = cqt_max
    if len(cqt_max_sub)<1:
        return -1,-1,-1
    a_diff = np.diff(cqt_max_sub)
    # a_find = [i for i in range(1, len(a_diff)) if
    #           (a_diff[i - 1] != 0 and a_diff[i] == 0) or (a_diff[i - 1] == 0 and a_diff[i] != 0)]
    starts = [i for i in range(1, len(a_diff)) if (a_diff[i - 1] != 0 and a_diff[i] == 0)]
    ends = [i for i in range(0, len(a_diff) - 1) if (a_diff[i] == 0 and a_diff[i + 1] != 0 and i> 0)]
    if a_diff[0] == 0 and a_diff[1] == 0:
        starts.insert(0,0)

    for x, y in zip(starts, ends):
        #print(x, y)
        note_min_step = get_min_range_frames_rhythm(base_frames, x)
        if y - x > note_min_step:
            result.append([x,y,cqt_max[x]])
    return result

def find_note_number(note_number,find_note):
    if note_number <= 11:
        offset = 0
    elif note_number >=12 and note_number <=23:
        offset = 12
    elif note_number >=24 and note_number <=35:
        offset = 24
    elif note_number >= 36 and note_number <= 47:
        offset = 36
    elif note_number >= 48 and note_number <= 59:
        offset = 48
    elif note_number >= 60 and note_number <= 71:
        offset = 60
    if 1 == find_note:
        return [int(offset + 0),int(offset + 1)]
    elif 2 == find_note:
        return [int(offset + 2),int(offset + 3)]
    elif 3 == find_note:
        return [int(offset + 4)]
    elif 4 == find_note:
        return [int(offset + 5), int(offset + 6)]
    elif 5 == find_note:
        return [int(offset + 7), int(offset + 8)]
    elif 6 == find_note:
        return [int(offset + 9), int(offset + 10)]
    elif 7 == find_note:
        return [int(offset + 11)]



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

    gap1 = 0.01 #0.5->1
    gap2 = 0.01 #0.5->1
    gap3 = 0.01
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
def get_peak_trough_by_denoise(raw_rms,rms,threshold,min_waterline):
    result = []
    all_max_sub_rms = []
    #rms = [x / np.std(rms) if x / np.std(rms) > np.max(rms) * threshold else 0 for x in rms]
    starts = [i for i in range(0,len(rms)-1) if rms[i] == 0 and rms[i+1] > threshold ]
    ends = [i for i in range(0,len(rms)-1) if rms[i] > threshold and rms[i + 1] == 0 ]
    if len(starts) == 0 or len(ends) == 0:
        return [],[]
    if ends[0]<starts[0]:
        #ends.pop(0)
        if starts[0] != 1:
            starts.insert(0,1)
    else:
        ends.append(len(rms) - 1)
    #print("====================================")
    for i in range(0,len(starts)):
        start = starts[i]
        if i>=len(ends):
            end = len(rms)-1
        else:
            end = ends[i]
        if np.abs(end - start)<1 or start > end:
            continue
        #print("start,end is {},{}".format(start,end))
        max_sub_rms = np.max(rms[start:end])
        if i==0:
            result.append([start, end, max_sub_rms])
            all_max_sub_rms.append(max_sub_rms)
        else:
            last_start = starts[i-1]
            min_sub_rms = np.min(raw_rms[last_start:start])
            #print("start,end,min_sub_rms,min_waterline is {},{},{},{}".format(start, end,min_sub_rms,min_waterline))
            if min_sub_rms <= min_waterline:
                result.append([start,end,max_sub_rms])
                all_max_sub_rms.append(max_sub_rms)
    return result,all_max_sub_rms

def get_topN_peak_by_denoise(rms,threshold,topN,waterline=10):
    result = []
    total = 0
    threshold = threshold.astype(np.float32)
    best_all_peak_trough = []
    beast_all_max_sub_rms = []
    best_threshold = 0
    max_peak_number = 0
    while True:
        print("eporch is {},threshold is {},max_peak_number is {},waterline is {}".format(total,threshold,max_peak_number,waterline))
        rms_copy = rms_smooth(rms, threshold, 6)
        rms_copy = [x if x >= threshold else 0 for x in rms_copy]
        all_peak_trough,all_max_sub_rms = get_peak_trough_by_denoise(rms,rms_copy,threshold,waterline)
        threshold *= 0.95
        if threshold <= waterline:
            #threshold = waterline
            pass
        total += 1
        if max_peak_number < len(all_peak_trough):
            max_peak_number = len(all_peak_trough)
            best_all_peak_trough = all_peak_trough
            beast_all_max_sub_rms = all_max_sub_rms
            best_threshold = threshold
        if len(all_peak_trough) >= topN or total > 50:
            break
    if len(all_peak_trough) >= topN:
        topN = len(all_peak_trough)
    if total > 50:
        all_peak_trough = best_all_peak_trough
        all_max_sub_rms = beast_all_max_sub_rms
    topN_indexs = find_n_largest(all_max_sub_rms,topN)
    for i in range(0,len(topN_indexs)):
        index = topN_indexs[i]
        start,end,rms = all_peak_trough[index]
        if start == 0:
            start = 1
        result.append(start)
    return result,rms_copy,best_threshold

def find_min_waterline(rms,step):
    threshold = 0.01
    result = []
    total = 0
    while True:
        for i in range(len(rms)-step):
            if np.max(rms[i:i+step]) <threshold:
                result.append([i,threshold])
                return result
        threshold *= 1.2
        #threshold = threshold.astype(np.float32)
        total += 1
        if total > 30:
            break
    return result
def find_best_waterline(raw_rms,step,topN):
    threshold = 0.01
    min_waterline = find_min_waterline(raw_rms,step)
    if len(min_waterline) > 0:
        threshold = min_waterline[0][1]
    result = threshold
    total = 0
    max_starts_total = 0
    best_starts = []
    while True:
        rms = [x if x >= threshold else 0 for x in raw_rms]
        starts = [i for i in range(0, len(rms) - 1) if rms[i] == 0 and rms[i + 1] > threshold]
        if max_starts_total < len(starts):
            #pass
            result = threshold
            max_starts_total = len(starts)
            best_starts = starts
        if topN - len(starts) <1:
            result = threshold
            return result,best_starts
        threshold *= 1.05
        #threshold = threshold.astype(np.float32)
        total += 1
        if total > 30:
            break
    return result,best_starts

def rms_smooth(rms,threshold,step):

    for i in range(step,len(rms)-1):
        if rms[i] < threshold and rms[i+1]>=threshold:
            start = i - step
            end = i + 1
            if np.min(rms[start:end]) > threshold*0.65:
                for j in range(start, end):
                    if rms[j] < threshold:
                        threshold = threshold.astype(np.float32)
                        rms[j] = threshold
    return rms

def get_onsets_frames_for_jz(filename):
    y, sr = load_and_trim(filename)
    CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref=np.max)
    w, h = CQT.shape
    CQT[50:w, :] = -100
    CQT[0:20, :] = -100

    # 标准节拍时间点
    #type_index = get_onsets_index_by_filename(filename,code)
    total_frames_number = get_total_frames_number(filename)
    # base_frames = onsets_base_frames_rhythm(type_index,total_frames_number)
    base_frames = onsets_base_frames(code, total_frames_number)
    base_onsets = librosa.frames_to_time(base_frames, sr=sr)

    first_frame = base_frames[1] - base_frames[0]
    rms = librosa.feature.rmse(y=y)[0]
    rms = [x / np.std(rms) for x in rms]
    min_waterline = find_min_waterline(rms, 8)
    first_frame_rms = rms[0:first_frame]
    first_frame_rms_max = np.max(first_frame_rms)
    waterline = 0
    if len(min_waterline) > 0:
        waterline = min_waterline[0][1]

    if first_frame_rms_max == np.max(rms):
        #print("=====================================")
        threshold = first_frame_rms_max * 0.35
        if threshold > waterline:
            threshold = waterline + 0.2
            threshold = np.float64(threshold)
        rms = rms_smooth(rms, threshold, 6)
        # rms = [x if x > first_frame_rms_max * 0.35 else 0 for x in rms]
    else:
        threshold = first_frame_rms_max * 0.6
        rms = rms_smooth(rms, threshold, 6)
        # rms = [x if x > first_frame_rms_max * 0.6 else 0 for x in rms]
    # rms = [x / np.std(rms) if x / np.std(rms) > first_frame_rms_max*0.8 else 0 for x in rms]
    # rms = rms/ np.std(rms)
    rms_diff = np.diff(rms)
    # print("rms_diff is {}".format(rms_diff))
    #print("rms max is {}".format(np.max(rms)))
    # all_peak_points = get_all_onsets_starts(rms,0.7)
    # all_peak_points = get_onsets_by_cqt_rms(y,16000,base_frames,0.7)
    topN = len(base_frames)
    best_starts_waterline = []
    threshold = first_frame_rms_max * 0.8
    if len(min_waterline) > 0:
        # waterline = min_waterline[0][1]
        # waterline *= 1.5
        waterline, best_starts_waterline = find_best_waterline(rms, 4, topN)
        waterline += 0.3
        if waterline < 0.6:
            waterline = 0.6

        # waterline = 0.8
        if threshold < waterline:
            # waterline +=0.0000000001
            threshold = waterline + 0.5
            threshold = np.float64(threshold)
            # pass
        #print("waterline is {}".format(waterline))
        #print("threshold is {}".format(threshold))
    all_peak_points, rms, threshold = get_topN_peak_by_denoise(rms, threshold, topN, waterline)
    # all_peak_points,_ = get_topN_peak_by_denoise(rms, first_frame_rms_max * 0.8, topN)
    # onsets_frames = get_real_onsets_frames_rhythm(y)
    # _, onsets_frames = get_onset_rmse_viterbi(y, 0.35)
    # onsets_frames = get_all_onsets_starts_for_beat(rms, 0.6)
    onsets_frames = []

    # all_peak_points = get_all_onsets_starts_for_beat(rms,0.6)
    # all_trough_points = get_all_onsets_ends(rms,-0.4)
    want_all_points = np.hstack((all_peak_points, onsets_frames))
    want_all_points = list(set(want_all_points))
    want_all_points.sort()
    if topN - len(want_all_points) >= 3 and topN - len(best_starts_waterline) < 3:
        want_all_points = best_starts_waterline
    want_all_points_diff = np.diff(want_all_points)
    if len(want_all_points) > 0:
        # 去掉挤在一起的线
        result = [want_all_points[0]]
        for i, v in enumerate(want_all_points_diff):
            if v > 4:
                result.append(want_all_points[i + 1])
            else:
                pass
        onsets_frames = result
    return onsets_frames,rms

def find_n_largest(a,topN):
    import heapq

    a = list(a)
    #a = [43, 5, 65, 4, 5, 8, 87]

    re1 = heapq.nlargest(topN, a)  # 求最大的三个元素，并排序
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

def get_real_onsets_frames_rhythm(y,modify_by_energy=False,gap = 0.1):
    y_max = max(y)
    # y = np.array([x if x > y_max*0.01 else y_max*0.01 for x in y])
    # 获取每个帧的能量
    energy = librosa.feature.rmse(y)
    #print(np.mean(energy))
    energy_diff = np.diff(energy)
    #print(energy_diff)
    onsets_frames = librosa.onset.onset_detect(y)

    #print(onsets_frames)
    #print(np.diff(onsets_frames))

    some_y = [energy[0][x] for x in onsets_frames]
    #print("some_y is {}".format(some_y)) # 节拍点对应帧的能量
    energy_mean = (np.sum(some_y) - np.max(some_y))/(len(some_y)-1)  # 获取能量均值
    #print("energy_mean for some_y is {}".format(energy_mean))
    energy_gap = energy_mean * gap
    # #energy_gap = np.max(energy[0][0:20])*0.8
    # some_energy_diff = [energy_diff[0][x] if x < len(energy_diff) else energy_diff[0][x-1]  for x in onsets_frames]
    # energy_diff_mean = np.mean(some_energy_diff)
    # print("some_energy_diff is {}".format(some_energy_diff))
    # print("energy_diff_meanis {}".format(energy_diff_mean))
    if modify_by_energy:
        onsets_frames = [x for x in range(len(energy[0])) if energy[0][x] > energy_gap*8]  # 直接对能量进行筛选，筛选掉能量过低的伪节拍点
    else:
        onsets_frames = [x for x in onsets_frames if energy[0][x] > energy_gap]  # 筛选掉能量过低的伪节拍点

    # 筛选过密的节拍点
    onsets_frames_new = []
    for i in range(0, len(onsets_frames)):
        if i == 0:
            onsets_frames_new.append(onsets_frames[i])
            last_frame = onsets_frames[i]
            continue
        if onsets_frames[i] - last_frame <= 2:
            # middle = int((onsets_frames[i] + last_frame) / 2)
            # # middle = onsets_frames[i]
            # onsets_frames_new.pop()
            # onsets_frames_new.append(middle)
            # last_frame = middle
            continue
        else:
            onsets_frames_new.append(onsets_frames[i])
            last_frame = onsets_frames[i]
    onsets_frames = onsets_frames_new
    return onsets_frames

def get_real_onsets_for_note(y,modify_by_energy=False,gap = 0.1):
    y_max = max(y)
    # y = np.array([x if x > y_max*0.01 else y_max*0.01 for x in y])
    # 获取每个帧的能量
    energy = librosa.feature.rmse(y)
    #print(np.mean(energy))
    energy_diff = np.diff(energy)
    #print(energy_diff)
    onsets_frames = librosa.onset.onset_detect(y)

    #print(onsets_frames)
    #print(np.diff(onsets_frames))

    some_y = [energy[0][x] for x in onsets_frames]
    #print("some_y is {}".format(some_y)) # 节拍点对应帧的能量
    energy_mean = (np.sum(some_y) - np.max(some_y))/(len(some_y)-1)  # 获取能量均值
    #print("energy_mean for some_y is {}".format(energy_mean))
    energy_gap = energy_mean * gap
    # #energy_gap = np.max(energy[0][0:20])*0.8
    # some_energy_diff = [energy_diff[0][x] if x < len(energy_diff) else energy_diff[0][x-1]  for x in onsets_frames]
    # energy_diff_mean = np.mean(some_energy_diff)
    # print("some_energy_diff is {}".format(some_energy_diff))
    # print("energy_diff_meanis {}".format(energy_diff_mean))
    if modify_by_energy:
        onsets_frames = [x for x in range(len(energy[0])) if energy[0][x] > energy_gap*8]  # 直接对能量进行筛选，筛选掉能量过低的伪节拍点
    else:
        onsets_frames = [x for x in onsets_frames if energy[0][x] > energy_gap]  # 筛选掉能量过低的伪节拍点

    # 筛选过密的节拍点
    onsets_frames_new = []
    for i in range(0, len(onsets_frames)):
        if i == 0:
            onsets_frames_new.append(onsets_frames[i])
            last_frame = onsets_frames[i]
            continue
        if onsets_frames[i] - last_frame <= 4:
            # middle = int((onsets_frames[i] + last_frame) / 2)
            # # middle = onsets_frames[i]
            # onsets_frames_new.pop()
            # onsets_frames_new.append(middle)
            # last_frame = middle
            continue
        else:
            onsets_frames_new.append(onsets_frames[i])
            last_frame = onsets_frames[i]
    onsets_frames = onsets_frames_new
    return onsets_frames

'''
删除伪节拍点（去掉识别结果中是节拍结果的点）
'''
def del_note_end_by_cqt(y,onset_frames):
    CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref=np.max)
    w, h = CQT.shape
    CQT[50:w, :] = np.min(CQT)
    CQT[0:20, :] = np.min(CQT)
    cqt_max = np.max(CQT)
    for i in range(w):
        for j in range(h):
            if CQT[i][j] > -20:
                CQT[i][j] = np.max(CQT)
    result = []
    for x in onset_frames:
        if h - x <= 4:
            continue
        # 节拍点后面2个位置都没有节拍亮点,即后面是断开的
        elif np.max(CQT[:,x+1]) != cqt_max or np.max(CQT[:, x + 2]) != cqt_max or np.max(CQT[:, x + 3]) != cqt_max or np.max(CQT[:, x + 4]) != cqt_max:
            continue
        else:
            result.append(x)
    return result

'''
删除伪节拍点（去掉识别结果中是节拍中部的点）
'''
def del_note_middle_by_cqt(y,onset_frames):
    CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref=np.max)
    w, h = CQT.shape
    CQT[50:w, :] = np.min(CQT)
    CQT[0:20, :] = np.min(CQT)
    cqt_max = np.max(CQT)
    for i in range(w):
        for j in range(h):
            if CQT[i][j] > -20:
                CQT[i][j] = np.max(CQT)
    result = []
    if len(onset_frames) > 0:
        result.append(onset_frames[0])
    for i in range(1,len(onset_frames)-1):
        cqt_before = CQT[:,onset_frames[i-1]:onset_frames[i]]
        cqt_after = CQT[:,onset_frames[i]:onset_frames[i+1]]
        note_height_before = get_note_height(cqt_before,cqt_max)
        note_height_after = get_note_height(cqt_after, cqt_max)
        if note_height_after == 0 or note_height_before == 0:
            continue
        elif np.abs(note_height_before - note_height_after) >= 1:
            result.append(onset_frames[i])
        elif np.max(CQT[:,onset_frames[i]-1]) != cqt_max or np.max(CQT[:, onset_frames[i] - 2]) != cqt_max or np.max(CQT[:, onset_frames[i] - 3]) != cqt_max:
            result.append(onset_frames[i])
    return result

'''
获取节拍起始点
'''
def get_note_start_by_cqt(y,onset_frames):
    CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref=np.max)
    w, h = CQT.shape
    CQT[50:w, :] = np.min(CQT)
    CQT[0:20, :] = np.min(CQT)
    cqt_max = np.max(CQT)
    for i in range(w):
        for j in range(h):
            if CQT[i][j] > -20:
                CQT[i][j] = np.max(CQT)
    result = []
    last = 0
    for x in onset_frames:
        if h - x <= 4 or x <3:
            continue
        # 节拍点后面2个位置都有节拍亮点,但前面2个位置都没有节拍亮点
        for j in range(w):
            # 水平方向前面不亮，垂直方面至少5个连续亮
            if CQT[j, x] == cqt_max and CQT[j, x - 1] != cqt_max \
                    and CQT[j+1,x] == cqt_max and CQT[j+2,x] == cqt_max and CQT[j+3,x] == cqt_max:
                result.append(x)
                last = x
            else:
                continue
    return result

def get_note_height(cqt,cqt_max):
    result = []
    note_height = 0
    w,h = cqt.shape
    for i in range(h):
        sub = cqt[:,i]
        for j in range(w):
            if sub[j] == cqt_max and sub[j+1] !=cqt_max and sub[j-1] == cqt_max:
                result.append(j)
    if len(result) > 0:
        note_height = max(set(result), key=result.count)
    return note_height

def del_overcrowding(onset_frames,step):
    result = [onset_frames[0]]
    for i in range(1,len(onset_frames)):
        if onset_frames[i] - onset_frames[i-1] > step:
            result.append(onset_frames[i])
    return result

def del_overcrowding_v2(onset_frames,step):
    result = [onset_frames[0]]
    last = onset_frames[0]
    for i in range(1,len(onset_frames)):
        if onset_frames[i] - last > step:
            result.append(onset_frames[i])
            last = onset_frames[i]
    return result

def cqt_split(filename,savepath,step_width,onsets_frames=[]):
    start_time = time.time()
    y, sr = librosa.load(filename)

    CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref=np.max)
    w, h = CQT.shape
    #CQT[50:w, :] = np.min(CQT)
    CQT[0:20, :] = np.min(CQT)
    for i in range(w):
        for j in range(h):
            if CQT[i][j] > -20:
                CQT[i][j] = np.max(CQT)
            # else:
            #     CQT[i][j] = np.min(CQT)
    cqt_max = np.max(CQT)

    # 拆分CQT
    step = int(h / 10)
    half = int(step / 2)
    middle = int(h / 2)
    cpu_count = multiprocessing.cpu_count()
    #print("cpu_count is {}".format(cpu_count))
    ps = Pool(cpu_count)
    result = []
    threads = []
    #print("len(onsets_frames) is {}".format(len(onsets_frames)))
    if len(onsets_frames)>0:
        print("tpye1 主进程开始执行>>> pid={}".format(os.getpid()))
        for i in onsets_frames:
            #print("len(onsets_frames) is {}".format(len(onsets_frames)))
            y2 = np.zeros(CQT.shape)
            if i >= h - step:
                break
            offset = middle - (i + half)
            for j in range(step):
                y2[:, i + j + offset] = CQT[:, i + j]
            ps.apply_async(savePltWorker, args=(y2, middle, sr, savepath, i,))  # 异步执行
            result.append(i)
            #if np.max(y2) == np.max(CQT):
    else:
        #step_width = int(h/150)
        start, end = get_start_and_end_for_note(y, sr)
        print("tpye2 主进程开始执行>>> pid={}".format(os.getpid()))
        for i in range(start, end, step_width):
        #while i < h:
            y2 = np.zeros(CQT.shape)
            if i >= h - step:
                break
            offset = middle - (i + half)
            for j in range(step):
                y2[:, i + j + offset] = CQT[:, i + j]
            before_y2 = y2[:,middle-4:middle]
            after_y2 = y2[:, middle:middle + 4]
            if np.max(y2) == cqt_max and np.max(after_y2) == cqt_max:
                # librosa.display.specshow(y2, y_axis='cqt_note', x_axis='time')
                # t = librosa.frames_to_time([middle], sr=sr)
                # plt.vlines(t, 0, sr, color='y', linestyle='--')  # 标出节拍位置
                # tmp = os.listdir(savepath)

                #plt.savefig(savepath + str(i) + '.jpg', bbox_inches='tight', pad_inches=0)
                #plt.clf()
                result.append(i+half)
                ps.apply_async(savePltWorker, args=(y2,middle,sr,savepath,i,))  # 异步执行
                #i += step_width*2
                #print("i is {}".format(i))
                #i += step_width
                # t1 = threading.Thread(target=savePltWorker,args=(y2,middle,sr,savepath,i,))
                # threads.append(t1)
                #ps.apply_async(worker, args=(i,savepath,))  # 异步执行
    # for t in threads:
    #     t.setDaemon(True)
    #     t.start()
    # 关闭进程池，停止接受其它进程
    ps.close()
    # 阻塞进程
    ps.join()
    print("主进程终止")
    end_time = time.time()
    print("runinggggggggggggggggggg is {}".format(end_time - start_time))
    print("1 time is {}".format(time.time()))
    return result
def savePltWorker(y2,middle,sr,savepath,i):
    #pass
    #print("middle,sr,savepath,i is : {},{},{},{}".format(middle,sr,savepath,i))
    librosa.display.specshow(y2, y_axis='cqt_note', x_axis='time')
    t = librosa.frames_to_time([middle], sr=sr)
    plt.vlines(t, 0, sr, color='y', linestyle='--')  # 标出节拍位置
    tmp = os.listdir(savepath)

    plt.savefig(savepath + str(i) + '.jpg', bbox_inches='tight', pad_inches=0)
    plt.clf()
def worker(arg,savepath):
    print("子进程开始执行>>> pid={},ppid={},编号{},{}".format(os.getpid(),os.getppid(),arg,savepath))
    time.sleep(0.5)
    print("子进程终止>>> pid={},ppid={},编号{},{}".format(os.getpid(),os.getppid(),arg,savepath))

def get_single_notes(filename,curr_num,savepath,modify_by_energy=False):
    y, sr = librosa.load(filename)
    rms = librosa.feature.rmse(y=y)[0]
    total_frames_number = len(rms)
    #print("time is {}".format(time))
    CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref=np.max)
    librosa.display.specshow(CQT, y_axis='cqt_note', x_axis='time')
    w, h = CQT.shape
    # onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    onset_frames = get_real_onsets_frames_rhythm(y, modify_by_energy=modify_by_energy)

    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    plt.vlines(onset_times, 0, sr, color='y', linestyle='--')
    #print(onset_samples)
    #plt.subplot(len(onset_times),1,1)
    #plt.show()
    plt.clf()

    # 多线程部分
    threads = []

    for i in range(0, len(onset_frames),2):
        # t1 = threading.Thread(target=save_split_notes_for_rhythm, args=(onset_frames,i,total_frames_number,CQT,y, sr,savepath))
        # threads.append(t1)
        save_split_notes_for_rhythm(onset_frames, i, total_frames_number, CQT, y, sr, savepath)
        curr_num += 1


    #plt.show()
    return onset_frames,curr_num

def save_split_notes_for_rhythm(onset_frames,i,total_frames_number,CQT,y, sr,savepath):
    w, h = CQT.shape
    half = 15
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
    plt.savefig(savepath + str(i + 1) + '.jpg', bbox_inches='tight', pad_inches=0)
    plt.clf()

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




def get_total_frames_number(path):
    # audio, sr = librosa.load(path)
    # energy = librosa.feature.rmse(audio)
    # frames = np.nonzero(energy >= np.max(energy) / 5)
    #
    # total = frames[1][-1]
    y, sr = librosa.load(path)
    #y, sr = load_and_trim(path)
    rms = librosa.feature.rmse(y=y)[0]
    #onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    total = len(rms)
    return total

def get_total_frames_number_for_note(path):
    # audio, sr = librosa.load(path)
    # energy = librosa.feature.rmse(audio)
    # frames = np.nonzero(energy >= np.max(energy) / 5)
    #
    # total = frames[1][-1]
    total = 0
    y, sr = librosa.load(path)
    onsets_frames = get_real_onsets_for_note(y, modify_by_energy=True, gap=0.05)
    if len(onsets_frames) > 0 :
        total = onsets_frames[-1] - onsets_frames[0]
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


def list_all_files(rootdir):
    import os
    _files = []
    list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
    for i in range(0,len(list)):
           path = os.path.join(rootdir,list[i])
           if os.path.isdir(path):
              _files.extend(list_all_files(path))
           if os.path.isfile(path):
              _files.append(path)
    return _files

def get_start_and_end_for_note(y, sr):
    silence_threshold = 0.2
    #y, sr = librosa.load(filename)
    times, states = get_viterbi_state(y, silence_threshold)
    rms = librosa.feature.rmse(y=y)[0]
    print("rms len is {}".format(len(rms)))
    rms = [x / np.std(rms) for x in rms]
    mean_rms = np.mean(rms)
    rms = [1 if x > mean_rms* 0.25 else 0 for x in rms]
    start = 1
    end = len(rms) - 1
    if len(rms) > 0:
        for i in range(1,len(rms)-20):
            # b = [states[j:j + 5] for j in range(i, i+20)]
            # c = [np.sum(x) for x in b]
            # min_c = np.min(c)
            if np.min(rms[i:i+5]) == 1 and (states[i] == 1 and np.max(states[i+10:i+20]) == 1):
                start = i
                break

        for i in range(len(rms)-2,30,-1):
            if np.min(rms[i-5:i]) == 1 and (states[i] == 1 and np.max(states[i-30:i-20]) == 1):
                end = i
                break
    return start,end
def draw_baseline_and_note_on_cqt(filename,display=True):
    y, sr = librosa.load(filename)
    start,end = get_start_and_end_for_note(y, sr)
    start_time = librosa.frames_to_time([start,end])
    #plt.axvline(start_time[0],color="r")
    plt.axvline(start_time[1],color="r",linestyle='dashed')
    #print("start,end is {},{}".format(start,end))
    base_frames = onsets_base_frames_for_note(filename)
    base_frames = [x + start - base_frames[0] for x in base_frames]
    print("base_frames is {}".format(base_frames))
    base_time = librosa.frames_to_time(base_frames, sr=sr)
    CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref=np.max)
    w, h = CQT.shape
    # for i in range(w):
    #     for j in range(h):
    #         if CQT[i][j] > -25:
    #             CQT[i][j] = np.max(CQT)
    #         else:
    #             CQT[i][j] = np.min(CQT)
    if display is False:
        CQT = np.where(CQT>-30, np.max(CQT), np.min(CQT))
        CQT[:,:start] = np.min(CQT)
        CQT[:, end:] = np.min(CQT)
        #CQT = find_note_line_for_cqt(CQT, start, end)
        #CQT = np.ones(CQT.shape) * np.min(CQT)
    CQT[0:10, :] = np.min(CQT)
    base_notes = base_note(filename)
    base_notes = [x + 2 - np.min(base_notes) if x != -1 else x for x in base_notes ]
    #print("base_notes is {}".format(base_notes))
    #CQT[base_notes[0], :] = -20
    CQT = add_base_note_to_cqt(CQT, base_notes, base_frames,end,filename)
    #plt.axhline(base_frames[0],color="w")
    librosa.display.specshow(CQT, y_axis='cqt_note', x_axis='time')
    plt.vlines(base_time, 0, sr, color='r', linestyle='dashed')
    plt.axis('off')
    plt.axes().get_xaxis().set_visible(False)
    plt.axes().get_yaxis().set_visible(False)
    return plt

def draw_on_cqt(filename,pic_path,display=True):
    #start_time = time.time()
    y, sr = librosa.load(filename)
    start,end = get_start_and_end_for_note(y, sr)
    #end_time = time.time()
    CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref=np.max)
    w, h = CQT.shape
    print("w,h is {},{}".format(w,h))
    # for i in range(w):
    #     for j in range(h):
    #         if CQT[i][j] > -25:
    #             CQT[i][j] = np.max(CQT)
    #         else:
    #             CQT[i][j] = np.min(CQT)
    if display is False:
        CQT = np.where(CQT>-30, np.max(CQT), np.min(CQT))
        CQT[:,:start] = np.min(CQT)
        CQT[:, end:] = np.min(CQT)
        #CQT = find_note_line_for_cqt(CQT, start, end)
        #CQT = np.ones(CQT.shape) * np.min(CQT)
    CQT[0:10, :] = np.min(CQT)

    #plt.axhline(base_frames[0],color="w")
    librosa.display.specshow(CQT, y_axis='cqt_note', x_axis='time')
    plt.axis('off')
    plt.axes().get_xaxis().set_visible(False)
    plt.axes().get_yaxis().set_visible(False)
    plt.savefig(pic_path, bbox_inches='tight', pad_inches=0)
    #print("rrrrr is {}".format(end_time - start_time))
    return plt,y,sr

def find_note_line_for_cqt(CQT,start,end):
    w,h = CQT.shape
    cqt_max = np.max(CQT)
    cqt_min = np.min(CQT)
    result = np.ones(CQT.shape) * np.min(CQT)

    for i in range(start,end):
        for j in range(4,w):
            if np.min(CQT[j-2:j,i:i+2]) == cqt_max:
                result[j,i] = cqt_max
                #break
    return result


if __name__ == '__main__':
    start_point = 0.2
    times = 6.45
    #code = '[0500,0500;1000;1500;0500;0250,0250,0250,0250;1000]'
    code = '[500,500,1000;500,500,1000;500,500,750,250;2000]'
    #code = '[2000;1000,1000;500,500,1000;2000]'
    #code = '[1000,1000;500,500,1000;1000,1000;2000]'
    #code = '[1000,--(1000);1000,--(1000);500,250,250,1000;--(1000),1000]'
    #code = '[500;1000,500,1000,500;500,500,500,250,250,500,500;250,250,500,500,1000]'
    #code = '[1000,--(1000);1000,--(1000);1000,-(500),500;1000,1000]'
    #code = '[750,250,500,500,500,-(500);500,1000,500,500,-(500);750,250,500,500,500,-(500)]'
    ds = onsets_base(code,times,start_point)


    plt.vlines(ds[:-1], 0, 2400, color='black', linestyle='dashed')
    plt.vlines(ds[-1], 0, 2400, color='white', linestyle='dashed')
    #plt.vlines(time, 0, 2400, color='white', linestyle='dashed')

    pitch_code = '[3,3,3,3,3,3,3,5,1,2,3]'
    chroma_pitch = get_chroma_pitch(pitch_code)
    print(chroma_pitch)
    plt.show()
    step_width = 2
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋律八（2）（60）.wav'
    image_dir = './single_notes/data/test/'
    start_time = time.time()
    y, sr = librosa.load(filename)
    onsets_frames = get_real_onsets_frames_rhythm(y, modify_by_energy=True, gap=0.1)
    onset_frames = cqt_split(filename, image_dir,step_width)
    #onset_frames = cqt_split(filename, image_dir, step_width, onsets_frames)
    #onset_frames = cqt_split(filename, image_dir, step_width)
    end_time = time.time()
    print("run time is {}".format(end_time - start_time))