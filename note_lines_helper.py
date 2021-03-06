# -*- coding: UTF-8 -*-
import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
from create_base import *
from myDtw import *
from find_mismatch import *
# 1. Get the file path to the included audio example
# Sonify detected beat events
# 定义加载语音文件并去掉两端静音的函数
def load_and_trim(path):
    audio, sr = librosa.load(path)
    energy = librosa.feature.rmse(audio)
    frames = np.nonzero(energy >= np.max(energy) / 5)
    indices = librosa.core.frames_to_samples(frames)[1]
    audio = audio[indices[0]:indices[-1]] if indices.size else audio[0:0]

    return audio, sr

def get_note_line_start_v2(cqt,rhythm_code):
    rhythm_code = rhythm_code.replace(";", ',')
    rhythm_code = rhythm_code.replace("[", '')
    rhythm_code = rhythm_code.replace("]", '')
    rhythm_code = [x for x in rhythm_code.split(',')]
    small_code_indexs = [i for i in range(len(rhythm_code)) if rhythm_code[i] == '250']
    threshold_length = 4
    threshold_length_before = 4
    threshold_length_midle = 4
    threshold_length_after = 4
    if len(small_code_indexs) < 1:
        threshold_length = 10
    else:
        before_half = [i for i in range(len(small_code_indexs)) if small_code_indexs[i] <= int(len(rhythm_code)/3) ]
        middle_half = [i for i in range(len(small_code_indexs)) if small_code_indexs[i] > int(len(rhythm_code)/3) and  small_code_indexs[i] <= int(len(rhythm_code)*2/3)]
        after_half = [i for i in range(len(small_code_indexs)) if small_code_indexs[i] > int(len(rhythm_code)*2/3)]
        if len(before_half) < 1:
            threshold_length_before = 10
        if len(middle_half) < 1:
            threshold_length_midle = 10
        if len(after_half) < 1:
            threshold_length_after = 10
    min_cqt = np.min(cqt)
    max_cqt = np.max(cqt)
    w,h = cqt.shape
    result = []
    end_result = []
    note_lines = []
    times = []
    best_longest, best_begin, best_row = 0, 0, 0
    end = 0
    best_col = 10
    for col in range(10,45):
        if np.max(cqt[col,:]) == min_cqt:
            best_col = col
            break
    #print("best_col is {}".format(best_col))
    for col in range(best_col,h-10):
        col_cqt = cqt[10:, col] # 列向量
        # 存在亮点
        if np.max(col_cqt) == max_cqt and col >= end:
            flag = True
            offset = 5
            while flag:
                cols_cqt = cqt[10:, col:col+offset] #柜向量
                best_longest, best_begin, best_row = get_longest_for_cols_cqt(cols_cqt,min_cqt)
                if best_begin + best_longest == offset:
                    offset += 1
                else:
                    flag = False
                    if col <= int(h/3):
                        threshold_length = threshold_length_before
                    elif col > int(h/3) and col <= int(h*2/3):
                        threshold_length = threshold_length_midle
                    else:
                        threshold_length = threshold_length_after

                    if best_longest > threshold_length :
                        result.append(col + best_begin)
                        end = col + best_begin + best_longest
                        end_result.append(end)
                        note_lines.append(best_row)
                        times.append(best_longest)
    return result,end_result,times,note_lines

def merge_note_line(result,end_result,times,note_lines):
    select_result = []
    select_end_result = []
    select_times = []
    select_note_lines = []
    i = 0
    while i <= len(result)-2:
        if result[i] + times[i] == result[i+1] and (note_lines[i] == note_lines[i+1] or np.abs(note_lines[i] - note_lines[i+1]) >=10 or np.abs(note_lines[i] - note_lines[i+1]) <=2):
            select_result.append(result[i])
            select_end_result.append(end_result[i+1])
            select_times.append(times[i] + times[i+1])
            select_note_lines.append(note_lines[i])
            i += 2
        else:
            select_result.append(result[i])
            select_end_result.append(end_result[i])
            select_times.append(times[i])
            select_note_lines.append(note_lines[i])
            i += 1
    return select_result,select_end_result,select_times,select_note_lines

def merge_note_line_by_distance(result,times,note_lines,rhythm_code,total_frames):
    index = 0
    rhythm_code = rhythm_code.replace(";", ',')
    rhythm_code = rhythm_code.replace("[", '')
    rhythm_code = rhythm_code.replace("]", '')
    if rhythm_code.find("(") >= 0:
        tmp = [x for x in rhythm_code.split(',')]
        for i in range(len(tmp)):
            if tmp[i].find("(") >= 0:
                index = i
                break
        rhythm_code = rhythm_code.replace("(", '')
        rhythm_code = rhythm_code.replace(")", '')
        rhythm_code = rhythm_code.replace("-", '')
        rhythm_code = rhythm_code.replace("--", '')
    rhythm_code = [x for x in rhythm_code.split(',')]
    rhythm_code = [int(x) for x in rhythm_code]
    if index > 0:
        rhythm_code[index - 1] += rhythm_code[index]
        del rhythm_code[index]

    sum_code = np.sum(rhythm_code)
    most_small_code = np.min(rhythm_code)
    most_small_gap = int(total_frames*most_small_code/sum_code)
    small_code_indexs = [i for i in range(len(rhythm_code)) if rhythm_code[i] == most_small_code]
    less_half = [i for i in range(len(small_code_indexs)) if small_code_indexs[i] < int(len(rhythm_code) / 2)]
    more_half = [i for i in range(len(small_code_indexs)) if small_code_indexs[i] > int(len(rhythm_code) / 2)]
    threshold = int(most_small_gap*0.5)
    before_half_threshold = int(2*most_small_gap*0.5)
    after_half_threshold = int(2 * most_small_gap * 0.5)
    times_threshold = threshold
    time_median = np.median(times)
    if len(less_half) > 0:
        before_half_threshold = threshold

    if len(more_half) > 0:
        after_half_threshold = threshold

    select_result = []
    select_times = []
    select_note_lines = []
    select_result.append(result[0])
    select_times.append(times[0])
    select_note_lines.append(note_lines[0])

    for i in range(1,len(result)):
        # 前半段
        if result[i] <= result[0] + 0.5* total_frames:
            threshold = before_half_threshold
        else:
            threshold = after_half_threshold
        if result[i] - select_result[-1] > threshold and (times[i] > times_threshold and times[i] > time_median/4 or np.abs(note_lines[i] - note_lines[i-1]) < 2):
            select_result.append(result[i])
            select_times.append(times[i])
            select_note_lines.append(note_lines[i])
    return select_result,select_times,select_note_lines


def del_false_same(base_notes, onset_frames, notes_lines, longest_numbers,keyMap):
    select_onset_frames = [onset_frames[0]]
    select_notes_lines = [notes_lines[0]]
    select_longest_numbers = [longest_numbers[0]]
    same_index = get_the_index_of_same(base_notes)
    if len(same_index) == 0:
        for i in range(1, len(notes_lines)):
            if notes_lines[i] != notes_lines[i - 1]:
                select_onset_frames.append(onset_frames[i])
                select_notes_lines.append(notes_lines[i])
                select_longest_numbers.append(longest_numbers[i])
            else:
                if longest_numbers[i-1] < onset_frames[i] - onset_frames[i-1]:
                    select_onset_frames.append(onset_frames[i])
                    select_notes_lines.append(notes_lines[i])
                    select_longest_numbers.append(longest_numbers[i])
                if longest_numbers[i - 1] == onset_frames[i] - onset_frames[i - 1]:
                    if not keyMap.get(onset_frames[i]) is None and keyMap.get(onset_frames[i]) > 1.75:
                        select_onset_frames.append(onset_frames[i])
                        select_notes_lines.append(notes_lines[i])
                        select_longest_numbers.append(longest_numbers[i])
                    else:
                        select_longest_numbers[-1] = select_longest_numbers[-1] + longest_numbers[i]
    else:
        for i in range(1, len(notes_lines)):
            if notes_lines[i] != notes_lines[i - 1]: #不同音高
                select_onset_frames.append(onset_frames[i])
                select_notes_lines.append(notes_lines[i])
                select_longest_numbers.append(longest_numbers[i])
            else:
                if longest_numbers[i-1] < onset_frames[i] - onset_frames[i-1]: #相同音高，但不相连
                    select_onset_frames.append(onset_frames[i])
                    select_notes_lines.append(notes_lines[i])
                    select_longest_numbers.append(longest_numbers[i])
                if longest_numbers[i - 1] == onset_frames[i] - onset_frames[i - 1]:#相同音高，且相连
                    offset = [np.abs(i - x) for x in same_index]
                    if np.min(offset) <=3:
                        select_onset_frames.append(onset_frames[i])
                        select_notes_lines.append(notes_lines[i])
                        select_longest_numbers.append(longest_numbers[i])
        #return onset_frames, notes_lines, longest_numbers

    return select_onset_frames, select_notes_lines,select_longest_numbers

def get_the_index_of_same(base_notes):
    same_index = []
    for i in range(1, len(base_notes)):
        if base_notes[i] == base_notes[i - 1]:
            same_index.append(i)
    return same_index

def get_longest_for_cols_cqt(cols_cqt,min_cqt):
    w,h = cols_cqt.shape
    best_longest, best_begin, best_row = 0, 0, 0
    for row in range(10, w - 10):
        row_cqt = cols_cqt[row]
        row_cqt = [1 if row_cqt[i] > min_cqt else 0 for i in range(len(row_cqt))]
        longest, begin = getLongestLine(row_cqt)
        if longest > best_longest:
            best_longest = longest
            best_begin = begin
            best_row = row

    return best_longest, best_begin, best_row

def get_note_line_start(cqt):
    result = []
    end_result = []
    w,h = cqt.shape
    cqt_max = np.max(cqt)
    cqt_min = np.min(cqt)
    end_position = 0
    for i in range(5,h-15):
        # 该列存在亮点
        if np.max(cqt[:,i]) == cqt_max and i > end_position:
            sub_cqt = cqt[:,i:]
            # 从上往下逐行判断
            for j in range(w-20,10,-1):
                row_cqt = sub_cqt[j]
                #如果该行存在连续8个亮点
                if np.min(row_cqt[0:8]) == cqt_max:
                    #判断上下区域是否有音高线
                    hight_cqt = cqt[j+10:j+20,i:i+10]
                    low_cqt = np.zeros(hight_cqt.shape)
                    if j - 20 > 0:
                        low_cqt = cqt[j-20:j-10,i:i+10]
                    check_nioce_cqt_start = j - 6 if j - 6 > 0 else 0
                    check_nioce_cqt_end = j+6 if j + 6 < w else w
                    check_nioce_cqt = cqt[check_nioce_cqt_start:check_nioce_cqt_end,i:i+10]
                    check_nioce_low_result = False
                    check_nioce_high_result = False
                    for n in range(0,int((check_nioce_cqt_end-check_nioce_cqt_start-1)/2)):
                        if np.max(check_nioce_cqt[n]) == cqt_min:
                            check_nioce_low_result = True
                            break
                    for n in range(check_nioce_cqt_end - check_nioce_cqt_start-1,int((check_nioce_cqt_end - check_nioce_cqt_start - 1) / 2),-1):
                        if np.max(check_nioce_cqt[n]) == cqt_min:
                            check_nioce_high_result = True
                            break

                    # 如果上下区域存在音高线，则说明不是噪声，则将起点作为节拍起点，同时找出连续区域的结束点
                    if check_nioce_low_result and check_nioce_high_result and (np.max(hight_cqt) == cqt_max or np.max(low_cqt) == cqt_max):
                        if len(result) == 0:
                            result.append(i)
                            #print("i,j is {}==={}".format(i,j))
                        else:
                            offset = [np.abs(x -i) for x in result]
                            if np.min(offset) > 10:
                                result.append(i)
                                #print("i,j is {}==={}".format(i, j))
                        longest_end_position = 0
                        #找出该连通块最大的长度
                        for k in range(j-10,j):
                            k_cqt = sub_cqt[k]
                            #print("k_cqt shape is {},{}".format(k_cqt.shape[0],np.min(k_cqt)))
                            indexs_min = [i for i in range(len(k_cqt)) if k_cqt[i] == cqt_min]
                            index_min = 0
                            if len(indexs_min)>0:
                                index_min = indexs_min[0]
                            end_position = i + index_min
                            if end_position > longest_end_position:
                                longest_end_position = end_position
                        end_result.append(longest_end_position)
                        check_nioce_high_result = False
                        check_nioce_low_result = False
                        break
    return result,end_result

def get_note_lines(cqt,result):
    note_lines = []
    w,h = cqt.shape
    cqt_max = np.max(cqt)
    cqt_min = np.min(cqt)
    selected_result = []
    longest_numbers = []
    for i in range(len(result)):
        x = result[i]
        if i < len(result)-1:
            next = result[i+1]
        else:
            next = h -10
        sub_cqt = cqt[:,x:next]
        note_line, longest_num = get_longest_note_line(sub_cqt)
        # flag,low_begin, low_note_line, low_longest_num,high_begin,high_note_line,high_longest_num = check_more_note_line(sub_cqt,note_line, longest_num)
        # #判断是否存在可再分的情况
        # if flag is True:
        #     if low_begin<high_begin:
        #         #前一条
        #         note_lines.append(low_note_line)
        #         longest_numbers.append(low_longest_num)
        #         selected_result.append(x)
        #         #后一条
        #         note_lines.append(high_note_line)
        #         longest_numbers.append(high_longest_num)
        #         selected_result.append(x+low_longest_num)
        #     else:
        #         # 前一条
        #         note_lines.append(high_note_line)
        #         longest_numbers.append(high_longest_num)
        #         selected_result.append(x)
        #         # 后一条
        #         note_lines.append(low_note_line)
        #         longest_numbers.append(low_longest_num)
        #         selected_result.append(x + high_longest_num)
        # else:
        #     note_lines.append(note_line)
        #     longest_numbers.append(longest_num)
        #     selected_result.append(x)
        note_lines.append(note_line)
        longest_numbers.append(longest_num)
        selected_result.append(x)
    return selected_result,note_lines,longest_numbers


def get_note_lines_v2(cqt,result):
    note_lines = []
    w,h = cqt.shape
    cqt_max = np.max(cqt)
    cqt_min = np.min(cqt)
    selected_result = []
    longest_numbers = []
    for i in range(len(result)):
        x = result[i]
        if i < len(result)-1:
            next = result[i+1]
        else:
            next = h -10
        sub_cqt = cqt[:,x:next]
        note_line, longest_num = get_longest_note_line_v2(sub_cqt)
        note_lines.append(note_line)
        longest_numbers.append(longest_num)
        selected_result.append(x)
    return selected_result,note_lines,longest_numbers

def check_more_note_line(sub_cqt,note_line, longest_num):
    w, h = sub_cqt.shape
    # print("w,h is {},{}".format(w,h))

    best_longest_num = 0
    best_note_line = 0
    best_begin = 0
    if h > 0:
        min_cqt = np.min(sub_cqt)
        #取音高线下侧
        for row in range(note_line-1, note_line -10,-1):
            row_cqt = sub_cqt[row]
            row_cqt = [1 if row_cqt[i] > min_cqt else 0 for i in range(len(row_cqt))]
            total_continue,begin  = getLongestLine(row_cqt)
            if total_continue > longest_num or (total_continue == longest_num and row<note_line-1):
                best_longest_num = total_continue
                best_note_line = row
                best_begin = begin
        low_note_line = best_note_line
        low_longest_num = best_longest_num
        low_begin = best_begin

        # 取音高线上侧
        best_longest_num = 0
        best_note_line = 0
        best_begin = 0
        for row in range(note_line+1, note_line + 10):
            row_cqt = sub_cqt[row]
            row_cqt = [1 if row_cqt[i] > min_cqt else 0 for i in range(len(row_cqt))]
            total_continue,begin = getLongestLine(row_cqt)
            if total_continue > longest_num or (total_continue == longest_num and row>note_line+1):
                best_longest_num = total_continue
                best_note_line = row
                best_begin = begin
        high_note_line = best_note_line
        high_longest_num = best_longest_num
        high_begin = best_begin

        if (longest_num - low_longest_num - high_longest_num) < 3:
            return True,low_begin,low_note_line, low_longest_num,high_begin,high_note_line,high_longest_num
        else:
            return False, low_begin, low_note_line, low_longest_num,high_begin,high_note_line,high_longest_num

def modify_some_note_line(cqt,onset_frame,low_start):
    note_line = 0
    w,h = cqt.shape
    cqt_max = np.max(cqt)
    cqt_min = np.min(cqt)

    x = onset_frame
    sub_cqt = cqt[:,x:x+15]
    # 从下往上逐行判断
    for j in range(low_start,w-10):
        row_cqt = sub_cqt[j]
        #如果存在连续的亮点，长度大于8
        max_acount = np.sum(row_cqt == cqt_max)
        min_acount = np.sum(row_cqt == cqt_min)
        if max_acount > min_acount and np.sum(sub_cqt[j-1] == cqt_min) > min_acount:
            note_line = j
            #print("x,j is {},{}".format(x,j))
            break
    return note_line

def check_all_note_lines(onset_frames,all_note_lines,cqt):
    note_lines_median = np.median(all_note_lines)
    #print("note_lines_median is {}".format(note_lines_median))
    selected_note_lines = []
    last = note_lines_median
    for i in range(len(all_note_lines)):
        x = all_note_lines[i]
        onset_frame = onset_frames[i]
        if np.abs(x - note_lines_median) > 10:
            low_start = x + 1
            note_line = modify_some_note_line(cqt, onset_frame, low_start)
            if np.abs(note_line - note_lines_median) > 8:
                note_line = int(note_lines_median)
            selected_note_lines.append(note_line)
        elif x - last > 8:
            selected_note_lines.append(x - 12)
        elif last - x > 8:
            selected_note_lines.append(x + 12)
        else:
            selected_note_lines.append(x)
        last = selected_note_lines[-1]
    return selected_note_lines

def del_false_note_lines(onset_frames,all_note_lines,rms,CQT):
    select_note_lines = []
    select_note_lines.append(all_note_lines[0])
    select_onset_frames = []
    select_onset_frames.append(onset_frames[0])
    #print("max rms is {}".format(np.max(rms)))
    for i in range(1,len(onset_frames)):
        current_onset = onset_frames[i]
        last_onset = onset_frames[i-1]
        current_note = all_note_lines[i]
        last_note = all_note_lines[i-1]
        # 如果当前音高线等于前一个音高线
        if current_note == last_note:
            if np.max(CQT[current_note:current_note+4,current_onset-1]) == np.min(CQT):
                select_note_lines.append(all_note_lines[i])
                select_onset_frames.append(onset_frames[i])
                print("np.abs(rms[i+1] - rms[i-1]) is {},{},{}".format(rms[current_onset+1] , rms[current_onset-1],np.abs(rms[current_onset+1] - rms[current_onset-1])))
            elif np.abs(rms[current_onset+1] - rms[current_onset-1]) > 0.08:
                select_note_lines.append(all_note_lines[i])
                select_onset_frames.append(onset_frames[i])
        else:
            select_note_lines.append(all_note_lines[i])
            select_onset_frames.append(onset_frames[i])
    return select_onset_frames,select_note_lines

'''
找漏的
'''
def find_loss_by_rms_mean(result,rms,CQT):
    select_result = result.copy()
    rms_on_onset_frames_cqt = [rms[x] for x in result]
    mean_rms_on_frames = np.mean(rms_on_onset_frames_cqt)
    print("mean_rms_on_frames is {}".format(mean_rms_on_frames))
    rms_diff = np.diff(rms)
    for i in range(5,len(rms)-5):
        off = [np.abs(x -i) for x in select_result]
        min_off = np.min(off)
        start = i - 2
        end = i + 2
        # 条件一：跨过均值线,振幅有增加
        cond1 = rms[start] < mean_rms_on_frames - 0.2 and rms[end] > mean_rms_on_frames + 0.2

        if cond1 and min_off > 10:
            #print("np.std(sub_rms) is {}".format(np.std(sub_rms)))
            if rms[i-3]< rms[i]:
                select_result.append(i-3)
            elif rms[i-2]< rms[i]:
                select_result.append(i - 2)
            elif rms[i-1]< rms[i]:
                select_result.append(i - 1)
            else:
                select_result.append(i)
    select_result.sort()
    rms_on_onset_frames_cqt = [rms[x] for x in select_result]
    mean_rms_on_frames = np.mean(rms_on_onset_frames_cqt)
    print("mean_rms_on_frames is {}".format(mean_rms_on_frames))
    return select_result

def continueUp(rms):
    sum1, res = 0, 0
    for i in range(len(rms)):
        #遇1加1，遇0置0
        sum1 = sum1*i + i
        if sum1 > res:
            #记录连续1的长度
            res = sum1
    return res

def del_end_range(onset_frames,all_note_lines,times,rms,frames_total,base_frames):
    base_frames = [ x - (base_frames[0]-onset_frames[0]) for x in base_frames]
    length = len(rms)
    select_onset_frames = []
    select_all_note_lines = []
    select_times = []
    threshold = 40
    end_frame = onset_frames[0] + frames_total
    if onset_frames[-1] - onset_frames[-2] < 25:
        threshold = 20
    for i in range(len(onset_frames)):
        onset_frame = onset_frames[i]
        note_line = all_note_lines[i]
        t = times[i]
        flag = True
        # if rms[onset_frame -1] < rms[onset_frame] and rms[onset_frame +1]  < rms[onset_frame]:
        #     flag = False
        if onset_frame < end_frame and onset_frame < length - threshold:
            if onset_frame > base_frames[-1] + 20 and note_line == all_note_lines[i-1]:
                continue
            select_onset_frames.append(onset_frame)
            select_all_note_lines.append(note_line)
            select_times.append(t)
    return select_onset_frames,select_all_note_lines,select_times

# 如果音高线以上有没有可分点
def find_other_note_line(onset_frame,note_line,sub_cqt):
    w,h = sub_cqt.shape
    #print("w,h is {},{}".format(w,h))
    min_cqt = np.min(sub_cqt)
    max_cqt = np.max(sub_cqt)
    longest_num = 0
    best_note_numbers = []
    best_other_onset_frames = []

    for row in range(note_line+1,w-10):
        row_cqt = sub_cqt[row]
        if np.max(row_cqt) == max_cqt:
            row_cqt = [1 if row_cqt[i] > min_cqt else 0 for i in range(len(row_cqt))]
            other_onset_frames,note_numbers = get_all_small_note_line(onset_frame,row_cqt)
            lenght_other_onset_frmaes = len(other_onset_frames)
            if lenght_other_onset_frmaes > longest_num:
                longest_num = lenght_other_onset_frmaes
                best_note_numbers = note_numbers
                best_other_onset_frames = other_onset_frames
    return best_other_onset_frames,best_note_numbers

def get_all_small_note_line(start,nums):
    sum1, res = 0, 0
    index = 0
    onset_frames = []
    note_numbers = []
    start_index = 0
    last_sum1 = 0
    for i in nums:
        #遇1加1，遇0置0
        sum1 = sum1*i + i

        if sum1 == 1: # 记录开始点
            start_index = index
        index +=1

        if sum1 == 0 and last_sum1 > 10:
            onset_frames.append(start + start_index)
            note_numbers.append(last_sum1)
        last_sum1 = sum1

    return onset_frames,note_numbers

def get_longest_note_line(sub_cqt):
    w,h = sub_cqt.shape
    #print("w,h is {},{}".format(w,h))

    longest_num = 0
    note_line = 0
    if h > 0:
        min_cqt = np.min(sub_cqt)
        for row in range(10,w-10):
            row_cqt = sub_cqt[row]
            row_cqt = [1 if row_cqt[i] > min_cqt else 0 for i in range(len(row_cqt))]
            total_continue = continueOne(row_cqt)
            if total_continue > 0.8 * h:
                return row, total_continue
            else:
                if total_continue > longest_num:
                    longest_num = total_continue
                    note_line = row
    return note_line,longest_num

def get_longest_note_line_v2(sub_cqt):
    w,h = sub_cqt.shape
    #print("w,h is {},{}".format(w,h))

    longest_num = 0
    note_line = 0
    if h > 0:
        min_cqt = np.min(sub_cqt)
        for row in range(10,w-10):
            row_cqt = sub_cqt[row]
            row_cqt = [1 if row_cqt[i] > min_cqt else 0 for i in range(len(row_cqt))]
            total_continue = continueOne(row_cqt)
            if total_continue > 0.3 * h:
                return row, total_continue
            else:
                if total_continue > longest_num:
                    longest_num = total_continue
                    note_line = row
    return note_line,longest_num

def get_longest_note_line(sub_cqt):
    w,h = sub_cqt.shape
    #print("w,h is {},{}".format(w,h))
    rows = []
    longest_num = 0
    note_line = 0
    if h > 0:
        min_cqt = np.min(sub_cqt)
        for row in range(10,w-10):
            row_cqt = sub_cqt[row]
            row_cqt = [1 if row_cqt[i] > min_cqt else 0 for i in range(len(row_cqt))]
            total_continue = continueOne(row_cqt)
            if total_continue > 0.8 * h:
                return row, total_continue
            else:
                if total_continue > longest_num:
                    longest_num = total_continue
                    note_line = row
    return note_line,longest_num

def continueOne(nums):
    sum1, res = 0, 0
    for i in nums:
        #遇1加1，遇0置0
        sum1 = sum1*i + i
        if sum1 > res:
            #记录连续1的长度
            res = sum1
    return res

def getLongestLine(nums):
    sum1, res,begin,index = 0, 0,0,1
    for i in nums:
        #遇1加1，遇0置0
        sum1 = sum1*i + i
        if sum1 > res:
            #记录连续1的长度
            res = sum1
            begin = index - sum1
        index += 1
    return res,begin



def get_loss_at_begin(cqt,result,all_note_lines,longest_numbers):
    min_cqt = np.min(cqt)
    w,h = cqt.shape
    first = result[0]
    best_longest, best_begin,best_row = 0,0,0
    if first > 60:
        f_cqt = cqt[:,:first]
        for row in range(10, w - 10):
            row_cqt = f_cqt[row]
            row_cqt = [1 if row_cqt[i] > min_cqt else 0 for i in range(len(row_cqt))]
            longest,begin = getLongestLine(row_cqt)
            if longest > best_longest:
                best_longest = longest
                best_begin = begin
                best_row = row

        if best_longest > 30:
            result.insert(0,best_begin)
            all_note_lines.insert(0,best_row)
            longest_numbers.insert(0, best_longest)
    return result,all_note_lines,longest_numbers

'''
求两个字符串的最长公共子串
思想：建立一个二维数组，保存连续位相同与否的状态
'''


def getNumofCommonSubstr(str1, str2):
    lstr1 = len(str1)
    lstr2 = len(str2)
    record = [[0 for i in range(lstr2 + 1)] for j in range(lstr1 + 1)]  # 多一位
    maxNum = 0  # 最长匹配长度
    p = 0  # 匹配的起始位

    for i in range(lstr1):
        for j in range(lstr2):
            if str1[i] == str2[j]:
                # 相同则累加
                record[i + 1][j + 1] = record[i][j] + 1
                if record[i + 1][j + 1] > maxNum:
                    # 获取最大匹配长度
                    maxNum = record[i + 1][j + 1]
                    # 记录最大匹配长度的终止位置
                    p = i + 1
    return str1[p - maxNum:p], maxNum

def get_match_notes(diff_base_notes_str,diff_longest_note_str):
    list_intersect,number = getNumofCommonSubstr(diff_base_notes_str, diff_longest_note_str)
    #print("diff_longest_note_str, diff_base_notes_str,intersect is {}==={}==={}".format(diff_longest_note_str, diff_base_notes_str,list_intersect))
    return list_intersect

def check_all_notes_trend(longest_note,base_notes):
    diff_longest_note = []
    diff_base_notes = []
    diff_longest_note_str = ''
    diff_base_notes_str = ''
    for i in range(1, len(longest_note)):
        if longest_note[i] > longest_note[i - 1]:
            tmp = 1
        if longest_note[i] == longest_note[i - 1]:
            tmp = 0
        if longest_note[i] < longest_note[i - 1]:
            tmp = 2
        diff_longest_note.append(str(tmp))
        diff_longest_note_str = ''.join(diff_longest_note)
    for i in range(1, len(base_notes)):
        if base_notes[i] > base_notes[i - 1]:
            tmp = 1
        if base_notes[i] == base_notes[i - 1]:
            tmp = 0
        if base_notes[i] < base_notes[i - 1]:
            tmp = 2
        diff_base_notes.append(str(tmp))
        diff_base_notes_str = ''.join(diff_base_notes)
    list_intersect = get_match_notes(diff_base_notes_str,diff_longest_note_str)
    start1 = diff_longest_note_str.find(list_intersect)
    end1 = start1 + len(list_intersect)

    start2 = diff_base_notes_str.find(list_intersect)
    end2 = start2 + len(list_intersect)
    list_intersect_before = []
    list_intersect_after = []
    if start1 >= 2 and start2 >=2:
        list_intersect_before = get_match_notes(diff_longest_note_str[0:start1],diff_base_notes_str[0:start2])
    if end1 >= 2 and end2 >=2:
        list_intersect_after = get_match_notes(diff_longest_note_str[end1+1:],diff_base_notes_str[end1+1:])

    return list_intersect_before,list_intersect,list_intersect_after

def find_hightest_after(start,rms):
    good_point = start
    for i in range(start+1,len(rms)-1):
        if rms[i] > rms[i-1]:
            good_point += 1
        else:
            break
    return good_point

def find_hightest_before(start,rms):
    good_point = start
    for i in range(start-1,0,-1):
        if rms[i-1] <= rms[i]:
            good_point -= 1
        else:
            break
    return good_point

def change_rms(onsets_frames,note_lines,rms,cqt,topN,rhythm_code):
    rhythm_code = rhythm_code.replace(";", ',')
    rhythm_code = rhythm_code.replace("[", '')
    rhythm_code = rhythm_code.replace("]", '')
    rhythm_code = [x for x in rhythm_code.split(',')]
    result = []
    keyMap = {}
    indexMap = {}
    #select_onset_frames = onsets_frames.copy()
    select_onset_frames = []
    select_onset_frames.append(onsets_frames[0])
    #print("all onsets_frames is {}".format(onsets_frames))
    new_added = []
    small_code_indexs = [i for i in range(len(rhythm_code)) if rhythm_code[i] == '250']

    threshold = 9
    if len(small_code_indexs) > 0:
        threshold = 5
    for i in range(1,len(note_lines)):
        if np.abs(note_lines[i] - note_lines[i-1])>1:
            select_onset_frames.append(onsets_frames[i])
    for i in range(1,len(rms)-3):
        if (i==1 and rms[2] > rms [1]) or (rms[i+1] > rms[i] and rms[i-1] > rms[i]):
            hightest_point_after = find_hightest_after(i, rms)
            if rms[hightest_point_after] - rms[i] > 0.3:
                #print("rms[hightest_point_after] - rms[i],i is {}=={}".format(rms[hightest_point_after] - rms[i],i))
                value = rms[hightest_point_after] - rms[i]
                result.append(value)
                keyMap[value] = i
                indexMap[i] = value
    topN_index = find_n_largest(result,topN)
    topN_key = [result[i] for i in topN_index]
    for key in topN_key:
        index = keyMap.get(key)
        offset = [np.abs(index - x) for x in select_onset_frames]
        if index > onsets_frames[0] and np.min(offset) > threshold:
            select_onset_frames.append(index)
            new_added.append(index)
            #print("add index is {}".format(index))
        elif index < onsets_frames[0] and key > 1.8:
            cols_cqt = cqt[10:, index:onsets_frames[0]]  # 柜向量
            min_cqt = np.min(cqt)
            best_longest, best_begin, best_row = get_longest_for_cols_cqt(cols_cqt, min_cqt)
            if best_longest > 10:
                select_onset_frames.append(index)
                new_added.append(index)
                #print("add index is {}".format(index))
    select_onset_frames.sort()
    #print("all frames is {}".format(select_onset_frames))
    return select_onset_frames,indexMap,new_added
def cal_note_score(longest_note,base_notes):
    detail_content = ''
    a = 0
    b = 0
    c = 0
    off = int(np.mean(base_notes) - np.mean(longest_note))
    # off = int((base_notes[0] - longest_note[0]))
    base_notes = [x - off for x in base_notes]
    #print("base_notes is {}".format(base_notes))
    euclidean_norm = lambda x, y: np.abs(x - y)
    if (len(longest_note) != len(base_notes)):
        each_note_score = 60 / len(base_notes)
        num_gap = len(base_notes)
        longest_note, base_notes = get_matched_note_lines_compared(longest_note, base_notes)
        notes_score = 0
        for i in range(len(longest_note)):
            if np.abs(longest_note[i] - base_notes[i]) <= 1:
                notes_score += each_note_score
                a += 1
            elif np.abs(longest_note[i] - base_notes[i]) <= 2:
                notes_score += each_note_score * 0.5
                b += 1
            else:
                c += 1
        notes_score = int(notes_score)
        #d, cost_matrix, acc_cost_matrix, path = dtw(longest_note, base_notes, dist=euclidean_norm)
        #notes_score = 60 - int(d * np.sum(acc_cost_matrix.shape))
        num_gap = num_gap - len(base_notes)
        detail_content += '识别的音高个数与标准不一致，未匹配的音高个数为' + str(num_gap) + '个。'
    else:
        each_note_score = 60 / len(longest_note)
        notes_score = 0
        for i in range(len(longest_note)):
            if np.abs(longest_note[i] - base_notes[i]) <= 1:
                notes_score += each_note_score
                a += 1
            elif np.abs(longest_note[i] - base_notes[i]) <= 2:
                notes_score += each_note_score * 0.5
                b += 1
            else:
                c += 1
        notes_score = int(notes_score)
    detail_content += '音高偏差较大的有' + str(c) + '个，偏差较小的有' + str(b) + '个，偏差在合理区间的有' + str(a) + '个'
    return notes_score,detail_content

def cal_note_score_by_diff(longest_note,base_notes):
    detail_content = ''
    a = 0
    b = 0
    c = 0
    longest_note = np.diff(longest_note)
    # off = int((base_notes[0] - longest_note[0]))
    base_notes = np.diff(base_notes)
    #print("base_notes is {}".format(base_notes))
    euclidean_norm = lambda x, y: np.abs(x - y)
    if (len(longest_note) != len(base_notes)):
        each_note_score = 60 / len(base_notes)
        num_gap = len(base_notes)
        longest_note, base_notes = get_matched_note_lines_compared(longest_note, base_notes)
        notes_score = 0
        for i in range(len(longest_note)):
            if np.abs(longest_note[i] - base_notes[i]) <=1:
                notes_score += each_note_score
                a += 1
            elif np.abs(longest_note[i] - base_notes[i]) == 2:
                notes_score += each_note_score * 0.8
                b += 1
            elif longest_note[i] * base_notes[i] > 0: # 相同趋势，即都为正，或都为负
                notes_score += each_note_score * 0.8
                b += 1
            else:
                c += 1
        notes_score = int(notes_score)
        #d, cost_matrix, acc_cost_matrix, path = dtw(longest_note, base_notes, dist=euclidean_norm)
        #notes_score = 60 - int(d * np.sum(acc_cost_matrix.shape))
        num_gap = num_gap - len(base_notes)
        detail_content += '识别的音高个数与标准不一致，未匹配的音高个数为' + str(num_gap) + '个。'
    else:
        each_note_score = 60 / len(longest_note)
        notes_score = 0
        longest_note, base_notes = get_matched_note_lines_compared(longest_note, base_notes)
        for i in range(len(longest_note)):
            if np.abs(longest_note[i] - base_notes[i]) <= 1:
                notes_score += each_note_score
                a += 1
            elif np.abs(longest_note[i] - base_notes[i]) == 2:
                notes_score += each_note_score * 0.8
                b += 1
            elif longest_note[i] * base_notes[i] > 0: # 相同趋势，即都为正，或都为负
                notes_score += each_note_score * 0.8
                b += 1
            else:
                c += 1
        notes_score = int(notes_score)
    detail_content += '音高偏差较大的有' + str(c) + '个，偏差较小的有' + str(b) + '个，偏差在合理区间的有' + str(a) + '个'
    return notes_score,detail_content

def cal_score_v2(filename, result, longest_note, base_frames):
    total_score, onsets_score, notes_score, trend_number, base_notes_number = get_score(filename, result, longest_note, base_frames)

    if total_score < 75 and total_score > 60:
        total_score -= 15
        onsets_score -= 6
        notes_score -= 9
    #print("cal_score_v2 total_score, onsets_score, notes_score is {},{},{} ".format(total_score, onsets_score, notes_score))
    return total_score,onsets_score,notes_score
def get_score(filename,result,longest_note,base_frames):

    base_notes = base_note(filename)
    #result, longest_note = del_false_same(base_notes, result, longest_note)
    off = int(np.mean(base_frames) - np.mean(result))
    # off = int((base_notes[0] - longest_note[0]))
    base_frames = [x - off for x in base_frames]
    min_d, best_y, _ = get_dtw_min(result, base_frames, 65)
    onsets_score = 40 - int(min_d)
    if len(result)<len(base_frames)*0.75:
        onsets_score = onsets_score - int(40 * (len(base_frames) - len(result))/len(base_frames))
    #print("onsets_score is {}".format(onsets_score))
    off = int(np.mean(base_notes) - np.mean(longest_note))
    #off = int((base_notes[0] - longest_note[0]))
    base_notes = [x - off for x in base_notes]
    #print("base_notes is {}".format(base_notes))
    euclidean_norm = lambda x, y: np.abs(x - y)
    if(len(longest_note) != len(base_notes)):
        d, cost_matrix, acc_cost_matrix, path = dtw(longest_note, base_notes, dist=euclidean_norm)
        notes_score = 60 - int(d * np.sum(acc_cost_matrix.shape))
    else:
        each_note_score = 60 / len(longest_note)
        notes_score = 0
        for i in range(len(longest_note)):
            if np.abs(longest_note[i] - base_notes[i]) <= 1:
                notes_score += each_note_score
            elif np.abs(longest_note[i] - base_notes[i]) <= 2:
                notes_score += each_note_score * 0.5
        notes_score = int(notes_score)
    if len(longest_note)<len(base_notes)*0.75 and notes_score < 55:
        notes_score = notes_score - int(60 * (len(base_notes) - len(longest_note))/len(base_notes))
    if notes_score <= 0:
        onsets_score = int(onsets_score / 2)
        notes_score = 0
    if notes_score >= 40 and onsets_score <= 5:
        onsets_score = int(40 * notes_score / 60)
    total_score = onsets_score + notes_score
    trend_score,trend_number = check_notes_trend(longest_note,base_notes)
    #print("trend_score is {}".format(60*trend_score))
    if 60*trend_score > notes_score:
        notes_score = int(60*trend_score)
        total_score = notes_score + onsets_score

    #print("1.onsets_score is {}".format(onsets_score))
    #print("1.notes_score is {}".format(notes_score))
    #print("1.total_score is {}".format(total_score))
    #如果动态时间规整得分不高，且总分及格的，重新按相对音高计算得分
    if notes_score < 45 and onsets_score < 35:
            onsets_score = int(40 * trend_number / len(base_notes))
            notes_score = int(60 * trend_number/len(base_notes))
            total_score = notes_score + onsets_score
    # if trend_score/len(base_notes)<0.25 and np.max([onsets_score,notes_score]) < 30 :
    #     total_score = 0
    #print("2.notes_score is {}".format(notes_score))
    #print("2.total_score is {}".format(total_score))

    list_intersect_before, list_intersect, list_intersect_after = check_all_notes_trend(longest_note, base_notes)
    #print("list_intersect_before,list_intersect,list_intersect_after is {},{},{}".format(list_intersect_before,list_intersect,list_intersect_after))
    # 如果总分小于85，则整体偏差分 30/85*total_score
    # if total_score < 85:
    #     gap = int(30/85*total_score)
    #     o_gap = int(gap* 0.4)
    #     n_gap = int(gap*0.6)
    #     total_score -= gap
    #     onsets_score -= o_gap
    #     notes_score -= n_gap
    return total_score,onsets_score,notes_score,trend_number,len(base_notes)

def check_notes_trend(longest_note,base_notes):
    diff_longest_note = []
    diff_base_notes = []
    diff_longest_note_str = ''
    diff_base_notes_str = ''
    for i in range(1,len(longest_note)):
        if longest_note[i] > longest_note[i-1]:
            tmp = 1
        if longest_note[i] == longest_note[i-1]:
            tmp = 0
        if longest_note[i] < longest_note[i-1]:
            tmp = 2
        diff_longest_note.append(str(tmp))
        diff_longest_note_str = ''.join(diff_longest_note)
    for i in range(1,len(base_notes)):
        if base_notes[i] > base_notes[i-1]:
            tmp = 1
        if base_notes[i] == base_notes[i-1]:
            tmp = 0
        if base_notes[i] < base_notes[i-1]:
            tmp = 2
        diff_base_notes.append(str(tmp))
        diff_base_notes_str = ''.join(diff_base_notes)

    list_intersect,number = getNumofCommonSubstr(diff_base_notes_str, diff_longest_note_str)
    #print("diff_longest_note_str, diff_base_notes_str,intersect is {}==={}==={}".format(diff_longest_note_str, diff_base_notes_str,list_intersect))
    #print("find intersect index is {}".format(diff_longest_note_str.find(list_intersect)))
    start = diff_longest_note_str.find(list_intersect)
    end = start + len(list_intersect)
    intersect_longest_note = longest_note[start:end+1]
    #print("sub longest_note is {}".format(intersect_longest_note))
    start = diff_base_notes_str.find(list_intersect)
    end = start + len(list_intersect)
    intersect_base_notes = base_notes[start:end+1]
    #print("sub base_notes is {}".format(intersect_base_notes))
    intersect_score = [1 if np.abs(intersect_longest_note[i] - intersect_base_notes[i])<1 else 0 for i in range(len(intersect_base_notes))]
    score = len(list_intersect)/len(diff_base_notes_str)

    return score,len(list_intersect)

def cal_score_v1(filename,onsets_frames,note_lines,base_frames, base_notes,times,rhythm_code):
    #type_index = get_onsets_index_by_filename_rhythm(filename)
    onset_score, lost_score, ex_score, min_d,detail_content = get_score_for_note_v2(onsets_frames.copy(), base_frames.copy(), rhythm_code)
    #print("score, lost_score, ex_score, min_d is {},{},{},{}".format(onset_score, lost_score, ex_score, min_d))
    # onset_score = cal_dtw_distance(onsets_frames.copy(), base_frames.copy())

    if onset_score == 0:
        return 0, 0, 0,''
    note_score,note_detail_content = cal_note_score_by_diff(note_lines, base_notes)
    onset_score = int(onset_score * 0.4)
    score = onset_score + note_score

    # if onset_score >= 36 and score < 90:
    #     note_score = int(60 * onset_score / 40)
    # elif onset_score > 30 and note_score < 36:
    #     note_score = int(60 * onset_score / 40)

    # if note_score > 50 and onset_score < 25:
    #     detail_content = '音高得分较好，节奏方面需要进一步加强'
    #     if onset_score<20:
    #         onset_score = onset_score + int(20 / 60 * note_score)
    #     else:
    #         onset_score = onset_score + int(15 / 60 * note_score)
    # elif note_score> 50 and score <= 8 and score > 70:
    #     onset_score = int(40 / 60 * note_score)
    #     print("changed 2")

    score = onset_score + note_score

    # # 漏唱数过多
    if onset_score < 10:
        detail_content = '识别的节奏与标准节奏偏差较大，节奏得分过低，整体得分计为不合格'
        note_score = int(60 * onset_score / 40)
        score = onset_score + note_score

    #旋律分小于0
    if note_score < 5:
        note_score = int(60 * onset_score / 40)
        score = onset_score + note_score
        detail_content = '识别的音高与标准音高偏差较大，旋律得分过低，整体得分以节奏为准'

    #判断每个节拍的时长
    total_offset = 0
    for i in range(1,len(onsets_frames)):
        if times[i-1] <= (onsets_frames[i] - onsets_frames[i-1])/3:
            total_offset += 1
    if total_offset >= 2 and score >= 90:
        tmp_score = int(5 * total_offset)
        score, onset_score, note_score = score - tmp_score, onset_score - int(tmp_score/2), note_score - int(tmp_score/2)
        detail_content = '存在节奏时长不够的情况，整体得分减扣相应的分值'
    #如果节拍完全吻合数较多
    length = len(onsets_frames) if len(onsets_frames) < len(base_frames) else len(base_frames)
    base_frames = [x - (base_frames[0] - onsets_frames[0]) for x in base_frames]
    onset_diff = [1 if onsets_frames[i]-base_frames[i]<2 else 0 for i in range(length)]
    # if len(onsets_frames) == len(base_frames):
    #     if np.sum(onset_diff)>=5 and score<91:
    #         score, onset_score, note_score = 91,40,51
    #         print("changed 6")
    # if np.abs(len(onsets_frames) - len(base_frames)) <= 3:
    #     onset_diff = [1 if onsets_frames[i] - base_frames[i] < 4 else 0 for i in range(length)]
    #     if np.sum(onset_diff)>=5 and score<80:
    #         score, onset_score, note_score = 80,35,45
    #         print("changed 7")
    detail_content = detail_content + '。' + note_detail_content
    return score,onset_score, note_score,detail_content

def del_middle_by_cqt(onset_frames,indexMap,cqt,rms):
    result = []
    w,h = cqt.shape
    i = 0
    for x in onset_frames:
        value = indexMap.get(x)
        if not value is None and value > 2.0:
            result.append(x)
        else:
            hightest_point_after = find_hightest_after(x, rms)
            hightest_point_before = find_hightest_before(x, rms)
            gap = rms[hightest_point_after] - rms[hightest_point_before]
            # 判断振幅变化程度
            if gap > 2.0:
                result.append(x)
            else:
                # if x < len(result) - 1:
                #     next = result[i + 1]
                # else:
                #     next = h - 10
                # sub_cqt = cqt[:, x:next]
                # next_note_line, next_longest_num = get_longest_note_line(sub_cqt)
                # if x == onset_frames[0]:
                #     last = onset_frames[0] - 10 if onset_frames[0] > 10 else 0
                # else:
                #     last = onset_frames[i - 1]
                # sub_cqt = cqt[:, last:x]
                # last_note_line, last_longest_num = get_longest_note_line(sub_cqt)
                # # 判断前后音高线是否相等
                # if np.abs(last_note_line - next_note_line) > 3:
                #     result.append(x)
                # else:
                flag = check_middle_by_cqt(x,cqt)
                if flag is False:
                    result.append(x)
        i += 1
    return result
def check_middle_by_cqt(onset_frame,cqt):
    w,h = cqt.shape
    max_cqt = np.max(cqt)
    min_cqt = np.min(cqt)
    col_cqt = cqt[:,onset_frame]
    flag = False
    for i in range(w-10,30,-1):
        if np.min(col_cqt[i-1:i+1]) == max_cqt:
            if np.max(cqt[i+1,onset_frame-3:onset_frame + 3]) == min_cqt and np.min(cqt[i,onset_frame-3:onset_frame + 3]) == max_cqt:
                flag = True
            break
    return flag

def check_noice(sub_cqt):
    w, h = sub_cqt.shape
    # print("w,h is {},{}".format(w,h))
    if h > 10:
        sub_cqt = sub_cqt[:,0:10]

    rows = []
    if h > 0:
        min_cqt = np.min(sub_cqt)
        for row in range(10, w - 10):
            row_cqt = sub_cqt[row]
            row_cqt = [1 if row_cqt[i] > min_cqt else 0 for i in range(len(row_cqt))]
            one_number = str(row_cqt).count("1")
            zero_number = str(row_cqt).count("0")
            if one_number > zero_number:
                rows.append(1)
            else:
                rows.append(0)
        total_continue = continueOne(rows)
        if total_continue > 8:
            return True,total_continue
        else:
            return False,total_continue
def check_fisrt_frame_is_noice(onset_frames,times, note_lines,cqt):
   if len(onset_frames) >= 2:
       sub_cqt = cqt[:,onset_frames[0]:onset_frames[1]]
       flag,total_continue = check_noice(sub_cqt)
       if flag:
           index = 1
           for i in range(2,len(onset_frames)):
               if onset_frames[i] > onset_frames[0] + total_continue + 5:
                   index = i
                   break
           return onset_frames[index:],times[index:], note_lines[index:]
       else:
           return onset_frames,times, note_lines
   else:
       sub_cqt = cqt[:, onset_frames[0]:onset_frames[0]+10]
       flag,total_continue  = check_noice(sub_cqt)
       if flag:
           index = 1
           for i in range(2, len(onset_frames)):
               if onset_frames[i] > onset_frames[0] + total_continue + 5:
                   index = i
                   break
           return onset_frames[index:], times[index:], note_lines[index:]
       else:
           return onset_frames,times, note_lines

def find_maybe_position_rhythm_code(start,total_frames,rhythm_code):
    index = 0
    rhythm_code = rhythm_code.replace(";", ',')
    rhythm_code = rhythm_code.replace("[", '')
    rhythm_code = rhythm_code.replace("]", '')
    if rhythm_code.find("(") >=0:
        tmp = [x for x in rhythm_code.split(',')]
        for i in range(len(tmp)):
            if tmp[i].find("(") >= 0:
                index = i
                break
        rhythm_code = rhythm_code.replace("(", '')
        rhythm_code = rhythm_code.replace(")", '')
        rhythm_code = rhythm_code.replace("-", '')
        rhythm_code = rhythm_code.replace("--", '')
    rhythm_code = [x for x in rhythm_code.split(',')]
    rhythm_code = [int(x) for x in rhythm_code]
    if index > 0:
        rhythm_code[index-1] += rhythm_code[index]
        del rhythm_code[index]

    sum_code = np.sum(rhythm_code)
    onset_frames = []
    onset_frames.append(start)
    for code in rhythm_code:
        next_onset = onset_frames[-1] + int(total_frames/sum_code*code)
        onset_frames.append(next_onset)
    return onset_frames
def modify_by_local_min_rms(onsets_frames,rms):
    select_onsets_frames = []
    for i in range(len(onsets_frames)):
        frame = onsets_frames[i]
        before_frame = get_local_min_before(frame,rms)
        select_onsets_frames.append(before_frame)
    return select_onsets_frames

def get_local_min_before(frame,rms):
    before_frame = frame
    for i in range(10):
        if rms[frame-i-1] < rms[frame-i]:
            before_frame = frame-i-1
        else:
            break
    return before_frame

def split_long_note_line(onsets_frames, note_lines, times,pitch_code):
    flag = False
    pitch_code = pitch_code.replace(":", ',')
    pitch_code = pitch_code.replace("[", '')
    pitch_code = pitch_code.replace("]", '')
    pitch_code = [x for x in pitch_code.split(',')]
    select_onsets_frames, select_note_lines, select_times = [],[],[]
    for i in range(1,len(pitch_code)-1):
        if pitch_code[i-1] ==  pitch_code[i] and pitch_code[i] ==  pitch_code[i+1]:
            flag = True
            break
    if flag is True:
        for i in range(len(onsets_frames)):
            if times[i]>60:
                split_length = int(times[i]/3)
                select_onsets_frames.append(onsets_frames[i])
                select_note_lines.append(note_lines[i])
                select_times.append(split_length)

                select_onsets_frames.append(onsets_frames[i] + split_length)
                select_note_lines.append(note_lines[i])
                select_times.append(split_length)

                select_onsets_frames.append(onsets_frames[i] + split_length*2)
                select_note_lines.append(note_lines[i])
                select_times.append(split_length)
            else:
                select_onsets_frames.append(onsets_frames[i])
                select_note_lines.append(note_lines[i])
                select_times.append(times[i])
    else:
        return onsets_frames, note_lines, times

    return select_onsets_frames, select_note_lines, select_times

def find_loss_by_rms_and_rhythm_code(onset_frames,note_lines, times,base_frames,end_point,CQT,rms,rhythm_code):
    rms = np.diff(rms, 2)
    rms, all_indexs = filter_hold_local_max(rms, base_frames)
    # print("********onset_frames is {}".format(onset_frames))
    # print("********base_frames is {}".format(base_frames))
    # print("********all_indexs is {}".format(all_indexs))
    selected_indexs = []
    min_gap = np.min(np.diff(base_frames))
    end = end_point
    for x in all_indexs:
        offset = [np.abs(x - y ) for y in onset_frames]
        min_gap_offset = np.min(offset)
        if min_gap_offset > min_gap * 0.6 and x > base_frames[0] and x < end:
            selected_indexs.append(x)
    number = len(base_frames) - len(onset_frames)
    if number < 1 or number >3:
        return onset_frames,note_lines, times,[]
    combinations = get_combinations(selected_indexs, number)
    if len(combinations) < 1:
        return onset_frames, note_lines, times,[]
    best_score = 0
    best_frames = []
    best_cb = []
    for cb in combinations:
        for n in range(len(cb)):
            tmp = onset_frames.copy()
            for i in range(number):
                tmp.append(cb[n][i])
            tmp.sort()
            onset_score,lost_score, ex_score, min_d,detail_content = get_score_for_note_v2(tmp,base_frames.copy(),rhythm_code)
            if onset_score > best_score:
                best_score = onset_score
                best_frames = tmp
                best_cb = cb[n]
            print(" cb[n],onset_score {}===={}===={}".format(cb[n],onset_score,tmp))
    onsets_frames, note_lines, times = get_note_lines(CQT, best_frames)
    return onsets_frames, note_lines, times,best_cb

def del_too_small_frames_by_rms(onsets_frames, note_lines,times,rms,base_frames,best_cb):
    rms = np.diff(rms, 2)
    rms, all_indexs = filter_hold_local_max(rms, base_frames)
    rms_max = np.max(rms)
    select_onset_frames = []
    select_note_lines = []
    select_times = []
    for i in range(len(onsets_frames)):
        frame = onsets_frames[i]
        offset = [np.abs(frame - x) for x in all_indexs]
        min_index = np.min(offset)
        nearly_index = offset.index(min_index)
        nearly_frame = all_indexs[nearly_index]
        if nearly_frame not in best_cb and rms[nearly_frame] < rms_max*0.1:
            continue
        select_onset_frames.append(frame)
        select_note_lines.append(note_lines[i])
        select_times.append(times[i])
    return select_onset_frames,select_note_lines,select_times

def draw_plt(filename,rhythm_code,pitch_code):
    y, sr = librosa.load(filename)
    rms = librosa.feature.rmse(y=y)[0]
    rms = [x / np.std(rms) for x in rms]
    time = librosa.get_duration(filename=filename)
    print("time is {}".format(time))
    CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref=np.max)
    CQT = np.where(CQT > -22, np.max(CQT), np.min(CQT))
    score1, onset_score1, note_score1,detail_content,onsets_frames,maybe_onset_frames = get_melody_score(filename,rhythm_code,pitch_code)
    base_frames = maybe_onset_frames[:-1]
    print("score, onset_score, note_scroe is {},{},{}".format(score1, onset_score1, note_score1 ))

    # score2, onset_score2, note_score2 = cal_score_v2(filename, onsets_frames, note_lines, base_frames)
    #
    # if max(score1,score2)>85:
    #     if score1 >= score2:
    #         score, onset_score, note_score = score1,onset_score1, note_score1
    #     else:
    #         score, onset_score, note_score = score2, onset_score2, note_score2
    # else:
    #     score, onset_score, note_score = score1, onset_score1, note_score1
    score, onset_score, note_score = score1, onset_score1, note_score1
    plt.subplot(3, 1, 1)
    # librosa.display.specshow(CQT)
    #librosa.display.specshow(CQT, y_axis='cqt_note', x_axis='time')
    librosa.display.specshow(CQT, x_axis='time')
    print(np.max(y))
    onstm = librosa.frames_to_time(onsets_frames, sr=sr)
    #end_time = librosa.frames_to_time(end_result, sr=sr)
    # end_position_time = librosa.frames_to_time(end_position, sr=sr)
    plt.vlines(onstm, 0, sr, color='y', linestyle='dashed')
    #plt.vlines(end_time, 0, sr, color='r', linestyle='dashed')
    # plt.vlines(end_position_time, 0,sr, color='r', linestyle='solid')
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Constant-Q power spectrogram (note)')
    plt.subplot(3, 1, 2)
    times = librosa.frames_to_time(np.arange(len(rms)))
    print("base_frames is {},size is {}".format(base_frames,len(base_frames)))
    base_frames = [x - (base_frames[0] - onsets_frames[0]) for x in base_frames]
    #base_frames = base_frames[3:8]
    #min_d, best_y, _ = get_dtw_min(onsets_frames, base_frames, 100)
    #base_times = librosa.frames_to_time(best_y, sr=sr)
    base_times = librosa.frames_to_time(base_frames, sr=sr)
    the_end_time = librosa.frames_to_time(maybe_onset_frames[-1], sr=sr)
    print("last is :{}".format(maybe_onset_frames[-1]))
    # rms_on_onset_frames_cqt = [rms[x] for x in onset_frames_cqt]
    # min_rms_on_onset_frames_cqt = np.min(rms_on_onset_frames_cqt)
    # rms = [1 if x >=min_rms_on_onset_frames_cqt else 0 for x in rms]
    plt.vlines(onstm, 0, np.max(rms), color='y', linestyle='dashed')
    plt.vlines(base_times, 0, np.max(rms), color='r', linestyle='dashed')
    plt.vlines(the_end_time, 0, np.max(rms), color='black', linestyle='dashed')
    rms_on_frames = [rms[x] for x in onsets_frames]
    mean_rms_on_frames = np.mean(rms_on_frames)
    plt.plot(times, rms)
    plt.xlim(0, np.max(times))
    plt.axhline(mean_rms_on_frames, color='r')

    plt.subplot(3, 1, 3)
    rms = np.diff(rms,1)
    rms = [x if x > 0 else 0 for x in rms]

    rms,max_indexs = filter_hold_local_max(rms, base_frames)
    max_indexs_time = librosa.frames_to_time(max_indexs, sr=sr)
    times = librosa.frames_to_time(np.arange(len(rms)))

    # rms_on_onset_frames_cqt = [rms[x] for x in onset_frames_cqt]
    # min_rms_on_onset_frames_cqt = np.min(rms_on_onset_frames_cqt)
    # rms = [1 if x >=min_rms_on_onset_frames_cqt else 0 for x in rms]
    rms_on_frames = [rms[x] for x in onsets_frames]
    mean_rms_on_frames = np.mean(rms_on_frames)
    plt.plot(times, rms)
    plt.axhline(mean_rms_on_frames, color='r')
    plt.vlines(onstm, 0, np.max(rms), color='y', linestyle='dashed')
    plt.vlines(max_indexs_time, 0, np.max(rms), color='r', linestyle='dashed')
    plt.xlim(0, np.max(times))
    return plt,score,onset_score, note_score,detail_content

def draw_plt_v2(filename,rhythm_code,pitch_code):
    y, sr = librosa.load(filename)
    rms = librosa.feature.rmse(y=y)[0]
    rms = [x / np.std(rms) for x in rms]
    time = librosa.get_duration(filename=filename)
    print("time is {}".format(time))
    CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref=np.max)
    CQT = np.where(CQT > -22, np.max(CQT), np.min(CQT))
    score1, onset_score1, note_score1,detail_content, CQT, rms, result, onsets_frames, starts_index = get_melody_score_v2(filename,rhythm_code,pitch_code)

    start, end, length = get_frame_length(result)
    base_frames = onsets_base_frames_rhythm(rhythm_code, length)
    base_frames = [x - (base_frames[0] - start) for x in base_frames]

    print("score, onset_score, note_scroe is {},{},{}".format(score1, onset_score1, note_score1 ))

    # score2, onset_score2, note_score2 = cal_score_v2(filename, onsets_frames, note_lines, base_frames)
    #
    # if max(score1,score2)>85:
    #     if score1 >= score2:
    #         score, onset_score, note_score = score1,onset_score1, note_score1
    #     else:
    #         score, onset_score, note_score = score2, onset_score2, note_score2
    # else:
    #     score, onset_score, note_score = score1, onset_score1, note_score1
    score, onset_score, note_score = score1, onset_score1, note_score1
    plt.subplot(3, 1, 1)
    # librosa.display.specshow(CQT)
    #librosa.display.specshow(CQT, y_axis='cqt_note', x_axis='time')
    librosa.display.specshow(CQT, x_axis='time')
    print(np.max(y))
    onstm = librosa.frames_to_time(onsets_frames, sr=sr)
    #end_time = librosa.frames_to_time(end_result, sr=sr)
    # end_position_time = librosa.frames_to_time(end_position, sr=sr)
    plt.vlines(onstm, 0, sr, color='y', linestyle='dashed')
    #plt.vlines(end_time, 0, sr, color='r', linestyle='dashed')
    # plt.vlines(end_position_time, 0,sr, color='r', linestyle='solid')
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Constant-Q power spectrogram (note)')
    plt.subplot(3, 1, 2)
    times = librosa.frames_to_time(np.arange(len(rms)))
    print("base_frames is {},size is {}".format(base_frames,len(base_frames)))
    base_frames = [x - (base_frames[0] - onsets_frames[0]) for x in base_frames]
    #base_frames = base_frames[3:8]
    #min_d, best_y, _ = get_dtw_min(onsets_frames, base_frames, 100)
    #base_times = librosa.frames_to_time(best_y, sr=sr)
    base_times = librosa.frames_to_time(base_frames, sr=sr)
    the_end_time = librosa.frames_to_time(end, sr=sr)
    # rms_on_onset_frames_cqt = [rms[x] for x in onset_frames_cqt]
    # min_rms_on_onset_frames_cqt = np.min(rms_on_onset_frames_cqt)
    # rms = [1 if x >=min_rms_on_onset_frames_cqt else 0 for x in rms]
    #plt.vlines(onstm, 0, np.max(rms), color='y', linestyle='dashed')
    #plt.vlines(base_times, 0, np.max(rms), color='r', linestyle='dashed')
    plt.vlines(the_end_time, 0, np.max(rms), color='black', linestyle='dashed')
    rms_on_frames = [rms[x] for x in onsets_frames]
    mean_rms_on_frames = np.mean(rms_on_frames)
    rms_max = [i for i in range(1,len(rms)-1) if rms[i] > rms[i-1] and rms[i] > rms[i+1] and rms[i] > mean_rms_on_frames]
    rms_max_time = librosa.frames_to_time(rms_max, sr=sr)
    plt.plot(times, rms)
    plt.xlim(0, np.max(times))
    plt.axhline(mean_rms_on_frames, color='r')
    plt.vlines(rms_max_time, 0, np.max(rms), color='b', linestyle='dashed')

    plt.subplot(3, 1, 3)
    rms = np.diff(rms,1)
    rms = [x if x > 0 else 0 for x in rms]

    rms,max_indexs = filter_hold_local_max(rms, base_frames)
    max_indexs_time = librosa.frames_to_time(max_indexs, sr=sr)
    times = librosa.frames_to_time(np.arange(len(rms)))


    # rms_on_onset_frames_cqt = [rms[x] for x in onset_frames_cqt]
    # min_rms_on_onset_frames_cqt = np.min(rms_on_onset_frames_cqt)
    # rms = [1 if x >=min_rms_on_onset_frames_cqt else 0 for x in rms]
    rms_on_frames = [rms[x] for x in onsets_frames]
    mean_rms_on_frames = np.mean(rms_on_frames)

    plt.plot(times, rms)
    plt.axhline(mean_rms_on_frames, color='r')
    plt.vlines(onstm, 0, np.max(rms), color='y', linestyle='dashed')
    plt.vlines(max_indexs_time, 0, np.max(rms), color='r', linestyle='dashed')

    plt.xlim(0, np.max(times))
    return plt,score,onset_score, note_score,detail_content

def get_melody_score(filename,rhythm_code,pitch_code):

    y, sr = librosa.load(filename)
    rms = librosa.feature.rmse(y=y)[0]
    rms = [x / np.std(rms) for x in rms]
    time = librosa.get_duration(filename=filename)
    #print("time is {}".format(time))
    CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref=np.max)
    w, h = CQT.shape
    #print("w.h is {},{}".format(w, h))
    # onsets_frames = get_real_onsets_frames_rhythm(y)
    base_frames, frames_total = onsets_base_frames_for_note(filename, rhythm_code)
    base_notes = base_note(filename, pitch_code)
    CQT = np.where(CQT > -22, np.max(CQT), np.min(CQT))
    # onsets_frames,end_position = get_note_line_start(CQT)
    # rhythm_codes = get_rhythm_codes(filename)
    onsets_frames, end_result, times, note_lines = get_note_line_start_v2(CQT, rhythm_code)
    #print("frames_total ===== is {}".format(onsets_frames[-1] + times[-1] - onsets_frames[0]))
    # onsets_frames, end_result, times, note_lines = merge_note_line(onsets_frames, end_result, times, note_lines)
    # onsets_frames = find_loss_by_rms_mean(onsets_frames, rms, CQT)
    #print("get_note_line_start_v2====01 onsets_frames is {}".format(onsets_frames))
    onsets_frames, indexMap, new_added = change_rms(onsets_frames, note_lines, rms, CQT, len(base_frames), pitch_code)
    #print("change_rms====02 onsets_frames is {}".format(onsets_frames))
    onsets_frames, times, note_lines = check_fisrt_frame_is_noice(onsets_frames, times, note_lines, CQT)
    onsets_frames = modify_by_local_min_rms(onsets_frames, rms)
    onsets_frames = del_middle_by_cqt(onsets_frames, indexMap, CQT, rms)
    #print("del_middle_by_cqt====03 onsets_frames is {}".format(onsets_frames))
    onsets_frames, note_lines, times = get_note_lines(CQT, onsets_frames)
    #print("get_note_lines====04 onsets_frames is {}".format(onsets_frames))
    onsets_frames, note_lines, times = del_false_same(base_notes, onsets_frames, note_lines, times, indexMap)
    #print("del_false_same====05 onsets_frames is {}".format(onsets_frames))
    #print("before check_all_note_lines is {}".format(note_lines))
    note_lines = check_all_note_lines(onsets_frames, note_lines, CQT)
    #print("after check_all_note_lines is {}".format(note_lines))
    onsets_frames, note_lines, times = del_end_range(onsets_frames, note_lines, times, rms, frames_total, base_frames)
    #print("del_end_range====06 onsets_frames is {}".format(onsets_frames))
    onsets_frames, times, note_lines = merge_note_line_by_distance(onsets_frames, times, note_lines, rhythm_code,
                                                                   end_result[-1] - onsets_frames[0])
    #print("merge_note_line_by_distance====07 onsets_frames is {}".format(onsets_frames))
    onsets_frames, note_lines, times = get_note_lines(CQT, onsets_frames)
    note_lines = check_all_note_lines(onsets_frames, note_lines, CQT)
    #print("get_note_lines====08 onsets_frames is {}".format(onsets_frames))
    # onsets_frames, note_lines, times = split_long_note_line(onsets_frames, note_lines, times, pitch_code)
    # print("0 onsets_frames is {},size is {}".format(onsets_frames,len(onsets_frames)))
    # print("0 end_result is {},size is {}".format(end_result,len(end_result)))
    # print("0 times is {},size is {}".format(times,len(times)))
    # print("0 note_lines is {},size is {}".format(note_lines,len(note_lines)))
    # list_intersect_before, list_intersect, list_intersect_after = check_all_notes_trend(note_lines, base_notes)
    # print("list_intersect_before,list_intersect,list_intersect_after is {},{},{}".format(list_intersect_before,
    #                                                                                      list_intersect,
    #                                                                                      list_intersect_after))
    if len(times) == len(onsets_frames):
        tmp = onsets_frames[-1] + times[-1] - onsets_frames[0]
        frames_total = frames_total if frames_total > tmp else tmp
    maybe_onset_frames = find_maybe_position_rhythm_code(onsets_frames[0], frames_total, rhythm_code)
    base_frames = maybe_onset_frames[:-1]

    onsets_frames, note_lines, times, best_cb = find_loss_by_rms_and_rhythm_code(onsets_frames, note_lines, times,
                                                                                 base_frames, maybe_onset_frames[-1],
                                                                                 CQT, rms, rhythm_code)
    onsets_frames, note_lines, times = del_real_middle_onset_frames(onsets_frames, note_lines, times, CQT)
    # onsets_frames, note_lines, times = del_too_small_frames_by_rms(onsets_frames, note_lines, times, rms,base_frames,best_cb)
    onsets_frames = [onsets_frames[i] for i in range(len(times)) if times[i] > 3]
    note_lines = [note_lines[i] for i in range(len(times)) if times[i] > 3]
    times = [times[i] for i in range(len(times)) if times[i] > 3]
    # print("maybe_onset_frames  is {}".format(maybe_onset_frames))
    # print("final onsets_frames is {}".format(onsets_frames))
    # print("final note_lines is {}".format(note_lines))
    # print("final times is {}".format(times))

    #如果节拍数还是少于标准数，则从疑似节拍中添加
    if len(onsets_frames) < len(base_frames) :
        #print('lossssssssss')
        onsets_frames, note_lines, times = add_from_maybe_onset_frames(onsets_frames, maybe_onset_frames, CQT, rms, base_frames)

    onsets_frames, note_lines, times = check_with_rms_onset(onsets_frames, note_lines, CQT, rms,maybe_onset_frames[-1],base_frames)

    score1, onset_score1, note_score1, detail_content = cal_score_v1(filename, onsets_frames, note_lines, base_frames,base_notes, times, rhythm_code)

    #print("score, onset_score, note_scroe is {},{},{}".format(score1, onset_score1, note_score1 ))

    score, onset_score, note_score = score1, onset_score1, note_score1

    return score,onset_score, note_score,detail_content,onsets_frames,maybe_onset_frames

def get_melody_score_v2(filename,rhythm_code,pitch_code):
    y, sr = load_and_trim(filename)
    y,sr = librosa.load(filename)
    rms = librosa.feature.rmse(y=y)[0]
    rms = [x / np.std(rms) for x in rms]
    time = librosa.get_duration(filename=filename)
    print("time is {}".format(time))
    CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref = np.max)
    w,h = CQT.shape
    print("w.h is {},{}".format(w,h))
    #onsets_frames = get_real_onsets_frames_rhythm(y)
    CQT = np.where(CQT > -22, np.max(CQT), np.min(CQT))
    max_cqt = np.max(CQT)
    min_cqt = np.min(CQT)
    result = []
    starts = []
    last_i = 1
    for i in range(1,h-4):
        col_cqt = CQT[10:,i]
        before_col_cqt = CQT[10:,i-1]
        after_col_cqt = CQT[10:, i+3]
        max_sum = np.sum([1 if x > np.min(col_cqt) else 0 for x in col_cqt])
        before_max_sum = np.sum([1 if x > np.min(before_col_cqt) else 0 for x in before_col_cqt])
        after_before_max_sum = np.sum([1 if x > np.min(after_col_cqt) else 0 for x in after_col_cqt])
        #sum = np.sum(np.array(col_cqt) - np.array(before_col_cqt))
        sum = np.sum([1 if (before_col_cqt[i] == min_cqt and col_cqt[i] == max_cqt) and max_sum > 0.7*before_max_sum else 0 for i in range(len(col_cqt))])
        start = np.sum([1 if max_sum > 1.3*before_max_sum and before_max_sum <= 1 and after_before_max_sum > 2 else 0 for i in range(len(col_cqt))])
        result.append(sum)
        if start == 0:
            starts.append(start)
        else:
            if i - last_i > 8:
                starts.append(start)
                last_i = i

        # for n in range(w-1,10,-1):
        #     if i >10:
        #         before_row = CQT[n,i-4:i]
        #         after_row = CQT[n, i:i+8]
        #         if np.max(before_row) == min_cqt and np.min(after_row) == max_cqt:
        #             starts.append(start)
        #             last_i = i


    result = [x if x>0 else 0 for x in result]
    result = [x/np.max(result) for x in result]
    result = [x if x>0.1 else 0 for x in result]


    starts = [x/np.max(starts) for x in starts]
    starts = [starts[i] if starts[i] ==1 and starts[i-1] == 0 else 0 for i in range(1,len(starts))]
    starts.insert(0,0)
    starts_index = [i for i in range(len(starts)) if starts[i]>0]
    print("==============starts_index is {}".format(starts_index))
    # for i in range(1,len(result)):
    #     offset = [np.abs(i-x) for x in starts_index]
    #     if (result[i] > 0.46)and np.min(offset) > 8:
    #         starts_index.append(i)

    start, end, length = get_frame_length(result)
    base_frames = onsets_base_frames_rhythm(rhythm_code, length)
    base_frames = [x - (base_frames[0] - start - 1) for x in base_frames]
    #result, max_indexs = filter_hold_local_max(result, base_frames)
    max_indexs = [i for i in range(len(result)) if i > start and i < end - 3 and result[i] > 0]
    onsets_frames, note_lines, note_times = get_note_lines_v2(CQT, max_indexs)

    print("==============note_lines is {}".format(note_lines))
    print("==============note_times is {}".format(note_times))
    note_lines_diff = np.diff(note_lines)
    max_indexs = [max_indexs[i + 1] for i in range(len(note_lines_diff)) if np.abs(note_lines_diff[i]) != 0 and note_times[i+1] > 5]
    for m in max_indexs:
        offset = [np.abs(m-x) for x in starts_index]
        if np.min(offset) > 8:
            starts_index.append(m)

    starts_index.sort()
    print("==============starts_index is {}".format(starts_index))
    rms = np.diff(rms, 1)
    rms = [x if x > 0 else 0 for x in rms]
    rms = [x / np.max(rms) for x in rms]
    rms_max_indexs = [i for i in range(1, len(rms) - 1) if
                      rms[i] > rms[i - 1] and rms[i] > rms[i + 1] and rms[i] > 0.1 or rms[i] > 0.6]

    print("rms_max_index is {}".format(rms_max_indexs))
    start, end, length = get_frame_length(result)
    base_frames = onsets_base_frames_rhythm(rhythm_code, length)
    base_frames = [x - (base_frames[0] - start) for x in base_frames]
    select_starts_index = add_loss_by_sub_dtw(rms_max_indexs, starts_index.copy(), base_frames)
    select_starts_index = starts_index.copy()
    print("select_starts_index is {}".format(select_starts_index))
    # plt.vlines(select_starts_index, 0, np.max(result), color='r', linestyle='dashed')
    onsets_frames = select_starts_index
    onsets_frames, note_lines, note_times = get_note_lines_v2(CQT, onsets_frames)
    onsets_frames = [onsets_frames[i] for i in range(len(note_times)) if note_times[i] > 5]
    note_lines = [note_lines[i] for i in range(len(note_times)) if note_times[i] > 5]
    note_times = [note_times[i] for i in range(len(note_times)) if note_times[i] > 5]
    base_notes = base_note(filename, pitch_code)
    score, onset_score, note_score, detail_content = cal_score_v1(filename, onsets_frames, note_lines, base_frames, base_notes, note_times, rhythm_code)

    if note_score > 55 and onset_score < 30:
        detail_content = detail_content + ',音高得分较好，节奏方面需要进一步加强'
        onset_score = int(note_score / 60 * 40)
        score = onset_score + note_score
    print('total_score,onset_score, note_scroe is ' + str(score) + ' ' + str(onset_score) + ' ' + str(note_score))

    return score, onset_score, note_score, detail_content,CQT,rms,result,onsets_frames,starts_index

def get_frame_length(cqt_col_diff):
    end = len(cqt_col_diff)
    for i in range(2,len(cqt_col_diff)):
        if np.max(cqt_col_diff[:i-1]) == 0 and cqt_col_diff[i] >0.1:
            start = i

    for i in range(len(cqt_col_diff)-2,0,-1):
        if np.max(cqt_col_diff[i+1:]) == 0 and cqt_col_diff[i] >0.1:
            end = i
    return start,end,end-start

def get_the_nearly_base_frame(frame,base_frames,last_frame,last_base_frame):
    offset = [np.abs(frame - x) for x in base_frames]
    nearly_index = offset.index(np.min(offset))
    if base_frames[nearly_index] > frame: #位于右侧
        if nearly_index -1 <= 0:
            return base_frames[nearly_index], nearly_index
        else:
            left_base_frame = base_frames[nearly_index-1]
            right_base_frame = base_frames[nearly_index]
            left_gap = left_base_frame - last_base_frame - (frame - last_frame)
            right_gap = right_base_frame - last_base_frame - (frame - last_frame)
            if left_gap < right_gap:
                return left_base_frame, nearly_index-1
            else:
                return right_base_frame, nearly_index

    elif base_frames[nearly_index] < frame: #位于左侧
        if nearly_index +1 >= len(base_frames):
            return base_frames[nearly_index], nearly_index
        else:
            left_base_frame = base_frames[nearly_index]
            right_base_frame = base_frames[nearly_index +1]
            left_gap = left_base_frame - last_base_frame - (frame - last_frame)
            right_gap = right_base_frame - last_base_frame - (frame - last_frame)
            if left_gap < right_gap:
                return left_base_frame,nearly_index
            else:
                return right_base_frame, nearly_index+1
    else:
        return base_frames[nearly_index],nearly_index

def get_all_the_nearly_base_frame(onset_frames,base_frames):
    all_nearly_base_frames = []
    all_nearly_indexs = []
    for x in onset_frames:
        if x == onset_frames[0]:
            nearly_base_frames,nearly_index =  base_frames[0],0
            last_nearly_base_frames, last_nearly_index = nearly_base_frames,nearly_index
            last_frame = x
            if nearly_base_frames not in all_nearly_base_frames:
                all_nearly_base_frames.append(nearly_base_frames)
                all_nearly_indexs.append(nearly_index)
        else:
            last_base_frame = last_nearly_base_frames
            nearly_base_frames, nearly_index = get_the_nearly_base_frame(x,base_frames,last_frame,last_base_frame)
            last_nearly_base_frames, last_nearly_index = nearly_base_frames, nearly_index
            last_frame = x
            if nearly_base_frames not in all_nearly_base_frames:
                all_nearly_base_frames.append(nearly_base_frames)
                all_nearly_indexs.append(nearly_index)
    return all_nearly_base_frames,all_nearly_indexs

def add_loss_by_sub_dtw(rms_max_indexs,starts_index, base_frames):
    select_starts_index = starts_index
    all_nearly_base_frames, all_nearly_indexs = get_all_the_nearly_base_frame(starts_index, base_frames)
    for i in range(1,len(all_nearly_indexs)):
        if all_nearly_indexs[i] - all_nearly_indexs[i-1]>1:
            start = starts_index[i-1]
            end = starts_index[i]
            start_base = all_nearly_base_frames[i-1]
            end_base = all_nearly_base_frames[i]
            sub_rms_max_indexs = [x for x in rms_max_indexs if x > start and x < end]
            sub_rms_max_indexs.insert(0,start)
            sub_rms_max_indexs.append(end)
            sub_rms_max_indexs = [x-sub_rms_max_indexs[0] for x in sub_rms_max_indexs if x > sub_rms_max_indexs[0]]
            sub_base_frames = [x for x in base_frames if x >=start_base and x <= end_base]
            sub_base_frames = [x - sub_base_frames[0] for x in sub_base_frames if x > sub_base_frames[0] and x < sub_base_frames[-1]]
            #系数
            rate = (end - start)/(end_base - start_base)
            sub_base_frames = [int(x*rate) for x in sub_base_frames]

            for x in sub_base_frames:
                offset = [np.abs(s - x) for s in sub_rms_max_indexs]
                min_index = offset.index(np.min(offset))
                selected = x + start_base
                select_starts_index.append(selected)
    if all_nearly_indexs[-1] < len(base_frames):
        start = starts_index[-1]
        start_base = all_nearly_base_frames[-1]
        end_base = base_frames[-1]
        sub_rms_max_indexs = [x for x in rms_max_indexs if x > start]
        sub_rms_max_indexs.insert(0, start)
        sub_rms_max_indexs = [x - sub_rms_max_indexs[0] for x in sub_rms_max_indexs if x > sub_rms_max_indexs[0]]
        sub_base_frames = [x for x in base_frames if x >= start_base and x <= end_base]
        sub_base_frames = [x - sub_base_frames[0] for x in sub_base_frames if
                           x > sub_base_frames[0]]
        # 系数
        if end_base != start and end_base != start_base:
            rate = (end_base - start) / (end_base - start_base)
            sub_base_frames = [int(x * rate) for x in sub_base_frames]

        for x in sub_base_frames:
            offset = [np.abs(s - x) for s in sub_rms_max_indexs]
            min_index = offset.index(np.min(offset))
            selected = x + start_base
            select_starts_index.append(selected)

    select_starts_index.sort()
    return select_starts_index

def add_from_maybe_onset_frames(onsets_frames,maybe_onset_frames,CQT,rms,base_frames):
    rms = np.diff(rms, 1)
    rms = [x if x > 0 else 0 for x in rms]

    last_onset = maybe_onset_frames[-1]
    print("== last is :{}".format(last_onset))
    rms, max_indexs = filter_hold_local_max(rms, base_frames)

    if len(max_indexs) < 1:
        onsets_frames, note_lines, times = get_note_lines(CQT, onsets_frames)
        return onsets_frames, note_lines, times

    rms_onset =[i for i in range(1,len(rms)-1) if i < len(rms) -1 and rms[i] > rms[i-1] and rms[i] > rms[i+1] and i < last_onset]
    loss_num = len(base_frames) - len(onsets_frames)

    macth_onset = []
    for x in onsets_frames:
        min_offset = np.min([np.abs(x - o) for o in rms_onset])
        for o in rms_onset:
            if np.abs(x -o) == min_offset:
                macth_onset.append(o)
    select_rms_onset = [x for x in rms_onset if x not in macth_onset]
    select_rms =  [rms[x] for x in select_rms_onset]

    finded_index = find_n_largest(select_rms, loss_num)
    finded_onset = [select_rms_onset[i] for i in finded_index]

    for x in finded_onset:
        onsets_frames.append(x)
    onsets_frames.sort()
    onsets_frames, note_lines, times = get_note_lines(CQT, onsets_frames)
    onsets_frames = [onsets_frames[i] for i in range(len(times)) if times[i] > 3]
    note_lines = [note_lines[i] for i in range(len(times)) if times[i] > 3]
    times = [times[i] for i in range(len(times)) if times[i] > 3]
    return onsets_frames, note_lines, times

def check_with_rms_onset(onsets_frames,note_lines,CQT,rms,last_onset,base_frames):
    rms = np.diff(rms, 1)
    rms = [x if x > 0 else 0 for x in rms]

    #last_onset = maybe_onset_frames[-1]
    print("== last is :{}".format(last_onset))
    rms, max_indexs = filter_hold_local_max(rms, base_frames)

    if len(max_indexs) < 1:
        onsets_frames, note_lines, times = get_note_lines(CQT, onsets_frames)
        return onsets_frames, note_lines, times

    rms_onset =[i for i in range(1,len(rms)-1) if i < len(rms) -1 and rms[i] > rms[i-1] and rms[i] > rms[i+1] and i < last_onset]
    min_rms_on_onsets_frames = np.min([rms[x] for x in onsets_frames])

    result = []
    min_cqt = np.min(CQT)
    max_cqt = np.max(CQT)

    for i in range(1,len(onsets_frames)):
        x = onsets_frames[i]
        row = note_lines[i]
        cols_cqt = CQT[row,x-2:x+2]
        if np.min(cols_cqt) == max_cqt and note_lines[i-1] == note_lines[i]:
            continue
        else:
            result.append(x)

    for x in rms_onset:
        row = note_lines[i]
        cols_cqt = CQT[row, x - 2:x + 2]
        if np.min(cols_cqt) == max_cqt and note_lines[i-1] == note_lines[i]:
            continue
        elif rms[x] > min_rms_on_onsets_frames:
            result.append(x)

    result.sort()
    onsets_frames, note_lines, times = get_note_lines(CQT, result)
    onsets_frames = [onsets_frames[i] for i in range(len(times)) if times[i] > 3]
    note_lines = [note_lines[i] for i in range(len(times)) if times[i] > 3]
    times = [times[i] for i in range(len(times)) if times[i] > 3]
    return onsets_frames, note_lines, times


def filter_hold_local_max(rms,base_frames):
    lenght = len(rms)
    base_frames_diff = list(np.diff(base_frames))
    min_gap = np.min(base_frames_diff)
    tmp = [x for x in base_frames_diff if x>min_gap]
    second_min_gap = np.min(tmp)
    min_gap_first_index = base_frames[base_frames_diff.index(min_gap)+1]
    base_frames_reverse = base_frames_diff.copy()
    base_frames_reverse.reverse()
    min_gap_last_index = base_frames[len(base_frames) - base_frames_reverse.index(min_gap)-1]
    first_rms = rms[:min_gap_first_index]
    second_rms = rms[min_gap_first_index:min_gap_last_index]
    last_rms = rms[min_gap_last_index:]
    rate = 0.5
    f_rms, f_max_indexs = hold_local_max(first_rms, 0, int(second_min_gap*rate/2),int(second_min_gap*rate))
    s_rms, s_max_indexs = hold_local_max(second_rms, 0, int(min_gap*rate/ 2), int(min_gap*rate))
    s_max_indexs = [x + min_gap_first_index for x in s_max_indexs]
    l_rms, l_max_indexs = hold_local_max(last_rms, 0, int(second_min_gap*rate/2),int(second_min_gap*rate))
    l_max_indexs = [x + min_gap_last_index for x in l_max_indexs]
    # if len(f_max_indexs)>0:
    #     f_max_indexs.extend(s_max_indexs).extend(l_max_indexs)
    #     all_indexs = f_max_indexs
    # elif len(s_max_indexs) >0:
    #     s_max_indexs.extend(l_max_indexs)
    #     all_indexs = s_max_indexs
    # else:
    #     all_indexs = l_max_indexs
    all_indexs = f_max_indexs + s_max_indexs + l_max_indexs
    result = np.zeros(lenght)
    for x in f_max_indexs:
        result[x] = rms[x] if rms[x] > 0 else 0
    for x in s_max_indexs:
        result[x] = rms[x] if rms[x] > 0 else 0
    for x in l_max_indexs:
        result[x] = rms[x] if rms[x] > 0 else 0
    #print("======all_indexs is {}".format(all_indexs))
    #print("======base_frames is {}".format(base_frames))
    all_indexs = [all_indexs[i] for i in range(1,len(all_indexs)) if all_indexs[i]-all_indexs[i-1] > 10]
    return result,all_indexs

def hold_local_max(rms,start,step,window_width):
    lenght = len(rms)
    max_indexs = []
    while start < lenght:
        window_start = start
        window_end = start + window_width
        win_rms = rms[window_start:window_end]
        win_rms_max = np.max(win_rms)
        win_rms_list = list(win_rms)
        max_index = win_rms_list.index(win_rms_max)
        if len(max_indexs) == 0:
            max_indexs.append(window_start + max_index)
            old_win_rms_max = win_rms_max
        else:
            if start + max_index - max_indexs[-1] < window_width: #如果跟前一个局部极大值的距离小于窗口宽度，则需要和前一个比较
                if win_rms_max > old_win_rms_max:
                    max_indexs.remove(max_indexs[-1])
                    max_indexs.append(window_start + max_index)
                    old_win_rms_max = win_rms_max
            else: #如果跟前一个局部极大值的距离大于等于窗口宽度，直接添加
                max_indexs.append(window_start + max_index)
                old_win_rms_max = win_rms_max
        start += step
    result = np.zeros(lenght)
    for x in max_indexs:
        result[x] = rms[x]
    return result,max_indexs
'''
数组元素组合
https://blog.csdn.net/suibianshen2012/article/details/80772905
'''
def get_combinations(list1,number):
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
    #print(list2)
    return list2

def del_real_middle_onset_frames(onset_frames,note_lines,times,CQT):
    w,h = CQT.shape
    width = 7
    cqt_min = np.min(CQT)
    cqt_max = np.max(CQT)
    select_onset_frames, select_note_lines, select_times = [],[],[]
    for i in range(len(onset_frames)):
        frame = onset_frames[i]
        if frame <= width:
            select_onset_frames.append(frame)
            select_note_lines.append(note_lines[i])
            select_times.append(times[i])
            continue
        flag = False
        for col in range(10,w-5):
            row_cqt = CQT[col,frame - width:frame + width]
            up_row_cqt = CQT[col+1,frame - width:frame + width]
            if np.min(row_cqt) == cqt_max and np.max(up_row_cqt) == cqt_min:
                flag = True
                break
        if flag:
            continue
        else:
            select_onset_frames.append(frame)
            select_note_lines.append(note_lines[i])
            select_times.append(times[i])
    return select_onset_frames, select_note_lines, select_times