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

def get_note_line_start_v2(cqt,rhythm_codes):
    small_code_indexs = [i for i in range(len(rhythm_codes)) if rhythm_codes[i] == '250']
    threshold_length = 4
    threshold_length_before = 4
    threshold_length_after = 4
    if len(small_code_indexs) < 1:
        threshold_length = 10
    else:
        less_half = [i for i in range(len(small_code_indexs)) if small_code_indexs[i] < int(len(rhythm_codes)/2) ]
        more_half = [i for i in range(len(small_code_indexs)) if small_code_indexs[i] > int(len(rhythm_codes) / 2)]
        if len(less_half) < 1:
            threshold_length_before = 10
        if len(more_half) < 1:
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
    print("best_col is {}".format(best_col))
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
                    if col <= int(h/2):
                        threshold_length = threshold_length_before
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

def merge_note_line_by_distance(result,times,note_lines):
    select_result = []
    select_times = []
    select_note_lines = []
    select_result.append(result[0])
    select_times.append(times[0])
    select_note_lines.append(note_lines[0])

    for i in range(1,len(result)):
        if result[i] - result[i-1] > 9 and times[i] > 5:
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
                    if keyMap.get(onset_frames[i]) > 1.75:
                        select_onset_frames.append(onset_frames[i])
                        select_notes_lines.append(notes_lines[i])
                        select_longest_numbers.append(longest_numbers[i])
                    else:
                        select_longest_numbers[-1] = select_longest_numbers[-1] + longest_numbers[i]
    else:
        return onset_frames, notes_lines, longest_numbers

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
        note_lines.append(note_line)
        longest_numbers.append(longest_num)
        selected_result.append(x)

    return selected_result,note_lines,longest_numbers

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
            print("x,j is {},{}".format(x,j))
            break
    return note_line

def check_all_note_lines(onset_frames,all_note_lines,cqt):
    note_lines_median = np.median(all_note_lines)
    print("note_lines_median is {}".format(note_lines_median))
    selected_note_lines = []
    for i in range(len(all_note_lines)):
        x = all_note_lines[i]
        onset_frame = onset_frames[i]
        if np.abs(x - note_lines_median)>10:
            low_start = x + 1
            note_line = modify_some_note_line(cqt,onset_frame,low_start)
            if np.abs(note_line - note_lines_median)>10:
                note_line = int(note_lines_median)
            selected_note_lines.append(note_line)
        else:
            selected_note_lines.append(x)
    return selected_note_lines

def del_false_note_lines(onset_frames,all_note_lines,rms,CQT):
    select_note_lines = []
    select_note_lines.append(all_note_lines[0])
    select_onset_frames = []
    select_onset_frames.append(onset_frames[0])
    print("max rms is {}".format(np.max(rms)))
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

def del_end_range(onset_frames,all_note_lines,times,rms):
    length = len(rms)
    select_onset_frames = []
    select_all_note_lines = []
    select_times = []
    for i in range(len(onset_frames)):
        onset_frame = onset_frames[i]
        note_line = all_note_lines[i]
        t = times[i]
        if onset_frame < length - 40:
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
    print("diff_longest_note_str, diff_base_notes_str,intersect is {}==={}==={}".format(diff_longest_note_str, diff_base_notes_str,list_intersect))
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

def change_rms(onsets_frames,rms,topN):
    result = []
    keyMap = {}
    indexMap = {}
    select_onset_frames = onsets_frames.copy()
    for i in range(1,len(rms)-3):
        if (i==1 and rms[2] > rms [1]) or (rms[i+1] > rms[i] and rms[i-1] > rms[i]):
            hightest_point_after = find_hightest_after(i, rms)
            if rms[hightest_point_after] - rms[i] > 0.3:
                print("rms[hightest_point_after] - rms[i] is {}".format(rms[hightest_point_after] - rms[i] ))
                value = rms[hightest_point_after] - rms[i]
                result.append(value)
                keyMap[value] = i
                indexMap[i] = value
    topN_index = find_n_largest(result,topN)
    topN_key = [result[i] for i in topN_index]
    for key in topN_key:
        index = keyMap.get(key)
        offset = [np.abs(index - x) for x in select_onset_frames]
        if index > onsets_frames[0] and np.min(offset) > 9:
            select_onset_frames.append(index)
    select_onset_frames.sort()
    return select_onset_frames,indexMap
def cal_note_score(longest_note,base_notes):
    off = int(np.mean(base_notes) - np.mean(longest_note))
    # off = int((base_notes[0] - longest_note[0]))
    base_notes = [x - off for x in base_notes]
    print("base_notes is {}".format(base_notes))
    euclidean_norm = lambda x, y: np.abs(x - y)
    if (len(longest_note) != len(base_notes)):
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
    return notes_score
def draw_plt(filename):
    y, sr = load_and_trim(filename)
    y, sr = librosa.load(filename)
    rms = librosa.feature.rmse(y=y)[0]
    rms = [x / np.std(rms) for x in rms]
    time = librosa.get_duration(filename=filename)
    print("time is {}".format(time))
    CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref=np.max)
    w, h = CQT.shape
    print("w.h is {},{}".format(w, h))
    # onsets_frames = get_real_onsets_frames_rhythm(y)
    base_frames = onsets_base_frames_for_note(filename)
    base_notes = base_note(filename)
    CQT = np.where(CQT > -22, np.max(CQT), np.min(CQT))
    # onsets_frames,end_position = get_note_line_start(CQT)
    rhythm_codes = get_rhythm_codes(filename)
    onsets_frames, end_result, times, note_lines = get_note_line_start_v2(CQT,rhythm_codes)
    #onsets_frames, end_result, times, note_lines = merge_note_line(onsets_frames, end_result, times, note_lines)
    #onsets_frames = find_loss_by_rms_mean(onsets_frames, rms, CQT)
    onsets_frames,keyMap = change_rms(onsets_frames,rms,len(base_frames))
    onsets_frames, note_lines, times = get_note_lines(CQT, onsets_frames)
    onsets_frames, note_lines, times = del_false_same(base_notes, onsets_frames, note_lines, times,keyMap)
    onsets_frames, note_lines,times = del_end_range(onsets_frames, note_lines,times, rms)
    onsets_frames, times, note_lines = merge_note_line_by_distance(onsets_frames, times, note_lines)

    print("0 onsets_frames is {},size is {}".format(onsets_frames,len(onsets_frames)))
    print("0 end_result is {},size is {}".format(end_result,len(end_result)))
    print("0 times is {},size is {}".format(times,len(times)))
    print("0 note_lines is {},size is {}".format(note_lines,len(note_lines)))
    list_intersect_before, list_intersect, list_intersect_after = check_all_notes_trend(note_lines, base_notes)
    print("list_intersect_before,list_intersect,list_intersect_after is {},{},{}".format(list_intersect_before,
                                                                                         list_intersect,
                                                                                         list_intersect_after))

    type_index = get_onsets_index_by_filename_rhythm(filename)
    onset_score, lost_score, ex_score, min_d = get_score_for_note(onsets_frames.copy(), base_frames.copy(), type_index)
    print("score, lost_score, ex_score, min_d is {},{},{},{}".format(onset_score, lost_score, ex_score, min_d))
    #onset_score = cal_dtw_distance(onsets_frames.copy(), base_frames.copy())
    note_scroe = cal_note_score(note_lines, base_notes)
    onset_score = int(onset_score*0.4)
    score = onset_score + note_scroe
    print("score, onset_score, note_scroe is {},{},{}".format(score, onset_score, note_scroe ))

    plt.subplot(3, 1, 1)
    # librosa.display.specshow(CQT)
    librosa.display.specshow(CQT, y_axis='cqt_note', x_axis='time')
    print(np.max(y))
    onstm = librosa.frames_to_time(onsets_frames, sr=sr)
    end_time = librosa.frames_to_time(end_result, sr=sr)
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
    # rms_on_onset_frames_cqt = [rms[x] for x in onset_frames_cqt]
    # min_rms_on_onset_frames_cqt = np.min(rms_on_onset_frames_cqt)
    # rms = [1 if x >=min_rms_on_onset_frames_cqt else 0 for x in rms]
    plt.vlines(onstm, 0, np.max(rms), color='y', linestyle='dashed')
    plt.vlines(base_times, 0, np.max(rms), color='r', linestyle='dashed')
    rms_on_frames = [rms[x] for x in onsets_frames]
    mean_rms_on_frames = np.mean(rms_on_frames)
    plt.plot(times, rms)
    plt.xlim(0, np.max(times))
    plt.axhline(mean_rms_on_frames, color='r')

    plt.subplot(3, 1, 3)
    rms = np.diff(rms,2)


    times = librosa.frames_to_time(np.arange(len(rms)))
    # rms_on_onset_frames_cqt = [rms[x] for x in onset_frames_cqt]
    # min_rms_on_onset_frames_cqt = np.min(rms_on_onset_frames_cqt)
    # rms = [1 if x >=min_rms_on_onset_frames_cqt else 0 for x in rms]
    rms_on_frames = [rms[x] for x in onsets_frames]
    mean_rms_on_frames = np.mean(rms_on_frames)
    plt.plot(times, rms)
    plt.axhline(mean_rms_on_frames, color='r')
    plt.vlines(onstm, 0, np.max(rms), color='y', linestyle='dashed')
    plt.xlim(0, np.max(times))
    return plt,score,onset_score, note_scroe


if __name__ == "__main__":
    #y, sr = load_and_trim('F:/项目/花城音乐项目/样式数据/ALL/旋律/1.31MP3/旋律1.100分.wav')
    filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律2.1(80).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/ALL/旋律/1.31MP3/旋律3.100分.wav'
    #filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律一（9）（100）.wav'
    #filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律一（14）（95）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋律五（3）（63）.wav'
    #filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏一（4）（96）.wav'
    #filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋3王（80）.wav'
    #filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋4谭（95）.wav'
    #filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋4文(75).wav'
    #filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋8录音1(80).wav'
    #filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1.3(93).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋3罗（80）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋律1_40312（95）.wav'
    filename = 'e:/test_image/m1/A/旋律1_40312（95）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋3罗（80）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋律十（1）（90）.wav'

    plt.close()
    plt, score,onset_score, note_scroe = draw_plt(filename)
    plt.show()
    plt.clf()


    dir_list = ['F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/']
    #dir_list = ['F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/']
    #dir_list = ['e:/test_image/m1/D/']
    #dir_list = []
    total_accuracy = 0
    total_num = 0
    result_path = 'e:/test_image/n/'
    # clear_dir(result_path)
    # 要测试的数量
    test_num = 100
    score = 0
    file_total = 0
    total_10 = 0
    total_15 = 0
    total_20 = 0
    for dir in dir_list:
        file_list = os.listdir(dir)
        # shuffle(file_list)  # 将语音文件随机排列
        # file_list = ['视唱1-01（95）.wav']
        for filename in file_list:
            # clear_dir(image_dir)
            # wavname = re.findall(pattern,filename)[0]
            print(dir + filename)
            # plt = draw_start_end_time(dir + filename)
            # plt = draw_baseline_and_note_on_cqt(dir + filename, False)
            plt, total_score,onset_score, note_scroe = draw_plt(dir + filename)
            # tmp = os.listdir(result_path)

            if filename.find("tune") > 0 or filename.find("add") > 0 or filename.find("shift") > 0:
                score = re.sub("\D", "", filename.split("-")[0])  # 筛选数字
            else:
                score = re.sub("\D", "", filename)  # 筛选数字

            if str(score).find("100") > 0:
                score = 100
            else:
                score = int(score) % 100

            if int(score) >= 90:
                grade = 'A'
            elif int(score) >= 75:
                grade = 'B'
            elif int(score) >= 60:
                grade = 'C'
            elif int(score) >= 1:
                grade = 'D'
            else:
                grade = 'E'
            # result_path = result_path + grade + "/"
            # plt.savefig(result_path + filename + '.jpg', bbox_inches='tight', pad_inches=0)
            #grade = 'A'
            #plt.savefig(result_path + grade + "/" + filename + '-'+ str(total_score) + '-' + str(onset_score) + '-' + str(note_scroe)  + '.jpg', bbox_inches='tight', pad_inches=0)
            if np.abs(total_score - int(score)) > 15:
                plt.savefig( result_path + grade + "/" + filename + '-' + str(total_score) + '-' + str(onset_score) + '-' + str(note_scroe) + '.jpg', bbox_inches='tight', pad_inches=0)
            plt.clf()

            if np.abs(total_score - int(score)) <= 10:
                total_10 += 1
            if np.abs(total_score - int(score)) <= 15:
                total_15 += 1
            if np.abs(total_score - int(score)) <= 20:
                total_20 += 1
    print("file_total,yes_total is {},{},{},{},{}".format(file_total, total_10, total_15, total_20,
                                                          total_10 / file_total))