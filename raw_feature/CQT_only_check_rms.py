import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
from create_base import *
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

def get_note_line_start_v2(cqt):
    min_cqt = np.min(cqt)
    max_cqt = np.max(cqt)
    w,h = cqt.shape
    result = []
    best_longest, best_begin, best_row = 0, 0, 0
    offset = 15
    end = 0
    for col in range(10,h-10):
        col_cqt = cqt[10:, col] # 列向量
        # 存在亮点
        if np.max(col_cqt) == max_cqt and col > end:
            flag = True
            while flag:
                cols_cqt = cqt[10:, col:col+offset] #柜向量
                best_longest, best_begin, best_row = get_longest_for_cols_cqt(cols_cqt,min_cqt)
                if best_longest == offset:
                    offset += 15
                else:
                    flag = False
                    result.append(col + best_begin)
                    end = col + best_begin + best_longest
    return result

def get_longest_for_cols_cqt(cols_cqt,min_cqt):
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
        if longest_num > 5 and longest_num < 30:
            note_lines.append(note_line)
            longest_numbers.append(longest_num)
            selected_result.append(x)
        elif longest_num >= 30:
            #如果音高线以上有没有可分点
            best_other_onset_frames, best_note_numbers = find_other_note_line(x,note_line+4,sub_cqt)
            if len(best_other_onset_frames) > 1:
                for i in range(len(best_other_onset_frames)):
                    note_lines.append(note_line)
                    longest_numbers.append(best_note_numbers[i])
                    selected_result.append(best_other_onset_frames[i])
            else:
                # 一分为二
                note_lines.append(note_line)
                longest_numbers.append(int(longest_num/2))
                selected_result.append(x)

                note_lines.append(note_line)
                longest_numbers.append(int(longest_num/2))
                selected_result.append(x + int(longest_num/2))


        # # 从下往上逐行判断
        # for j in range(10,w-10):
        #     row_cqt = sub_cqt[j]
        #     #如果存在连续的亮点，长度大于8
        #     max_acount = np.sum(row_cqt == cqt_max)
        #     min_acount = np.sum(row_cqt == cqt_min)
        #     if max_acount > min_acount and np.sum(sub_cqt[j-1] == cqt_min) > min_acount:
        #         note_lines.append(j)
        #         selected_result.append(x)
        #         #print("x,j is {},{}".format(x,j))
        #         break
        # if j == w -11:
        #     if len(note_lines)>0:
        #         note_lines.append(note_lines[-1])
        #         selected_result.append(x)
        #     else:
        #         # 从下往上逐行判断
        #         for j in range(10, w - 10):
        #             row_cqt = sub_cqt[j]
        #             max_acount = np.sum(row_cqt == cqt_max)
        #             if max_acount > 5:
        #                 note_lines.append(j)
        #                 selected_result.append(x)
        #                 # print("x,j is {},{}".format(x,j))
        #                 break

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
        start = i - 1
        end = i + 2
        # 条件一：振幅有增加
        sub_rms = [rms[start + 1] - rms[start],rms[start + 2] - rms[start],rms[start + 3] - rms[start]]
        cond1 = np.max(sub_rms) > 0.1

        # 条件一：跨过均值线
        #cond2 = (rms[i] <= mean_rms_on_frames and rms[i+1]>mean_rms_on_frames) or (rms[i-1] <= mean_rms_on_frames and rms[i]>mean_rms_on_frames)
        cond2 = rms_diff[i] > 0.3

        if cond2 and min_off > 10:
            #print("np.std(sub_rms) is {}".format(np.std(sub_rms)))
            print("np.max(sub_rms) is {}".format(np.max(sub_rms)))
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

def del_end_range(onset_frames,all_note_lines,rms):
    length = len(rms)
    select_onset_frames = []
    select_all_note_lines = []
    for i in range(len(onset_frames)):
        onset_frame = onset_frames[i]
        note_line = all_note_lines[i]
        if onset_frame < length - 40:
            select_onset_frames.append(onset_frame)
            select_all_note_lines.append(note_line)
    return select_onset_frames,select_all_note_lines

# 如果音高线以上有没有可分点
def find_other_note_line(onset_frame,note_line,sub_cqt):
    w,h = sub_cqt.shape
    print("w,h is {},{}".format(w,h))
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
    print("w,h is {},{}".format(w,h))

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

def del_false_same(base_notes,onset_frames,notes_lines):
    select_onset_frames = [onset_frames[0]]
    select_notes_lines = [notes_lines[0]]
    same_index = get_the_index_of_same(base_notes)
    if len(same_index) == 0:
        for i in range(1,len(notes_lines)):
            if notes_lines[i] != notes_lines[i-1]:
                select_onset_frames.append(onset_frames[i])
                select_notes_lines.append(notes_lines[i])
    else:
        for i in range(1, len(notes_lines)):
            if notes_lines[i] != notes_lines[i - 1]:
                select_onset_frames.append(onset_frames[i])
                select_notes_lines.append(notes_lines[i])
            else:
                # print("i ,longest_numbers[i-1] < onset_frames[i] - onset_frames[i-1] is {},{},{}".format(i,longest_numbers[i-1] ,onset_frames[i] - onset_frames[i-1]))
                if (i in same_index and longest_numbers[i - 1] < onset_frames[i] - onset_frames[i - 1]) or \
                        longest_numbers[i - 1] < onset_frames[i] - onset_frames[i - 1] - 2:
                    select_onset_frames.append(onset_frames[i])
                    select_notes_lines.append(notes_lines[i])
    return select_onset_frames,select_notes_lines
def get_the_index_of_same(base_notes):
    same_index = []
    for i in range(1, len(base_notes)):
        if base_notes[i] == base_notes[i - 1]:
            same_index.append(i)
    return same_index

def get_loss_at_begin(cqt,result,all_note_lines,longest_numbers):
    min_cqt = np.min(cqt)
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



#y, sr = load_and_trim('F:/项目/花城音乐项目/样式数据/ALL/旋律/1.31MP3/旋律1.100分.wav')
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律2.1(80).wav'
filename = 'F:/项目/花城音乐项目/样式数据/ALL/旋律/1.31MP3/旋律3.100分.wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律一（9）（100）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律一（14）（95）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/节奏/节2罗（75）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏一（4）（96）.wav'
filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋2熙(0).wav'
filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1罗（96）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋10罗（92）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋2录音4(72).wav'
#filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋10罗（92）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律一（13）（98）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1熙(90).wav'
#filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1谭（98）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋3.3(96).wav'
filename = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/两只老虎20190624-2939.wav'


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

librosa.display.specshow(CQT ,x_axis='time')

plt.show()