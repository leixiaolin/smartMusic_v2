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
    for i in range(len(result)):
        x = result[i]
        sub_cqt = cqt[:,x:x+15]
        # 从下往上逐行判断
        for j in range(10,w-10):
            row_cqt = sub_cqt[j]
            #如果存在连续的亮点，长度大于8
            max_acount = np.sum(row_cqt == cqt_max)
            min_acount = np.sum(row_cqt == cqt_min)
            if max_acount > min_acount and np.sum(sub_cqt[j-1] == cqt_min) > min_acount:
                note_lines.append(j)
                selected_result.append(x)
                #print("x,j is {},{}".format(x,j))
                break
        if j == w -11:
            if len(note_lines)>0:
                note_lines.append(note_lines[-1])
                selected_result.append(x)
            else:
                # 从下往上逐行判断
                for j in range(10, w - 10):
                    row_cqt = sub_cqt[j]
                    max_acount = np.sum(row_cqt == cqt_max)
                    if max_acount > 5:
                        note_lines.append(j)
                        selected_result.append(x)
                        # print("x,j is {},{}".format(x,j))
                        break

    return selected_result,note_lines

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
            elif np.abs(rms[current_onset+1] - rms[current_onset-1]) > 0.3:
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

#y, sr = load_and_trim('F:/项目/花城音乐项目/样式数据/ALL/旋律/1.31MP3/旋律1.100分.wav')
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律2.1(80).wav'
filename = 'F:/项目/花城音乐项目/样式数据/ALL/旋律/1.31MP3/旋律3.100分.wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律一（9）（100）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律一（14）（95）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/节奏/节2罗（75）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏一（4）（96）.wav'
filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋2熙(0).wav'
filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1罗（96）.wav'
filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋10罗（92）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋2录音4(72).wav'
filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋10罗（92）.wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律一（13）（98）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1熙(90).wav'
filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1谭（98）.wav'


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
onsets_frames,end_position = get_note_line_start(CQT)
onsets_frames = find_loss_by_rms_mean(onsets_frames,rms,CQT)
print("onsets_frames is {}".format(onsets_frames))
onsets_frames,all_note_lines = get_note_lines(CQT,onsets_frames)
all_note_lines = check_all_note_lines(onsets_frames,all_note_lines,CQT)
print("all_note_lines is {}".format(all_note_lines))
onsets_frames, all_note_lines = del_false_note_lines(onsets_frames,all_note_lines,rms,CQT)
onsets_frames, all_note_lines = del_end_range(onsets_frames,all_note_lines,rms)
#CQT[:,onsets_frames[1]:h] = -100
total_frames_number = get_total_frames_number(filename)
print("total_frames_number is {}".format(total_frames_number))
plt.subplot(3,1,1)
#librosa.display.specshow(CQT)
librosa.display.specshow(CQT, y_axis='cqt_note',x_axis='time')
print(np.max(y))
onstm = librosa.frames_to_time(onsets_frames, sr=sr)
end_position_time = librosa.frames_to_time(end_position, sr=sr)
plt.vlines(onstm, 0,sr, color='y', linestyle='dashed')
#plt.vlines(end_position_time, 0,sr, color='r', linestyle='solid')
# plt.colorbar(format='%+2.0f dB')
# plt.title('Constant-Q power spectrogram (note)')
plt.subplot(3,1,2)
times = librosa.frames_to_time(np.arange(len(rms)))
# rms_on_onset_frames_cqt = [rms[x] for x in onset_frames_cqt]
# min_rms_on_onset_frames_cqt = np.min(rms_on_onset_frames_cqt)
# rms = [1 if x >=min_rms_on_onset_frames_cqt else 0 for x in rms]
rms_on_frames = [rms[x] for x in onsets_frames]
mean_rms_on_frames = np.mean(rms_on_frames)
plt.plot(times, rms)
plt.axhline(mean_rms_on_frames, color='r')

plt.subplot(3,1,3)
rms = np.diff(rms)
times = librosa.frames_to_time(np.arange(len(rms)))
# rms_on_onset_frames_cqt = [rms[x] for x in onset_frames_cqt]
# min_rms_on_onset_frames_cqt = np.min(rms_on_onset_frames_cqt)
# rms = [1 if x >=min_rms_on_onset_frames_cqt else 0 for x in rms]
rms_on_frames = [rms[x] for x in onsets_frames]
mean_rms_on_frames = np.mean(rms_on_frames)
plt.plot(times, rms)
plt.axhline(mean_rms_on_frames, color='r')
plt.show()