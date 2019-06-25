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



def get_frame_length(cqt_col_diff):
    for i in range(2,len(cqt_col_diff)):
        if np.max(cqt_col_diff[:i-1]) == 0 and cqt_col_diff[i] >0.1:
            start = i

    for i in range(len(cqt_col_diff)-2,0,-1):
        if np.max(cqt_col_diff[i+1:]) == 0 and cqt_col_diff[i] >0.1:
            end = i
    return start,end,end-start

#y, sr = load_and_trim('F:/项目/花城音乐项目/样式数据/ALL/旋律/1.31MP3/旋律1.100分.wav')
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律2.1(80).wav'
filename = 'F:/项目/花城音乐项目/样式数据/ALL/旋律/1.31MP3/旋律3.100分.wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律一（9）（100）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律一（14）（95）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/节奏/节2罗（75）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏一（4）（96）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋2熙(0).wav'
#filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1罗（96）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋10罗（92）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋2录音4(72).wav'
#filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋10罗（92）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律一（13）（98）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1熙(90).wav'
#filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1谭（98）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋3.3(96).wav'
filename = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/两只老虎20190624-2939.wav'


rhythm_code = '[500,500,500,500;500,500,500,500;500,500,1000;500,500;1000]'
pitch_code = '[1,2,3,1,1,2,3,1,3,4,5,3,4,5]'

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
result = []
for i in range(1,h):
    col_cqt = CQT[10:,i]
    before_col_cqt = CQT[10:,i-1]
    max_sum = np.sum([1 if x > np.min(col_cqt) else 0 for x in col_cqt])
    before_max_sum = np.sum([1 if x > np.min(before_col_cqt) else 0 for x in before_col_cqt])
    #sum = np.sum(np.array(col_cqt) - np.array(before_col_cqt))
    sum = np.sum([1 if col_cqt[i] != before_col_cqt[i] and max_sum > 0.4*before_max_sum else 0 for i in range(len(col_cqt))])
    result.append(sum)

result = [x if x>0 else 0 for x in result]
result = [x/np.max(result) for x in result]
result = [x if x>0.1 else 0 for x in result]

times = range(len(result))
plt.subplot(4,1,1)
librosa.display.specshow(CQT ,x_axis='time')

plt.subplot(4,1,2)
plt.plot(times, result)
plt.xlim(0, np.max(times))
start,end,length = get_frame_length(result)
plt.vlines(start, 0, np.max(result), color='r', linestyle='dashed')
plt.vlines(end, 0, np.max(result), color='r', linestyle='dashed')
base_frames = onsets_base_frames_rhythm(rhythm_code, length)
base_frames = [x - (base_frames[0]-start) for x in base_frames]
plt.vlines(base_frames, 0, np.max(result), color='b', linestyle='dashed')

plt.subplot(4,1,3)
rms = np.diff(rms, 1)
rms = [x if x > 0 else 0 for x in rms]
rms = [x/np.max(rms) for x in rms]
plt.plot(times, rms)
plt.xlim(0, np.max(times))

plt.subplot(4,1,4)
together = [result[i] if result[i] > rms[i] else rms[i] for i in range(len(result))]
plt.plot(times, together)
plt.xlim(0, np.max(times))
plt.show()