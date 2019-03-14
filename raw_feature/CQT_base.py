import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
from create_base import *
from myDtw import *
from find_mismatch import *

filepath = 'F:\项目\花城音乐项目\样式数据\音乐样本2019-01-29\节奏九\\'
# filename = 'F:/项目/花城音乐项目/样式数据/ALL/节奏/节奏八/节奏八（标准音频）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/ALL/节奏/节奏八/节奏8.100分.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/ALL/旋律/1.31MP3/旋律1.100分.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/ALL/旋律/1.31MP3/旋律2.100分.wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏4卢(65).wav'
#filename = 'F:/项目/花城音乐项目/样式数据/ALL/节奏/节奏八/节奏八（1）(90).wav'
#filename = 'F:/项目/花城音乐项目/样式数据/ALL/节奏/节奏八/节奏八（2）（90分）.wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏8_40434（30）.wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/视唱1-01（95）.wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/视唱1-02（90）.wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律1.1(95).wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律2（四）(96).wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律1.1(95).wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律2.1(80).wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律2.3(55).wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律二（10）（75）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律二（8）（100）.wav'

# 2. Load the audio as a waveform `y`
#    Store the sampling rate as `sr`

codes = np.array(['[1000,1000;2000;1000,500,500;2000]',
                  '[2000;1000,1000;500,500,1000;2000]',
                  '[1000,1000;500,500,1000;1000,1000;2000]',
                  '[1000,--(1000);1000,--(1000);500,250,250,1000;--(1000),1000]',
                  '[500;1000,500,1000,500;500,500,500,250,250,500,500;250,250,500,500,1000]',
                  '[1000,--(1000);1000,--(1000);1000,-(500),500;1000,1000]',
                  '[750,250,500,500,500,-(500);500,1000,500,500,-(500);750,250,500,500,500,-(500)]',
                  '[500,1000,500,500,250,250;1000,500,750,250,500;3000]',
                  '[500,500,500;1000,500;500,500,500;1500;500,500,500;1000,500;500;1000;1500]',
                  '[500,500,1000;500,500;1000;375,125,250,250,375,125,250,250;500,500,1000]',
                  '[500,500,1000;500,500,1000;500,500,750,250;2000]',
                  '[1000,1000;500,500,1000;1000,500,500;2000]',
                  '[1000,1000;500,500,1000;500,250,250,250;2000]',
                  '[500,1000,500;250,250,250,250,500,500;500,500,500,500;2000]'])
# 1. Get the file path to the included audio example
# Sonify detected beat events
# 定义加载语音文件并去掉两端静音的函数
def load_and_trim(path):
    audio, sr = librosa.load(path)
    energy = librosa.feature.rmse(audio)
    frames = np.nonzero(energy >= np.max(energy) / 10)
    indices = librosa.core.frames_to_samples(frames)[1]
    audio = audio[indices[0]:indices[-1]] if indices.size else audio[0:0]

    return audio, sr

def get_max_strength(chromagram):
    c_max = np.argmax(chromagram, axis=0)
    #print(c_max.shape[0])
    #print(c_max)
   # print(np.diff(c_max))
    # chromagram_diff = np.diff(chromagram,axis=0)
    # print(chromagram_diff)
    # sum_chromagram_diff = chromagram_diff.sum(axis=0)
    # test = np.array(sum_chromagram_diff)
    # plt.plot(test)

    img = np.zeros(chromagram.shape, dtype=np.float32)
    w, h = chromagram.shape
    for x in range(h):
        # img.item(x, c_max[x], 0)
        img.itemset((c_max[x], x), 1)
    return img

#y, sr = load_and_trim('F:/项目/花城音乐项目/样式数据/ALL/旋律/1.31MP3/旋律1.100分.wav')
y, sr = load_and_trim(filename)

CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref = np.max)
w,h = CQT.shape
CQT[40:w,:] = -100
CQT[0:20,:] = -100
#librosa.display.specshow(CQT)
plt.figure(figsize=(10, 10))
plt.subplot(4,1,1) #要生成两行两列，这是第一个图plt.subplot('行','列','编号')
# plt.colorbar(format='%+2.0f dB')
# plt.title('Constant-Q power spectrogram (note)')
librosa.display.specshow(CQT, y_axis='cqt_note',x_axis='time')
onsets_frames =  librosa.onset.onset_detect(y)
onsets_frames = get_real_onsets_frames_rhythm(y)
onsets_frames = get_onsets_frames_by_cqt_for_rhythm(y,sr)
# 标准节拍时间点
type_index = get_onsets_index_by_filename_rhythm(filename)
total_frames_number = get_total_frames_number(filename)
base_frames = onsets_base_frames_rhythm(type_index,total_frames_number)
base_onsets = librosa.frames_to_time(base_frames, sr=sr)
print(np.max(y))
onstm = librosa.frames_to_time(onsets_frames, sr=sr)
plt.vlines(base_onsets, 0,sr, color='y', linestyle='solid')
print(CQT.shape)
q1,q2 = CQT.shape
print(plt.figure)

plt.subplot(4,1,2) #要生成两行两列，这是第一个图plt.subplot('行','列','编号')
librosa.display.waveplot(y, sr=sr)
plt.vlines(base_onsets, -1*np.max(y),np.max(y), color='y', linestyle='solid')


# duration = librosa.get_duration(filename=filename)
# # 标准节拍时间点
# base_onsets = onsets_base(codes[11], duration, onstm[0])
# plt.vlines(base_onsets[:-1], -1*np.max(y),np.max(y), color='r', linestyle='dashed')
# plt.vlines(base_onsets[-1], -1*np.max(y),np.max(y), color='white', linestyle='dashed')
plt.subplot(4,1,3)
rms = librosa.feature.rmse(y=y)[0]
rms = rms/ np.std(rms)
rms_diff = np.diff(rms)
print("rms_diff is {}".format(rms_diff))
times = librosa.frames_to_time(np.arange(len(rms)))
plt.plot(times, rms)
#plt.axhline(0.02, color='r', alpha=0.5)
plt.xlabel('Time')
plt.ylabel('RMS')
plt.axis('tight')
print("rms max is {}".format(np.max(rms)))
want_all_points = get_all_onsets_starts(rms)
#want_all_points = get_all_onsets_ends(rms)
#want_all_points = [x for i,x in enumerate(all_points) if i < len(all_points)-1 and (peak_trough_rms_diff[i]>1)]
print("want_all_points is {}".format(want_all_points))
want_all_points_time = librosa.frames_to_time(want_all_points)
plt.vlines(want_all_points_time, 0,np.max(rms), color='y', linestyle='solid')
# 标准节拍时间点
base_frames = onsets_base_frames_rhythm(type_index,total_frames_number)
print("base_frames is {}".format(base_frames))

min_d, best_y, onsets_frames = get_dtw_min(onsets_frames, base_frames, 65,move=False)
base_onsets = librosa.frames_to_time(best_y, sr=sr)
plt.vlines(base_onsets,  0,np.max(rms), color='r', linestyle='dashed')
# 找出漏唱的线的帧
standard_y = want_all_points
recognize_y = best_y
miss_onsets = get_mismatch_line(standard_y,recognize_y)
miss_onsets_time = librosa.frames_to_time(miss_onsets[1], sr=sr)
plt.vlines(miss_onsets_time,  0,np.max(rms), color='black', linestyle='dashed')


plt.subplot(4,1,4)
chromagram = librosa.feature.chroma_cqt(y, sr=sr)

c_max = np.argmax(chromagram,axis=0)
c_max_diff = np.diff(c_max) # 一阶差分
img = np.zeros(chromagram.shape,dtype=np.float32)
w,h = chromagram.shape
for x in range(len(c_max_diff)):
    #img.item(x, c_max[x], 0)
    if x > 0 and (c_max_diff[x] == 1 or c_max_diff[x] == -1):
        c_max[x] = c_max[x-1]

for x in range(h):
    #img.item(x, c_max[x], 0)
    img.itemset((c_max[x],x), 1)
    img.itemset((c_max[x],x), 1)
    img.itemset((c_max[x],x), 1)
# 最强音色图
img  = get_max_strength(CQT)
librosa.display.specshow(img, x_axis='time', y_axis='chroma', cmap='coolwarm')
plt.vlines(base_onsets, 0, sr, color='y', linestyle='solid')
plt.show()