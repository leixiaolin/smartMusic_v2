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

filename = 'F:/项目/花城音乐项目/样式数据/ALL/旋律/1.31MP3/旋律1.100分.wav'
#y, sr = load_and_trim('F:/项目/花城音乐项目/样式数据/ALL/旋律/1.31MP3/旋律1.100分.wav')
y, sr = load_and_trim(filename)
hop_length = 1048
#chromagram = librosa.feature.chroma_cqt(y, sr=sr, hop_length=hop_length)
chromagram = librosa.feature.chroma_cqt(y, sr=sr)
# chromagram[11,:] = 1
plt.figure(figsize=(15, 5))

# 原始音色图
# librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', cmap='coolwarm')
# plt.colorbar()
# plt.show()

c_max = np.argmax(chromagram,axis=0)
print(c_max.shape[0])
print(c_max)
c_max_diff = np.diff(c_max) # 一阶差分
print(np.diff(c_max))
print(c_max_diff.shape)

img = np.zeros(chromagram.shape,dtype=np.float32)
w,h = chromagram.shape
for x in range(len(c_max_diff)):
    #img.item(x, c_max[x], 0)
    if x > 0 and (c_max_diff[x] == 1 or c_max_diff[x] == -1):
        c_max[x] = c_max[x-1]

for x in range(h):
    #img.item(x, c_max[x], 0)
    img.itemset((c_max[x],x), 0.5)
    img.itemset((c_max[x],x), 0.5)
    img.itemset((c_max[x],x), 0.5)

# 最强音色图
# librosa.display.specshow(img, x_axis='time',  cmap='coolwarm')

# 音频时长
time = librosa.get_duration(y)
print("time is {}".format(time))
# 节拍点
onsets_frames = librosa.onset.onset_detect(y)
print(onsets_frames)

# 节拍时间点
onstm = librosa.frames_to_time(onsets_frames, sr=sr)
print(onstm)
#plt.rcParams['figure.figsize'] = (2.0, 2.0) # 设置figure_size尺寸
#plt.rcParams['savefig.dpi'] = 28 #图片像素
#plt.rcParams['figure.dpi'] = 28 #分辨率
#librosa.display.specshow(librosa.amplitude_to_db(D))
#plt.vlines(onstm, 0, sr, color='r', linestyle='dashed')
#plt.colorbar()

code = '[500,500,1000;500,500,1000;500,500,750,250;2000]'
pitch_code = '[3,3,3,3,3,3,3,5,1,2,3]'
pitch_v = get_chroma_pitch(pitch_code)
onsets_base_frames = onsets_base_frames(code,h)
onsets_base_frames[-1] = onsets_base_frames[-1]-1
print(onsets_base_frames)
print(np.diff(onsets_base_frames))
print(pitch_v)
v0 = 0
for i,v in enumerate(onsets_base_frames[1:]):
    print("v0,v is {},{},{}".format(v0,v,pitch_v[i]))
    if v == 337:
        print('=====')
    for f in range(v0,v):
        if img.item(pitch_v[i],f) == 0.5:
            img.itemset((pitch_v[i],f), 1)
            img.itemset((pitch_v[i],f), 1)
            img.itemset((pitch_v[i],f), 1)
        else:
            img.itemset((pitch_v[i], f), 0.8)
            img.itemset((pitch_v[i], f), 0.8)
            img.itemset((pitch_v[i], f), 0.8)
    v0 = v

start_point = 0.02
ds = onsets_base(code,time,start_point)
print("ds is {}".format(ds))
plt.vlines(ds, 0, sr, color='b', linestyle='solid')
librosa.display.specshow(img, x_axis='time',  cmap='coolwarm')
#librosa.display.specshow(chromagram, x_axis='time',  cmap='coolwarm')
plt.show()
