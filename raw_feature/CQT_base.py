import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
from create_base import *

filepath = 'F:\项目\花城音乐项目\样式数据\音乐样本2019-01-29\节奏九\\'
# filename = 'F:/项目/花城音乐项目/样式数据/ALL/节奏/节奏八/节奏八（标准音频）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/ALL/节奏/节奏八/节奏8.100分.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/ALL/旋律/1.31MP3/旋律1.100分.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/ALL/旋律/1.31MP3/旋律2.100分.wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏4卢(65).wav'
#filename = 'F:/项目/花城音乐项目/样式数据/ALL/节奏/节奏八/节奏八（1）(90).wav'
#filename = 'F:/项目/花城音乐项目/样式数据/ALL/节奏/节奏八/节奏八（2）（90分）.wav'
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

#y, sr = load_and_trim('F:/项目/花城音乐项目/样式数据/ALL/旋律/1.31MP3/旋律1.100分.wav')
y, sr = load_and_trim(filename)

CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref = np.max)
#librosa.display.specshow(CQT)
plt.figure(figsize=(10, 10))
plt.subplot(3,1,1) #要生成两行两列，这是第一个图plt.subplot('行','列','编号')
# plt.colorbar(format='%+2.0f dB')
# plt.title('Constant-Q power spectrogram (note)')
librosa.display.specshow(CQT, y_axis='cqt_note',x_axis='time')
onsets_frames =  librosa.onset.onset_detect(y)
onsets_frames = get_real_onsets_frames_rhythm(y)
print(np.max(y))
onstm = librosa.frames_to_time(onsets_frames, sr=sr)
plt.vlines(onstm, 0,sr, color='y', linestyle='solid')
print(CQT.shape)
q1,q2 = CQT.shape
print(plt.figure)

plt.subplot(3,1,2) #要生成两行两列，这是第一个图plt.subplot('行','列','编号')


librosa.display.waveplot(y, sr=sr)
plt.vlines(onstm, -1*np.max(y),np.max(y), color='y', linestyle='solid')


# duration = librosa.get_duration(filename=filename)
# # 标准节拍时间点
# base_onsets = onsets_base(codes[11], duration, onstm[0])
# plt.vlines(base_onsets[:-1], -1*np.max(y),np.max(y), color='r', linestyle='dashed')
# plt.vlines(base_onsets[-1], -1*np.max(y),np.max(y), color='white', linestyle='dashed')


plt.subplot(3,1,3)

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
librosa.display.specshow(img, x_axis='time', y_axis='chroma', cmap='coolwarm')
plt.vlines(onstm, 0, sr, color='y', linestyle='solid')
plt.show()