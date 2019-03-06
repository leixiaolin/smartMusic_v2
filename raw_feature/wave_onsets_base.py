import librosa
import matplotlib.pyplot as plt
import numpy as np
import librosa.display
from create_base import *

# 波形幅度包络图
filepath = 'F:\项目\花城音乐项目\样式数据\音乐样本2019-01-29\节奏九\\'
#filename = 'F:/项目/花城音乐项目/样式数据/ALL/节奏/节奏八/节奏八（标准音频）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/ALL/节奏/节奏八/节奏8.40分.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/ALL/节奏/节奏八/节奏八（11）（60）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/ALL/节奏/节奏八/节奏八（1）(90).wav'
#filename = 'F:/项目/花城音乐项目/样式数据/ALL/节奏/节奏五/节奏5.20分.wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏2.4(90).wav'
# 2. Load the audio as a waveform `y`
#    Store the sampling rate as `sr`z

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

# 定义加载语音文件并去掉两端静音的函数
def load_and_trim(path):
    audio, sr = librosa.load(path)
    energy = librosa.feature.rmse(audio)
    frames = np.nonzero(energy >= np.max(energy) / 5)
    indices = librosa.core.frames_to_samples(frames)[1]
    audio = audio[indices[0]:indices[-1]] if indices.size else audio[0:0]

    return audio, sr

y, sr = load_and_trim(filename)

# # 波形
# librosa.display.waveplot(y, sr=sr)
# D = librosa.stft(y)
# librosa.display.specshow(librosa.amplitude_to_db(D))
#

# print(sr/time)
# # 节拍点
# onsets_frames = librosa.onset.onset_detect(y)
# print(onsets_frames)
#
# # 节拍时间点
# onstm = librosa.frames_to_time(onsets_frames, sr=sr)
# print(onstm)
# plt.vlines(onstm, 0, sr, color='black', linestyle='dashed')
# plt.axis('off')
# plt.axes().get_xaxis().set_visible(False)
# plt.axes().get_yaxis().set_visible(False)
y_max = max(y)
#y = np.array([x if x > y_max*0.01 else y_max*0.01 for x in y])
# 获取每个帧的能量
energy = librosa.feature.rmse(y)
energy_mean = np.mean(energy)  # 获取能量均值
energy_gap = energy_mean * 0.8
print(np.mean(energy))
onsets_frames = librosa.onset.onset_detect(y)

print(onsets_frames)
print(np.diff(onsets_frames))
indices = librosa.core.frames_to_samples(onsets_frames)
print(indices)
some_y = [energy[0][x] for x in onsets_frames]
onsets_frames = [x for x in onsets_frames  if energy[0][x]> energy_gap]  # 筛选能量过低的伪节拍点
print(some_y)
D = librosa.stft(y)
librosa.display.waveplot(y, sr=sr)
print(np.max(y))
onstm = librosa.frames_to_time(onsets_frames, sr=sr)
#
plt.vlines(onstm, -1*np.max(y),np.max(y), color='y', linestyle='solid')
# 音频时长
duration = librosa.get_duration(filename=filename)
# 标准节拍时间点
base_onsets = onsets_base(codes[7], duration, onstm[0])
# plt.vlines(base_onsets[:-1], -1*np.max(y),np.max(y), color='r', linestyle='dashed')
# plt.vlines(base_onsets[-1], -1*np.max(y),np.max(y), color='white', linestyle='dashed')
plt.show()