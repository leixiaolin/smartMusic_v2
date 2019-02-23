import librosa
import matplotlib.pyplot as plt
import numpy as np
import librosa.display
from create_base import *

# 波形幅度包络图
filepath = 'F:\项目\花城音乐项目\样式数据\音乐样本2019-01-29\节奏九\\'
# filename = 'F:/项目/花城音乐项目/样式数据/ALL/节奏/节奏八/节奏八（标准音频）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/ALL/节奏/节奏八/节奏8.100分.wav'
filename = 'F:/项目/花城音乐项目/样式数据/ALL/节奏/节奏八/节奏8.95分.wav'
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
                  '[500,500,1000;500,500;1000;375,125,250,250,375,125,250,250;500,500,1000]'])

# 定义加载语音文件并去掉两端静音的函数
def load_and_trim(path):
    audio, sr = librosa.load(path)
    energy = librosa.feature.rmse(audio)
    frames = np.nonzero(energy >= np.max(energy) / 10)
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

onsets_frames = librosa.onset.onset_detect(y)
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
plt.vlines(base_onsets[:-1], -1*np.max(y),np.max(y), color='r', linestyle='dashed')
plt.vlines(base_onsets[-1], -1*np.max(y),np.max(y), color='white', linestyle='dashed')
plt.show()