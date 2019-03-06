import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
from create_base import *

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

#filename = 'F:/项目/花城音乐项目/样式数据/ALL/节奏/节奏八（6）(95).wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏2林(20).wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏4卢(65).wav'
#y, sr = load_and_trim('F:/项目/花城音乐项目/样式数据/ALL/旋律/1.31MP3/旋律1.100分.wav')
y, sr = load_and_trim(filename)
librosa.display.waveplot(y, sr=sr)
#onsets_frames = get_real_onsets_frames(y)
onsets_frames = get_real_onsets_frames_by_strength(y,sr)
onstm = librosa.frames_to_time(onsets_frames, sr=sr)
# 音频时长
duration = librosa.get_duration(filename=filename)
# 标准节拍时间点
base_onsets = onsets_base(codes[7], duration, onstm[0])
plt.vlines(onstm, -1*np.max(y),np.max(y), color='b', linestyle='solid')
# plt.vlines(base_onsets[:-1], -1*np.max(y),np.max(y), color='r', linestyle='dashed')
# plt.vlines(base_onsets[-1], -1*np.max(y),np.max(y), color='white', linestyle='dashed')
plt.show()