import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np


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

#y, sr = load_and_trim('F:/项目/花城音乐项目/样式数据/ALL/旋律/1.31MP3/旋律1.100分.wav')
y, sr = load_and_trim('F:/项目/花城音乐项目/样式数据/ALL/旋律/1.31MP3/旋律3.100分.wav')
hop_length = 512
#chromagram = librosa.feature.chroma_cqt(y, sr=sr, hop_length=hop_length)
chromagram = librosa.feature.chroma_cqt(y, sr=sr)
# chromagram[11,:] = 1
# plt.figure(figsize=(15, 5))
# librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='coolwarm')
# plt.colorbar()
# plt.show()

chromagram *= 255.0
chromagram = chromagram[::-1]
plt.imshow(chromagram)
plt.show()