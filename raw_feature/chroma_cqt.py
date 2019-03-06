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

#filename = 'F:/项目/花城音乐项目/样式数据/ALL/旋律/1.31MP3/旋律6.100分.wav'
filename = 'F:/项目/花城音乐项目/样式数据/ALL/旋律/1.31MP3/旋律2.100分.wav'
#y, sr = load_and_trim('F:/项目/花城音乐项目/样式数据/ALL/旋律/1.31MP3/旋律1.100分.wav')
y, sr = load_and_trim(filename)
hop_length = 1048
#chromagram = librosa.feature.chroma_cqt(y, sr=sr, hop_length=hop_length)
chromagram = librosa.feature.chroma_cqt(y, sr=sr)
# chromagram[11,:] = 1
plt.figure(figsize=(15, 5))

plt.subplot(2,1,1)
# 原始音色图
librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', cmap='coolwarm')
plt.colorbar()
# plt.show()
plt.subplot(2,1,2)
c_max = np.argmax(chromagram,axis=0)
print(c_max.shape[0])
print(c_max)
print(np.diff(c_max))
# chromagram_diff = np.diff(chromagram,axis=0)
# print(chromagram_diff)
# sum_chromagram_diff = chromagram_diff.sum(axis=0)
# test = np.array(sum_chromagram_diff)
# plt.plot(test)

img = np.zeros(chromagram.shape,dtype=np.float32)
w,h = chromagram.shape
for x in range(h):
    #img.item(x, c_max[x], 0)
    if x < 30:
        img.itemset((c_max[x],x), 0.5)
        img.itemset((c_max[x],x), 0.5)
        img.itemset((c_max[x],x), 0.5)
    else:
        img.itemset((c_max[x],x), 1)
        img.itemset((c_max[x],x), 1)
        img.itemset((c_max[x],x), 1)

# 最强音色图
librosa.display.specshow(img, cmap='coolwarm')

#plt.colorbar()
plt.show()
