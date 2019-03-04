import librosa
import matplotlib.pyplot as plt
import numpy as np
import librosa.display


#filename = 'F:/项目/花城音乐项目/样式数据/ALL/旋律/1.31MP3/旋律1.100分.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/ALL/节奏/节奏一/节奏1.100分.wav'
filename = 'F:/项目/花城音乐项目/样式数据/ALL/旋律/1.31MP3/旋律6.100分.wav'

def load_and_trim(path):
    audio, sr = librosa.load(path)
    energy = librosa.feature.rmse(audio)
    frames = np.nonzero(energy >= np.max(energy) / 10)
    indices = librosa.core.frames_to_samples(frames)[1]
    audio = audio[indices[0]:indices[-1]] if indices.size else audio[0:0]

    return audio, sr

#y, sr = librosa.load(filename)
y,sr = load_and_trim(filename)

chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)


c_max = np.argmax(chroma_cens,axis=0)
print(c_max.shape[0])
print(c_max)
print(np.diff(c_max))

img = np.zeros(chroma_cens.shape,dtype=np.float32)
w,h = chroma_cens.shape
for x in range(h):
    #img.item(x, c_max[x], 0)
    img.itemset((c_max[x],x), 1)
    img.itemset((c_max[x],x), 1)
    img.itemset((c_max[x],x), 1)

# 最强音色图
plt.figure(figsize=(15, 5))
librosa.display.specshow(img, x_axis='time', y_axis='chroma', cmap='coolwarm')

plt.show()
