import matplotlib.pyplot as plt
import librosa
import librosa.display
import librosa.util
import numpy as np
import pandas as pd

# 要转换的输入wav音频文件
filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/9.08MP3/旋律/xx3.wav', '[2000;250,250,250,250,1000;2000;500,500,1000]', '[6,5,6,3,5,6,3,2,1,6-]'
input_wav = 'F:/项目/花城音乐项目/样式数据/9.08MP3/旋律/xx3.wav'

y, sr = librosa.load(input_wav, sr=None, duration=None)
chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

c = pd.DataFrame(chroma)
c0 = (c == 1)
c1 = c0.astype(int)
labels = np.array(range(1, 13))
note_values = labels.dot(c1)

plt.figure(figsize=(15, 20))

plt.subplots_adjust(wspace=1, hspace=0.2)

plt.subplot(311)
librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
plt.xlabel('note')
plt.ylabel('beat')
note_values = labels.dot(c1)

plt.subplot(312)
librosa.display.waveplot(y, sr=sr)
plt.xlabel('second')
plt.ylabel('amplitude')

plt.subplot(313)
plt.grid(linewidth=0.5)
plt.xticks(range(0, 600, 50))
plt.yticks(range(1, 13), ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"])
plt.scatter(range(len(note_values)), note_values, marker="s", s=1, color="red")

plt.show()
