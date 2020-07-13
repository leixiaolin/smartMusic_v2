import librosa
import numpy as np
import matplotlib.pyplot as plt
path = 'F:/项目/花城音乐项目/样式数据/ALL/旋律/1.31MP3/旋律6.100分.wav'
y,sr = librosa.load(path,sr=44100)
S = np.abs(librosa.stft(y,n_fft=1024,hop_length=512,center=False))
sf = np.zeros((1,S.shape[1]-1))
for i in range(S.shape[1]-1):
    sf[:,i] = sum((S[:,i+1]-S[:,i])**2)

print(sf.shape)
plt.plot(sf[0])
plt.show()