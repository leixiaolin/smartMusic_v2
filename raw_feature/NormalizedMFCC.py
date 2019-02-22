import librosa
import matplotlib.pyplot as plt
import numpy as np
import librosa.display
from python_speech_features import mfcc
import pickle
import wave
from mpl_toolkits.axes_grid1 import make_axes_locatable


mfcc_dim = 13
texts = ['test']

mfcc_dim = 13
sr = 16000
min_length = 1 * sr
slice_length = 3 * sr

path ='F:/项目/花城音乐项目/样式数据/ALL/旋律/1.31MP3/旋律2.100分.wav'

def load_and_trim(path, sr=16000):
    audio = np.memmap(path, dtype='h', mode='r')
    audio = audio[2000:-2000]
    audio = audio.astype(np.float32)
    energy = librosa.feature.rmse(audio)
    frames = np.nonzero(energy >= np.max(energy) / 5)
    indices = librosa.core.frames_to_samples(frames)[1]
    audio = audio[indices[0]:indices[-1]] if indices.size else audio[0:0]

    slices = []
    for i in range(0, audio.shape[0], slice_length):
        s = audio[i: i + slice_length]
        if s.shape[0] >= min_length:
            slices.append(s)

    return audio, slices


def pcm2wav(pcm_path, wav_path, channels=1, bits=16, sample_rate=sr):
    data = open(pcm_path, 'rb').read()
    fw = wave.open(wav_path, 'wb')
    fw.setnchannels(channels)
    fw.setsampwidth(bits // 8)
    fw.setframerate(sample_rate)
    fw.writeframes(data)
    fw.close()



audio, slices = load_and_trim(path)
print('Duration: %.2f s' % (audio.shape[0] / sr))
plt.figure(figsize=(12, 5))
plt.plot(np.arange(len(audio)), audio)
plt.title('Raw Audio Signal')
plt.xlabel('Time')
plt.ylabel('Audio Amplitude')
plt.show()

feature = mfcc(audio, sr, numcep=mfcc_dim)
print('Shape of MFCC:', feature.shape)
fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(111)
im = ax.imshow(feature, cmap=plt.cm.jet, aspect='auto')
plt.title('Normalized MFCC')
plt.ylabel('Time')
plt.xlabel('MFCC Coefficient')
plt.colorbar(im, cax=make_axes_locatable(ax).append_axes('right', size='5%', pad=0.05))
ax.set_xticks(np.arange(0, 13, 2), minor=False);
plt.show()
