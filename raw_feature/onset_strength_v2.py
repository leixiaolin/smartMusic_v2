# First, load some audio and plot the spectrogram
import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
from create_base import *


#filename = 'F:/项目/花城音乐项目/样式数据/ALL/旋律/2.2MP3/旋律一（8）(90).wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏2林(20).wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏4卢(65).wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏1_40227（100）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏4_40323（90）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏3-04（95）.wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏2.4(90).wav'
#filename = 'F:/项目/花城音乐项目/参考代码/tensorflow_models_nets-master/raw_data/onsets/test/A\节奏1-02（90）.wav'

def load_and_trim(path):
    audio, sr = librosa.load(path)
    energy = librosa.feature.rmse(audio)
    frames = np.nonzero(energy >= np.max(energy) / 5)
    indices = librosa.core.frames_to_samples(frames)[1]
    audio = audio[indices[0]:indices[-1]] if indices.size else audio[0:0]

    return audio, sr

#y, sr = librosa.load(filename)
y,sr = load_and_trim(filename)

D = np.abs(librosa.stft(y))
times = librosa.frames_to_time(np.arange(D.shape[1]))
plt.figure()
ax1 = plt.subplot(3, 1, 1)
librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
                         y_axis='log', x_axis='time')
plt.title('Power spectrogram')

ax1 = plt.subplot(3, 1, 2)
# Construct a standard onset function
librosa.display.waveplot(y, sr=sr)


onset_env = librosa.onset.onset_strength(y=y, sr=sr)
plt.subplot(3, 1, 3, sharex=ax1)
plt.plot(times, 2 + onset_env / onset_env.max(), alpha=0.8,label='Mean (mel)')
#plt.plot(times, 2 + onset_env, alpha=0.8,label='Mean (mel)')

# Median aggregation, and custom mel options

onset_env = librosa.onset.onset_strength(y=y, sr=sr,
                                         aggregate=np.mean,
                                         fmax=8000, n_mels=256)
plt.plot(times, 1 + onset_env / onset_env.max(), alpha=0.8,label='Median (custom mel)')
#plt.plot(times, 1 + onset_env, alpha=0.8,label='Median (custom mel)')

# Constant-Q spectrogram instead of Mel

onset_env = librosa.onset.onset_strength(y=y, sr=sr,
                                         feature=librosa.cqt)
plt.plot(times, onset_env / onset_env.max(), alpha=0.8,label='Mean (CQT)')
#plt.plot(times, onset_env, alpha=0.8,label='Mean (CQT)')
plt.legend(frameon=True, framealpha=0.75)
plt.ylabel('Normalized strength')
plt.yticks([])
plt.axis('tight')
plt.tight_layout()
plt.show()