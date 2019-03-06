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
ax1 = plt.subplot(2, 1, 1)
# librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
#                          y_axis='log', x_axis='time')
# plt.title('Power spectrogram')

# CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref = np.max)
# #librosa.display.specshow(CQT)
# # plt.colorbar(format='%+2.0f dB')
# # plt.title('Constant-Q power spectrogram (note)')
# librosa.display.specshow(CQT, y_axis='cqt_note',x_axis='time')
librosa.display.waveplot(y, sr=sr)

# Construct a standard onset function
all_onset = []

onset_env = librosa.onset.onset_strength(y=y, sr=sr)
plt.subplot(2, 1, 2, sharex=ax1)
audio, sr = librosa.load(filename)
energy = librosa.feature.rmse(audio)
max_energy_index = find_n_largest(energy[0],10)
max_energy = [1 if x in max_energy_index else 0 for x in range(len(onset_env)) ]
print("max_energy_index is {}".format(max_energy_index))
plt.plot(times, max_energy, alpha=1, label='Median (custom mel)')
plt.legend(frameon=True, framealpha=0.75)
plt.ylabel('Normalized strength')
plt.yticks([])
plt.axis('tight')

plt.tight_layout()
plt.show()