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
# plt.title('Power spectrogram')

# CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref = np.max)
# #librosa.display.specshow(CQT)
# # plt.colorbar(format='%+2.0f dB')
# # plt.title('Constant-Q power spectrogram (note)')
# librosa.display.specshow(CQT, y_axis='cqt_note',x_axis='time')
ax1 = plt.subplot(3, 1, 2)
# Construct a standard onset function
librosa.display.waveplot(y, sr=sr)

# Construct a standard onset function
all_onset = []

onset_env = librosa.onset.onset_strength(y=y, sr=sr)
plt.subplot(3, 1, 3, sharex=ax1)
max_onset_env = [x if onset_env[i] > onset_env[i-1] and onset_env[i] > onset_env[i+1] and onset_env[i] > np.max(onset_env)*0.5 else 0 for i,x in enumerate(onset_env[1:-1]) ]
max_onset_env.append(0)
max_onset_env.insert(0,0)
max_onset_env_index = [i for i,x in enumerate(onset_env[1:-1]) if onset_env[i] > onset_env[i-1] and onset_env[i] > onset_env[i+1] and onset_env[i] > np.max(onset_env)*0.5]
print("max_onset_env_index is {}".format(max_onset_env_index))
#plt.plot(times, 2 + onset_env / onset_env.max(), alpha=0.8, label='Mean (mel)')
plt.plot(times, 2 + max_onset_env / onset_env.max())
all_onset = np.hstack((all_onset,max_onset_env_index))
# Median aggregation, and custom mel options

onset_env = librosa.onset.onset_strength(y=y, sr=sr,
                                         aggregate=np.median,
                                         fmax=8000, n_mels=512)
#print("onset_env is {}".format(onset_env))
max_onset_env = [x if onset_env[i] > onset_env[i-1] and onset_env[i] > onset_env[i+1] and onset_env[i] > np.max(onset_env)*0.5 else 0 for i,x in enumerate(onset_env[1:-1]) ]
max_onset_env.append(0)
max_onset_env.insert(0,0)
max_onset_env_index = [i for i,x in enumerate(onset_env[1:-1]) if onset_env[i] > onset_env[i-1] and onset_env[i] > onset_env[i+1] and onset_env[i] > np.max(onset_env)*0.5]
print("max_onset_env_index is {}".format(max_onset_env_index))
#plt.plot(times, 1 + onset_env / onset_env.max(), alpha=0.8, label='Median (custom mel)')
plt.plot(times, 1 + max_onset_env / onset_env.max())
all_onset = np.hstack((all_onset,max_onset_env_index))
# 节拍点
# onsets_frames = librosa.onset.onset_detect(y)
# print("onsets_frames is {}".format(onsets_frames))
# Constant-Q spectrogram instead of Mel

onset_env = librosa.onset.onset_strength(y=y, sr=sr,
                                         feature=librosa.cqt)

max_onset_env = [x if onset_env[i] > onset_env[i-1] and onset_env[i] > onset_env[i+1] and onset_env[i] > np.max(onset_env)*0.7 else 0 for i,x in enumerate(onset_env[1:-1]) ]
max_onset_env.append(0)
max_onset_env.insert(0,0)
max_onset_env_index = [i for i,x in enumerate(onset_env[1:-1]) if onset_env[i] > onset_env[i-1] and onset_env[i] > onset_env[i+1] and onset_env[i] > np.max(onset_env)*0.7]
print("max_onset_env_index is {}".format(max_onset_env_index))
#plt.plot(times, onset_env / onset_env.max(), alpha=0.8, label='Mean (CQT)')
#plt.plot(times, max_onset_env / onset_env.max(), alpha=0.8, label='Mean (CQT)')

all_onset = np.hstack((all_onset,max_onset_env_index))
news_ids = []
for id in all_onset:
    if id not in news_ids:
        news_ids.append(int(id))
all_onset = news_ids
all_onset.sort()
print("all_onset1 is {}".format(all_onset))
#all_onset = get_onsets_by_all(y,sr)
all_onset = get_onsets_by_all_v2(y,sr,1)
#all_onset = get_real_onsets_frames(y)
print("all_onset2 is {}".format(all_onset))
test = np.zeros(len(max_onset_env))
for v in all_onset:
    test[v] = 1
plt.plot(times, test / onset_env.max())
plt.legend(frameon=True, framealpha=0.75)
plt.ylabel('Normalized strength')
plt.yticks([])
plt.axis('tight')

plt.tight_layout()
plt.show()