# First, load some audio and plot the spectrogram
import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np

#filename = 'F:/项目/花城音乐项目/样式数据/ALL/旋律/1.31MP3/旋律一（8）(90).wav'
filename = 'F:/项目/花城音乐项目/样式数据/ALL/旋律/2.2MP3/旋律一（8）(90).wav'
#filename = 'F:/项目/花城音乐项目/样式数据/ALL/节奏/节奏一/节奏1.100分.wav'

def load_and_trim(path):
    audio, sr = librosa.load(path)
    energy = librosa.feature.rmse(audio)
    frames = np.nonzero(energy >= np.max(energy) / 10)
    indices = librosa.core.frames_to_samples(frames)[1]
    audio = audio[indices[0]:indices[-1]] if indices.size else audio[0:0]

    return audio, sr

#y, sr = librosa.load(filename)
y,sr = load_and_trim(filename)

D = np.abs(librosa.stft(y))
times = librosa.frames_to_time(np.arange(D.shape[1]))
plt.figure()
ax1 = plt.subplot(2, 1, 1)
librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
                         y_axis='log', x_axis='time')
plt.title('Power spectrogram')

# Construct a standard onset function

onset_env = librosa.onset.onset_strength(y=y, sr=sr)
plt.subplot(2, 1, 2, sharex=ax1)
plt.plot(times, 2 + onset_env / onset_env.max(), alpha=0.8,
         label='Mean (mel)')

# Median aggregation, and custom mel options

onset_env = librosa.onset.onset_strength(y=y, sr=sr,
                                         aggregate=np.median,
                                         fmax=8000, n_mels=512)
plt.plot(times, 1 + onset_env / onset_env.max(), alpha=0.8,
         label='Median (custom mel)')
print("onset_env is {}".format(onset_env))
max_onset_env = [i for i,x in enumerate(onset_env[1:-1]) if onset_env[i] > onset_env[i-1] and onset_env[i] > onset_env[i+1]]
print("max_onset_env is {}".format(max_onset_env))
# 节拍点
onsets_frames = librosa.onset.onset_detect(y)
print("onsets_frames is {}".format(onsets_frames))
# Constant-Q spectrogram instead of Mel

onset_env = librosa.onset.onset_strength(y=y, sr=sr,
                                         feature=librosa.cqt)
plt.plot(times, onset_env / onset_env.max(), alpha=0.8,
         label='Mean (CQT)')
plt.legend(frameon=True, framealpha=0.75)
plt.ylabel('Normalized strength')
plt.yticks([])
plt.axis('tight')
plt.tight_layout()
plt.show()