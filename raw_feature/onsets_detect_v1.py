# Get onset times from a signal
import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np

y, sr = librosa.load(librosa.util.example_audio_file(),
                     offset=30, duration=2.0)
onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
librosa.frames_to_time(onset_frames, sr=sr)
# array([ 0.07 ,  0.395,  0.511,  0.627,  0.766,  0.975,
# 1.207,  1.324,  1.44 ,  1.788,  1.881])

# Or use a pre-computed onset envelope

o_env = librosa.onset.onset_strength(y, sr=sr)
times = librosa.frames_to_time(np.arange(len(o_env)), sr=sr)
onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr)

import matplotlib.pyplot as plt
D = np.abs(librosa.stft(y))
plt.figure()
ax1 = plt.subplot(3, 1, 1)
librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
                         x_axis='time', y_axis='log')
plt.title('Power spectrogram')

plt.subplot(3, 1, 2, sharex=ax1)
librosa.display.waveplot(y, sr=sr)


plt.subplot(3, 1, 3, sharex=ax1)
plt.plot(times, o_env, label='Onset strength')
plt.vlines(times[onset_frames], 0, o_env.max(), color='r', alpha=0.9,
           linestyle='--', label='Onsets')
plt.axis('tight')
plt.legend(frameon=True, framealpha=0.75)
plt.show()