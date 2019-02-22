import librosa
import matplotlib.pyplot as plt
import numpy as np
import librosa.display

# 1. Get the file path to the included audio example
filepath = 'F:\项目\花城音乐项目\样式数据\ALL\节奏\节奏八\\'
filename = filepath + '节奏8.100分.wav'
# 2. Load the audio as a waveform `y`
#    Store the sampling rate as `sr`

y, sr = librosa.load(filename, sr = None)
mfcc = librosa.feature.mfcc(y, sr)
onsets_frames = librosa.onset.onset_detect(y)
librosa.display.specshow(librosa.amplitude_to_db(mfcc))
#librosa.display.specshow(mfcc)
#plt.vlines(onsets_frames, 0, sr, color='r', linestyle='--')
plt.show()
