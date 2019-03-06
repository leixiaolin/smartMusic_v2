import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
from create_base import *
# 1. Get the file path to the included audio example
# Sonify detected beat events
# 定义加载语音文件并去掉两端静音的函数
def load_and_trim(path):
    audio, sr = librosa.load(path)
    energy = librosa.feature.rmse(audio)
    frames = np.nonzero(energy >= np.max(energy) / 5)
    indices = librosa.core.frames_to_samples(frames)[1]
    audio = audio[indices[0]:indices[-1]] if indices.size else audio[0:0]

    return audio, sr

#y, sr = load_and_trim('F:/项目/花城音乐项目/样式数据/ALL/旋律/1.31MP3/旋律1.100分.wav')
y, sr = load_and_trim('F:/项目/花城音乐项目/样式数据/ALL/旋律/1.31MP3/旋律3.100分.wav')

CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref = np.max)
#librosa.display.specshow(CQT)
librosa.display.specshow(CQT, y_axis='cqt_note',x_axis='time')
onsets_frames = get_real_onsets_frames_rhythm(y)
print(np.max(y))
onstm = librosa.frames_to_time(onsets_frames, sr=sr)
plt.vlines(onstm, 0,sr, color='y', linestyle='solid')
# plt.colorbar(format='%+2.0f dB')
# plt.title('Constant-Q power spectrogram (note)')
plt.show()