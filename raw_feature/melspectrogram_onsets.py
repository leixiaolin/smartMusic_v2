import librosa
import matplotlib.pyplot as plt
import numpy as np
import librosa.display

# 1. Get the file path to the included audio example
filename = 'F:/项目/花城音乐项目/样式数据/ALL/旋律/1.31MP3/旋律2.100分.wav'
# 2. Load the audio as a waveform `y`
#    Store the sampling rate as `sr`

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

y, sr = load_and_trim('F:/项目/花城音乐项目/样式数据/ALL/旋律/1.31MP3/旋律3.100分.wav')
ps = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
ps_db = librosa.power_to_db(ps, ref = np.max)
librosa.display.specshow(ps_db,x_axis='time', y_axis='mel')
#librosa.display.specshow(librosa.power_to_db(ps, ref = np.max))
# plt.axis('off')

# 音频时长
time = librosa.get_duration(filename=filename)
print(sr/time)
# 节拍点
onsets_frames = librosa.onset.onset_detect(y)
print(onsets_frames)

# 节拍时间点
onstm = librosa.frames_to_time(onsets_frames, sr=sr)
print(onstm)
#plt.rcParams['figure.figsize'] = (2.0, 2.0) # 设置figure_size尺寸
#plt.rcParams['savefig.dpi'] = 28 #图片像素
#plt.rcParams['figure.dpi'] = 28 #分辨率
#librosa.display.specshow(librosa.amplitude_to_db(D))
plt.vlines(onstm, 0, sr, color='black', linestyle='dashed')
#plt.axis('off')
# plt.axes().get_xaxis().set_visible(False)
# plt.axes().get_yaxis().set_visible(False)
plt.savefig('f:/3.png', bbox_inches='tight', pad_inches=0)


plt.show()
