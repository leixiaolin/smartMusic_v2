import librosa
import matplotlib.pyplot as plt
import numpy as np
import librosa.display

# 波形幅度包络图
filepath = 'F:\项目\花城音乐项目\样式数据\音乐样本2019-01-29\节奏九\\'
filename = 'F:/项目/花城音乐项目/样式数据/ALL/旋律/1.31MP3/旋律2.100分.wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏5_40240（30）.wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏六（13）（50）.wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏1怡(90).wav'


# 2. Load the audio as a waveform `y`
#    Store the sampling rate as `sr`

# 定义加载语音文件并去掉两端静音的函数
def load_and_trim(path):
    audio, sr = librosa.load(path)
    energy = librosa.feature.rmse(audio)
    frames = np.nonzero(energy >= np.max(energy) / 50)
    indices = librosa.core.frames_to_samples(frames)[1]
    audio = audio[indices[0]:indices[-1]] if indices.size else audio[0:0]

    return audio, sr

y, sr = load_and_trim(filename)
rms = librosa.feature.rmse(y=y)[0]

times = librosa.frames_to_time(np.arange(len(rms)))

plt.figure(figsize=(12, 4))
plt.plot(times, rms)
plt.axhline(0.02, color='r', alpha=0.5)
plt.xlabel('Time')
plt.ylabel('RMS')
plt.axis('tight')
plt.tight_layout()
print("rms min :{}".format(np.min(rms)))
print("rms max :{}".format(np.max(rms)))
print("rms mean :{}".format(np.mean(rms)))

#plt.axis('off')
# plt.axes().get_xaxis().set_visible(False)
# plt.axes().get_yaxis().set_visible(False)
plt.show()