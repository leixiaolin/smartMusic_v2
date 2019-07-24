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

def get_base_pitch_from_cqt(cqt):
    h,w = cqt.shape
    cqt_max = np.max(cqt)
    cqt_min = np.min(cqt)
    base_pitch = np.zeros(w)
    for i in range(w):
        col_cqt = cqt[:,i]
        tmp = [i for i in range(10,h-14) if col_cqt[i] == cqt_max and col_cqt[i+13] == cqt_max and np.min(col_cqt[i:i+13]) == cqt_min]
        if len(tmp) > 0:
            base_pitch[i] = tmp[0]
    return base_pitch



#y, sr = load_and_trim('F:/项目/花城音乐项目/样式数据/ALL/旋律/1.31MP3/旋律1.100分.wav')
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律2.1(80).wav'
filename = 'F:/项目/花城音乐项目/样式数据/ALL/旋律/1.31MP3/旋律3.100分.wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律一（9）（100）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律一（14）（95）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/节奏/节2罗（75）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏一（4）（96）.wav'
filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋2熙(0).wav'
filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1罗（96）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋10罗（92）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋2录音4(72).wav'
#filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋10罗（92）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律一（13）（98）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1熙(90).wav'
#filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1谭（98）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋3.3(96).wav'
filename = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/两只老虎20190624-2939.wav'
filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1王（98）.wav'
filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋9.1(73).wav'
filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/小学8题20190624-3898-7.wav', '[2000;250,250,250,250,1000;2000;500,500,1000]', '[6,5,6,3,5,6,3,2,1,6-]'  # 这一条故意唱错了两个音，节奏是对的，这个扣一分即可 61
filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.12MP3/旋律/小学8题20190702-2647-6.wav', '[1000,500,500;2000;250,250,500,500,500;2000]', '[6,5,3,6,3,5,3,2,1,6]'  #
# filename = 'F:/项目/花城音乐项目/样式数据/6.18MP3/节奏/17；57.wav'
# filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-6285-1.wav', '[1000,1000;500,250,250,1000;1000,500,500;2000]'  # 100
# filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-6285-2.wav', '[1000,500,500;2000;250,250,500,500,500;2000]'  # 100
# filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-6285-3.wav', '[2000;250,250,250,250,1000;2000;500,500,1000]'  # 100
filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-6285-4.wav', '[1000,250,250,250,250;2000;1000,500,500;2000]'  # 100
filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.12MP3/旋律/小学8题20190702-2647-1.wav', '[1000,1000;500,250,250,1000;1000,500,500;2000]'  # 可给满分
filename, onset_code = 'F:/项目/花城音乐项目/样式数据/6.18MP3/节奏/17；57.wav','[500,500,250,250,250,250;500,250,250,1000;250,250,250,250,750,250;250,250,500,1000]'
filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.01MP3/旋律/中学8题20190701-1547 节奏二.wav', '[1000,1000;1500,500;500,250,250,500,500;2000]'  #可给接近满分                         97 ?????
filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.12MP3/旋律/小学8题20190702-2647-8.wav', '[1000,250,250,250,250;2000;1000,500,500;2000]', '[1,3,5,1+,6,5,1,3,2,1]'  #节奏全对，旋律错最后一个  86

filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/两只老虎20190624-1089.wav', '[500,500,500,500;500,500,500,500;500,500,1000;500,500;1000]', '[1,2,3,1,1,2,3,1,3,4,5,3,4,5]'  # 音准节奏均正确，给分偏低  94
filename,rhythm_code,pitch_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/两只老虎20190624-1328.wav','[500,500,500,500;500,500,500,500;500,500,1000;500,500;1000]','[1,2,3,1,1,2,3,1,3,4,5,3,4,5]'       #音准节奏均正确，给分偏低  95
# filename,rhythm_code,pitch_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/两只老虎20190624-1586.wav','[500,500,500,500;500,500,500,500;500,500,1000;500,500;1000]','[1,2,3,1,1,2,3,1,3,4,5,3,4,5]'      #音准节奏均正确，给分偏低  95
# filename,rhythm_code,pitch_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/两只老虎20190624-2939.wav','[500,500,500,500;500,500,500,500;500,500,1000;500,500;1000]','[1,2,3,1,1,2,3,1,3,4,5,3,4,5]'      #音准节奏均正确，给分偏低 90
# filename,rhythm_code,pitch_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/两只老虎20190624-7881.wav','[500,500,500,500;500,500,500,500;500,500,1000;500,500;1000]','[1,2,3,1,1,2,3,1,3,4,5,3,4,5]'          # 音准节奏均正确，给分偏低 90
# filename,rhythm_code,pitch_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/两只老虎20190624-8973.wav','[500,500,500,500;500,500,500,500;500,500,1000;500,500;1000]','[1,2,3,1,1,2,3,1,3,4,5,3,4,5]'       #音准节奏均正确，给分偏低 98
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/小学8题20190624-3898-5.wav', '[1000,1000;500,250,250,1000;1000,500,500;2000]', '[3,1,5,5,6,5,1+,6,3,5]'  #这个给九分多是可以的 86
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/小学8题20190624-3898-6.wav', '[1000,500,500;2000;250,250,500,500,500;2000]', '[6,5,3,6,3,5,3,2,1,6-]'  #故意把最后一个音唱错了，节奏全对,扣0.5左右即可 85
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/小学8题20190624-3898-7.wav', '[2000;250,250,250,250,1000;2000;500,500,1000]', '[6,5,6,3,5,6,3,2,1,6-]'  # 这一条故意唱错了两个音，节奏是对的，这个扣一分即可 72
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/小学8题20190624-3898-8.wav', '[1000,250,250,250,250;2000;1000,500,500;2000]', '[1,3,5,1+,6,5,1,3,2,1]'  # 这一条节奏不是太稳，但音高基本正确,9.5分是没问题的 93

filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.12MP3/旋律/小学8题20190702-2647-5.wav', '[1000,1000;500,250,250,500;1000,500,500;2000]', '[3,1,5,5,6,5,1+,6,3,5]'  #  准确 96
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.12MP3/旋律/小学8题20190702-2647-6.wav', '[1000,500,500;2000;250,250,500,500,500;2000]', '[6,5,3,6,3,5,3,2,1,6]'  #基本可给满分 59
filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.12MP3/旋律/小学8题20190702-2647-7.wav', '[2000;250,250,250,250,1000;2000;500,500,1000]', '[6,5,6,3,5,6,3,2,1,6]'  #第一个节奏应该扣分，最后一个音没唱，应该没分  79
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.12MP3/旋律/小学8题20190702-2647-8.wav', '[1000,250,250,250,250;2000;1000,500,500;2000]', '[1,3,5,1+,6,5,1,3,2,1]'  #节奏全对，旋律错最后一个  86
# filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋4.4(0).wav'
# filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋2.3(95).wav'
# filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋2.4(50).wav'
# filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋2录音1(90).wav'

y, sr = load_and_trim(filename)
y,sr = librosa.load(filename)
rms = librosa.feature.rmse(y=y)[0]
rms = [x / np.std(rms) for x in rms]
time = librosa.get_duration(filename=filename)
print("time is {}".format(time))
CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref = np.max)
w,h = CQT.shape
print("w.h is {},{}".format(w,h))
#onsets_frames = get_real_onsets_frames_rhythm(y)

CQT = np.where(CQT > -30, np.max(CQT), np.min(CQT))

CQT = signal.medfilt(CQT, (3, 3))  # 二维中值滤波
plt.subplot(2,1,1)
librosa.display.specshow(CQT ,x_axis='time')


plt.subplot(2,1,2)
base_pitch = get_base_pitch_from_cqt(CQT)
base_pitch = signal.medfilt(base_pitch, 7)  # 二维中值滤波
t = librosa.frames_to_time(np.arange(len(base_pitch)))
plt.plot(t, base_pitch)
plt.xlim(0, np.max(t))
plt.ylim(0, 84)

plt.show()