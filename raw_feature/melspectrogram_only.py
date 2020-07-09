import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
from create_base import *
# 1. Get the file path to the included audio example
# Sonify detected beat events
# 定义加载语音文件并去掉两端静音的函数

#y, sr = load_and_trim('F:/项目/花城音乐项目/样式数据/ALL/旋律/1.31MP3/旋律1.100分.wav')
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律2.1(80).wav'
filename = 'F:/项目/花城音乐项目/样式数据/ALL/旋律/1.31MP3/旋律3.100分.wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律一（9）（100）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律一（14）（95）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/节奏/节2罗（75）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏一（4）（96）.wav'
filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋2熙(0).wav'
filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1罗（96）.wav'
filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋10罗（92）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋2录音4(72).wav'
filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋10罗（92）.wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律一（13）（98）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1熙(90).wav'
filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1谭（98）.wav'
filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋3.3(96).wav'
filename = 'F:/项目/花城音乐项目/音符起始点检测/ismir_2018_dataset_for_reviewing/jingju_a_cappella_singing_dataset/wav/danAll/daeh-Bie_yuan_zhong-Mei_fei-qm.wav'
filename = 'F:/项目/花城音乐项目/音符起始点检测/part1/wav/laosheng/lsxp-Wo_zheng_zai-Kong_cheng_ji01-upf.wav'



y, sr = load_and_trim(filename)
y,sr = librosa.load(filename)
rms = librosa.feature.rmse(y=y)[0]
rms = [x / np.std(rms) for x in rms]
time = librosa.get_duration(filename=filename)
print("time is {}".format(time))
ps = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
#librosa.display.specshow(librosa.power_to_db(ps, ref = np.max),x_axis='time', y_axis='mel')
librosa.display.specshow(librosa.power_to_db(ps, ref = np.max), y_axis='cqt_note',x_axis='time')

onset_times = [6.9698205821822246, 7.5894989411718115, 8.264301652413579, 8.885923240505779, 9.55491569235235, 11.51082342248964, 12.117943740061385, 12.825207034671061, 13.181546815149462, 14.069984461806495, 16.025324719231367, 16.564204874760474, 17.197595734773326, 17.77633412843597, 18.442772437054227, 19.637502315982143, 20.516055283847113, 20.727375880526537, 21.303560130960868, 22.906435109518124, 23.506956398197463, 24.663191907458298, 25.254844191461338, 26.8163170655966, 27.330629181756105, 28.00985668564216, 29.07547547246497, 29.63598246150745, 30.209914350225702, 30.547039472933957, 31.096296839432092, 31.36660459373114, 31.747299957373887, 31.984472326979194, 34.82386195327391, 36.00966747328043, 36.309135506138055, 36.55588245711194]

plt.vlines(onset_times,1,200,colors="r",linestyles='solid')

plt.show()