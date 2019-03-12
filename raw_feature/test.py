import crepe
from scipy.io import wavfile
import matplotlib.pyplot as plt
import librosa


filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏8_40434（30）.wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/视唱1-01（95）.wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/视唱1-02（90）.wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律1.1(95).wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律2（四）(96).wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律2卢(98).wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律2.1(80).wav'
sr, audio = wavfile.read(filename)
time = librosa.get_duration(filename=filename)
print(time)
time, frequency, confidence, activation = crepe.predict(audio, sr,step_size=10, viterbi=True)
plt.plot(time,frequency)
plt.show()
print(frequency)
