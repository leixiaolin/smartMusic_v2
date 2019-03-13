import crepe
from scipy.io import wavfile
import matplotlib.pyplot as plt
import librosa
from filters import *
from create_base import *


filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏8_40434（30）.wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/视唱1-01（95）.wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/视唱1-02（90）.wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律1.1(95).wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律2（四）(96).wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律2卢(98).wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律2.1(80).wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律2.3(55).wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律二（10）（75）.wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律二（8）（100）.wav'


sr, audio = wavfile.read(filename)
audio, sr = load_and_trim(filename)
time = librosa.get_duration(audio)
print("time is {}".format(time))
time, frequency, confidence, activation = crepe.predict(audio, sr,model_capacity='full', step_size=10, viterbi=True)
step = 3
len = len(frequency)
frequency = ArithmeticAverage(frequency.copy(), step)
#y2 = ArithmeticAverage(y.copy(),step)
#y2 = MedianAverage(y.copy(),step)

frequency = expand_output(frequency,step,len)
frequency = get_nearly_note(frequency,step)
plt.plot(time,frequency)
plt.show()
print(frequency)
