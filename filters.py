import scipy.signal as signal
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import matplotlib
import librosa
import librosa.display

'''
算术平均滤波法
'''


def ArithmeticAverage(inputs, per):
    if np.shape(inputs)[0] % per != 0:
        lengh = np.shape(inputs)[0] / per
        for x in range(int(np.shape(inputs)[0]), int(lengh + 1) * per):
            inputs = np.append(inputs, inputs[np.shape(inputs)[0] - 1])
    inputs = inputs.reshape((-1, per))
    mean = []
    for tmp in inputs:
        mean.append(tmp.mean())
    return mean


'''
递推平均滤波法
'''


def SlidingAverage(inputs, per):
    if np.shape(inputs)[0] % per != 0:
        lengh = np.shape(inputs)[0] / per
        for x in range(int(np.shape(inputs)[0]), int(lengh + 1) * per):
            inputs = np.append(inputs, inputs[np.shape(inputs)[0] - 1])
    inputs = inputs.reshape((-1, per))
    tmpmean = inputs[0].mean()
    mean = []
    for tmp in inputs:
        mean.append((tmpmean + tmp.mean()) / 2)
        tmpmean = tmp.mean()
    return mean


'''
递推去毛刺滤波法
'''


def deburring(inputs, step):
    for i in range(0, len(inputs) - step):
        if inputs[i] == inputs[i + step]:
            for j in range(1,step):
                inputs[i + j] = inputs[i]
    return inputs

'''
中位值平均滤波法
'''


def MedianAverage(inputs, per):
    if np.shape(inputs)[0] % per != 0:
        lengh = np.shape(inputs)[0] / per
        for x in range(int(np.shape(inputs)[0]), int(lengh + 1) * per):
            inputs = np.append(inputs, inputs[np.shape(inputs)[0] - 1])
    inputs = inputs.reshape((-1, per))
    mean = []
    for tmp in inputs:
        tmp = np.delete(tmp, np.where(tmp == tmp.max())[0], axis=0)
        if tmp.size != 0:
            tmp = np.delete(tmp, np.where(tmp == tmp.min())[0], axis=0)
            mean.append(tmp.mean())
        else:
            mean.append(0)
    return mean


'''
限幅平均滤波法
Amplitude:	限制最大振幅
'''


def AmplitudeLimitingAverage(inputs, per, Amplitude):
    if np.shape(inputs)[0] % per != 0:
        lengh = np.shape(inputs)[0] / per
        for x in range(int(np.shape(inputs)[0]), int(lengh + 1) * per):
            inputs = np.append(inputs, inputs[np.shape(inputs)[0] - 1])
    inputs = inputs.reshape((-1, per))
    mean = []
    tmpmean = inputs[0].mean()
    tmpnum = inputs[0][0]  # 上一次限幅后结果
    for tmp in inputs:
        for index, newtmp in enumerate(tmp):
            if np.abs(tmpnum - newtmp) > Amplitude:
                tmp[index] = tmpnum
            tmpnum = newtmp
        mean.append((tmpmean + tmp.mean()) / 2)
        tmpmean = tmp.mean()
    return mean


'''
一阶滞后滤波法
a:			滞后程度决定因子，0~1
'''


def FirstOrderLag(inputs, a):
    tmpnum = inputs[0]  # 上一次滤波结果
    for index, tmp in enumerate(inputs):
        inputs[index] = (1 - a) * tmp + a * tmpnum
        tmpnum = tmp
    return inputs


'''
加权递推平均滤波法
'''


def WeightBackstepAverage(inputs, per):
    weight = np.array(range(1, np.shape(inputs)[0] + 1))  # 权值列表
    weight = weight / weight.sum()

    for index, tmp in enumerate(inputs):
        inputs[index] = inputs[index] * weight[index]
    return inputs


'''
消抖滤波法
N:			消抖上限
'''


def ShakeOff(inputs, N):
    usenum = inputs[0]  # 有效值
    i = 0  # 标记计数器
    for index, tmp in enumerate(inputs):
        if tmp != usenum:
            i = i + 1
            if i >= N:
                i = 0
                inputs[index] = usenum
    return inputs


'''
限幅消抖滤波法
Amplitude:	限制最大振幅
N:			消抖上限
'''


def AmplitudeLimitingShakeOff(inputs, Amplitude, N):
    # print(inputs)
    tmpnum = inputs[0]
    for index, newtmp in enumerate(inputs):
        if np.abs(tmpnum - newtmp) > Amplitude:
            inputs[index] = tmpnum
        tmpnum = newtmp
    # print(inputs)
    usenum = inputs[0]
    i = 0
    for index2, tmp2 in enumerate(inputs):
        if tmp2 != usenum:
            i = i + 1
            if i >= N:
                i = 0
                inputs[index2] = usenum
    # print(inputs)
    return inputs

def expand_output(output,step,len):
    tmp = []
    for x in output:
        for i in range(step):
            tmp.append(x)
    return np.array(tmp[0:len])

# 波形幅度包络图
filepath = 'F:\项目\花城音乐项目\样式数据\音乐样本2019-01-29\节奏九\\'
filename = 'F:/项目/花城音乐项目/样式数据/ALL/旋律/1.31MP3/旋律2.100分.wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏8_40434（30）.wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/视唱1-01（95）.wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/视唱1-02（90）.wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律1.1(95).wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律2（四）(96).wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律2卢(98).wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律2.1(80).wav'
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

if __name__ == "__main__":
    y, sr = load_and_trim(filename)
    time = librosa.get_duration(filename=filename)
    T = np.arange(0, 0.5, 1 / 4410.0)
    num = signal.chirp(T, f0=10, t1=0.5, f1=1000.0)
    pl.subplot(2, 1, 1)
    librosa.display.waveplot(y, sr=sr)
    print(y.shape)
    #pl.plot(num)
    #result = ArithmeticAverage(num.copy(), 30)

    # print(num - result)
    pl.subplot(2, 1, 2)
    step = 20
    #pl.plot(result)
    #y2 = SlidingAverage(y.copy(), step)
    y2 = ArithmeticAverage(y.copy(),step)
    #y2 = SlidingAverage(y.copy(),step)

    y2 = expand_output(y2,step,len(y))
    librosa.display.waveplot(y2, sr=sr)
    print(y2.shape)
    #pl.plot(time,y2)
    pl.show()