
import librosa
import os
from collections import Counter
import numpy as np
from functools import reduce


def clean2(y, sr):
    frequencies, D = librosa.ifgram(y, sr=sr)
    top = abs(D).max() / 30
    D[abs(D)<top] = 0
    y = librosa.istft(D)
    return y

def getfrequeciesdistribute(filepath, num=100):
    y, sr = librosa.load(filepath)
    y = clean2(y, sr)
    frequencies, D = librosa.ifgram(y, sr=sr)
    frequencies = frequencies.astype(int)
    frequencies = frequencies // 2 * 2
    c = Counter(frequencies.flatten().tolist())
    data = c.most_common(num)
    data = list(filter(lambda x :x[0 ] !=0 ,data))
    s = reduce(lambda x1, x2: (0, x1[1] + x2[1]), data)[1]
    frequeciesdistribute = np.array(list(map(lambda t: t[0] * t[1], data)))
    frequeciesdistribute = frequeciesdistribute / s
    #print(frequeciesdistribute)
    return frequeciesdistribute

def getfrequeciesdistribute2(filepath, num=100):
    y, sr = librosa.load(filepath)
    y = clean2(y, sr)
    frequencies, D = librosa.ifgram(y, sr=sr)
    frequencies = frequencies.astype(int)
    frequencies = frequencies // 2 * 2
    c = Counter(frequencies.flatten().tolist())
    data = c.most_common(num)
    data = list(filter(lambda x :x[0 ] !=0 ,data))
    frequeciesdistribute = np.array(list(map(lambda t: t[0], data)))
    return frequeciesdistribute

if __name__ == '__main__':
    filename = 'F:/项目/花城音乐项目/样式数据/9.08MP3/旋律/zx1.wav'
    # freq = getfrequeciesdistribute(filename, 20)
    freq = getfrequeciesdistribute2(filename, 10)
    print(freq)

    filename = 'F:/项目/花城音乐项目/样式数据/9.12MP3/tuningFork01.wav'
    filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/9.08MP3/旋律/xx3.wav', '[2000;250,250,250,250,1000;2000;500,500,1000]', '[6,5,6,3,5,6,3,2,1,6-]'
    y, sr = librosa.load(filename,offset=0.75, duration=0.2)
    y, sr = librosa.load(filename, offset=1.1, duration=0.2)    # -0.029\
    # y, sr = librosa.load(filename, offset=1.3, duration=0.2)    # 0.160
    # y, sr = librosa.load(filename, offset=2.6, duration=0.2)    # -0.48
    # y, sr = librosa.load(filename, offset=2.8, duration=0.2)    #-0.169
    # y, sr = librosa.load(filename, offset=3, duration=0.2)
    # y, sr = librosa.load(filename, offset=3.3, duration=0.2)
    # y, sr = librosa.load(filename, offset=3.55, duration=0.2)


    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)
    np.set_printoptions(threshold=np.nan)
    print(pitches[np.nonzero(pitches)])

    pitches = pitches[magnitudes > np.median(magnitudes)]
    p = librosa.pitch_tuning(pitches)
    print(p)

    tun = librosa.estimate_tuning(y=y, sr=sr)
    print(tun)

    onset_frames_time = [0.7662585,1.27709751,2.80961451,3.0185941,3.29723356,3.57587302,3.80807256,4.80653061,7.2678458,7.70902494]
    onset_frames_time_diff = np.diff(onset_frames_time)
    onset_frames_time_diff = list(onset_frames_time_diff)
    onset_frames_time_diff.append(0.2)
    for i,o in enumerate(onset_frames_time):
        offset = round(o,2)
        duration = round(onset_frames_time_diff[i],2)
        y, sr = librosa.load(filename, offset=offset, duration=duration)
        pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)
        pitches = pitches[magnitudes > np.median(magnitudes)]
        p = pitches[np.nonzero(pitches)][:10]
        print("i,p is {},{}".format(i,p))