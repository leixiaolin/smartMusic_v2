# -*- coding:utf-8 -*-
import re
import numpy, wave,matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import librosa
import librosa.display
from PIL import Image
import re
import shutil
from create_base import *
from create_labels_files import *
from myDtw import *

def load_and_trim(path):
    audio, sr = librosa.load(path)
    energy = librosa.feature.rmse(audio)
    frames = np.nonzero(energy >= np.max(energy) / 5)
    indices = librosa.core.frames_to_samples(frames)[1]
    audio = audio[indices[0]:indices[-1]] if indices.size else audio[0:0]

    return audio, sr


def load_and_trim_v2(path,offset,duration):
    audio, sr = librosa.load(path, offset=offset, duration=duration)
    energy = librosa.feature.rmse(audio)
    frames = np.nonzero(energy >= np.max(energy) / 5)
    indices = librosa.core.frames_to_samples(frames)[1]
    audio = audio[indices[0]:indices[-1]] if indices.size else audio[0:0]

    return audio, sr

def clear_dir(dis_dir):
    shutil.rmtree(dis_dir)
    os.mkdir(dis_dir)
#savepath = 'e:/test_image/'
savepath = './single_onsets/data/test/'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏10_40411（80）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏1（二）(100).wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏1_40227（100）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏1林(70).wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏2_40314（100）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏2_40409（98）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏2林(25).wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏2语(85).wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏10_40411（85）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏10-04（80）.wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/视唱6-01(70).wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/视唱2-02（92）.wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/视唱2-03（90）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律十（1）（100）.wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律三（14）（90）.wav'



clear_dir(savepath)
y, sr = librosa.load(filename)
librosa.display.waveplot(y, sr=sr)
n_fft = 441
hop_size = 220
x_1_chroma = librosa.feature.chroma_stft(y=y, sr=sr, tuning=3, norm=2,hop_length=hop_size, n_fft=n_fft)
plt.title('Chroma Representation of $X_1$')
librosa.display.specshow(x_1_chroma, x_axis='time',y_axis='chroma', cmap='gray_r', hop_length=hop_size)
h = x_1_chroma.shape[0]
#onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
onsets_frames_strength = librosa.onset.onset_strength(y, sr=sr)
times = librosa.frames_to_time(np.arange(len(onsets_frames_strength)), sr=sr)
onset_frames = librosa.onset.onset_detect(onset_envelope=onsets_frames_strength, sr=sr)
onsets_frames_strength_max = []
if len(onset_frames) <= 5:
    #onset_frames = [i for i in range(len(onsets_frames_strength)) if onsets_frames_strength[i]/np.max(onsets_frames_strength) > 0.1]
    last_index = -5
    onset_frames = []
    for i in range(len(onsets_frames_strength)):
        if onsets_frames_strength[i]/np.max(onsets_frames_strength) > 0.05 and i - last_index > 3 :
            onset_frames.append(i)
            last_index = i
            onsets_frames_strength_max.append(onsets_frames_strength[i])
    # onsets_frames_strength_max = find_n_largest(onsets_frames_strength_max, 10)
    # onsets_frames_strength_max = np.array(onsets_frames_strength_max)
    # onset_frames = librosa.onset.onset_detect(onset_envelope=onsets_frames_strength_max, sr=sr)

onset_times = librosa.frames_to_time(onset_frames, sr=sr)
plt.vlines(onset_times, 0, h, color='r', linestyle='--')
onset_samples = librosa.time_to_samples(onset_times)
print(onset_samples)
#plt.subplot(len(onset_times),1,1)
plt.show()

for i in range(0, len(onset_times)):
    start = onset_samples[i] - sr/2
    if start < 0:
        start =0
    end = onset_samples[i] + sr/2

    #y2 = [x if i> start and i<end else 0 for i,x in enumerate(y)]
    tmp = [x if i> start and i<end else 0 for i,x in enumerate(y) ]
    y2 = np.zeros(len(tmp))
    middle = int(len(y2)/2)
    offset = middle - onset_samples[i]
    for j in range(len(tmp)):
        if tmp[j] > 0:
            y2[j + offset] = tmp[j]
    #y2 = [tmp[i + offset] for i in range(len(tmp)) if tmp[i]>0]
    #y2 = [0.03 if i> start and i<end else 0.02 for i,x in enumerate(y)]
    #y2[int(len(y2) / 2)] = np.max(y)  # 让图片展示归一化
    y2 = np.array(y2)
    print("len(y2) is {}".format(len(y2)))

    print("(end - start)*sr is {}".format((end - start)*sr))
    #plt.subplot(len(onset_times),1,i+1)
    #y, sr = librosa.load(filename, offset=2.0, duration=3.0)
    #librosa.display.waveplot(y2, sr=sr)
    n_fft = 441
    hop_size = 220
    x_1_chroma = librosa.feature.chroma_stft(y=y2, sr=sr, tuning=0, norm=2, hop_length=hop_size, n_fft=n_fft)
    plt.title('Chroma Representation of $X_1$')
    librosa.display.specshow(x_1_chroma, x_axis='time', y_axis='chroma', cmap='gray_r', hop_length=hop_size)

    # CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref=np.max)
    # # librosa.display.specshow(CQT)
    # librosa.display.specshow(CQT, y_axis='cqt_note', x_axis='time')

    t = librosa.samples_to_time([middle], sr=sr)
    h = x_1_chroma.shape[0]
    plt.vlines(t, 0, h, color='r', linestyle='--') # 标出节拍位置
    fig = matplotlib.pyplot.gcf()
    #fig.set_size_inches(4, 4)
    if "." in filename:
        Filename = filename.split(".")[0]
    plt.axis('off')
    plt.axes().get_xaxis().set_visible(False)
    plt.axes().get_yaxis().set_visible(False)
    plt.savefig(savepath + str(i+1) + '.jpg', bbox_inches='tight', pad_inches=0)
    plt.clf()
#plt.show()