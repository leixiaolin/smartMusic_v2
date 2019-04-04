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

savepath = './test/jpg/'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏1.3(95).wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏1（二）(100).wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏1_40227（100）.wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏1林(70).wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏2_40314（100）.wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏2_40409（98）.wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏2林(25).wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏2语(85).wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏10_40411（85）.wav'

raw_path = './test/wav/'
filenames =os.listdir(raw_path)


print(filenames)

for i,filename in enumerate(filenames):
    y, sr = librosa.load(raw_path+filename)
    librosa.display.waveplot(y, sr=sr)
    #onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    onset_frames,onsets_frames_strength = get_onsets_by_all(y,sr)

    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    plt.vlines(onset_times, 0, y.max(), color='r', linestyle='--')
    onset_samples = librosa.time_to_samples(onset_times)
    print(onset_samples)
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(4, 4)
    if "." in filename:
        Filename = filename.split(".")[0]
    plt.axis('off')
    plt.axes().get_xaxis().set_visible(False)
    plt.axes().get_yaxis().set_visible(False)
    plt.savefig(savepath + str(i + 1) + '.jpg', bbox_inches='tight', pad_inches=0)
    print('hello')
    plt.clf()
    #plt.subplot(len(onset_times),1,1)
    # plt.show()


    # for i in range(0, len(onset_times)):
    #     start = onset_samples[i] - sr/2
    #     if start < 0:
    #         start =0
    #     end = onset_samples[i] + sr/2
    #     #y2 = [x if i> start and i<end else 0 for i,x in enumerate(y)]
    #     y2 = [x for i,x in enumerate(y) if i> start and i<end]
    #     #y2 = [0.03 if i> start and i<end else 0.02 for i,x in enumerate(y)]
    #     y2[int(len(y2) / 2)] = np.max(y)  # 让图片展示归一化
    #     t = librosa.samples_to_time([onset_samples[i]-start], sr=sr)
    #     plt.vlines(t, -1*np.max(y), np.max(y), color='r', linestyle='--') # 标出节拍位置
    #     y2 = np.array(y2)
    #     print("len(y2) is {}".format(len(y2)))
    #
    #     print("(end - start)*sr is {}".format((end - start)*sr))
    #     #plt.subplot(len(onset_times),1,i+1)
    #     #y, sr = librosa.load(filename, offset=2.0, duration=3.0)
    #     librosa.display.waveplot(y2, sr=sr)
    #     fig = matplotlib.pyplot.gcf()
    #     fig.set_size_inches(4, 4)
    #     if "." in filename:
    #         Filename = filename.split(".")[0]
    #     plt.axis('off')
    #     plt.axes().get_xaxis().set_visible(False)
    #     plt.axes().get_yaxis().set_visible(False)
    #     plt.savefig(savepath + str(i+1) + '.jpg', bbox_inches='tight', pad_inches=0)
    #     print('hello')
    #     plt.clf()
#plt.show()