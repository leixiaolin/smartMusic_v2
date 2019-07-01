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

def clear_dir(dis_dir):
    shutil.rmtree(dis_dir)
    os.mkdir(dis_dir)

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

savepath = 'e:/test_image/t/'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏1.3(95).wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏1（二）(100).wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏1_40227（100）.wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏1林(70).wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏2_40314（100）.wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏2_40409（98）.wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏2林(25).wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏2语(85).wav'
#filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/节奏/节2罗（75）.wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/视唱1-01（95）.wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律二（11）（60）.wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/视唱1-02（90）.wav'




y, sr = librosa.load(filename)
rms = librosa.feature.rmse(y=y)[0]
time = librosa.get_duration(filename=filename)
total_frames_number = len(rms)
print("time is {}".format(time))
CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref = np.max)
librosa.display.specshow(CQT, y_axis='cqt_note',x_axis='time')
w,h = CQT.shape
#onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
onset_frames = get_real_onsets_frames_rhythm(y,modify_by_energy=False)

onset_times = librosa.frames_to_time(onset_frames, sr=sr)
plt.vlines(onset_times, 0,sr, color='y', linestyle='--')
onset_samples = librosa.time_to_samples(onset_times)
print(onset_samples)
#plt.subplot(len(onset_times),1,1)
plt.show()

clear_dir(savepath)
for i in range(0, len(onset_frames)):
    half = 15
    start = onset_frames[i] - half
    if start < 0:
        start =0
    end = onset_frames[i] + half
    if end >=total_frames_number:
        end = total_frames_number - 1
    #y2 = [x if i> start and i<end else 0 for i,x in enumerate(y)]
    CQT_sub = np.zeros(CQT.shape)
    middle = int(h / 2)
    offset = middle - onset_frames[i]
    for j in range(int(start), int(end)):
        CQT_sub[:, j + offset] = CQT[:, j]
    #CQT = CQT_T
    librosa.display.specshow(CQT_sub, y_axis='cqt_note', x_axis='time')
    #y2 = [x for i,x in enumerate(y) if i> start and i<end]
    #y2 = [0.03 if i> start and i<end else 0.02 for i,x in enumerate(y)]
    #y2[int(len(y2) / 2)] = np.max(y)  # 让图片展示归一化
    t = librosa.frames_to_time([middle], sr=sr)
    plt.vlines(t, 0,sr, color='y', linestyle='--')# 标出节拍位置
    #y2 = np.array(y2)
    #print("len(y2) is {}".format(len(y2)))

    print("(end - start)*sr is {}".format((end - start)*sr))
    #plt.show()
    #plt.subplot(len(onset_times),1,i+1)
    #y, sr = librosa.load(filename, offset=2.0, duration=3.0)
    #librosa.display.waveplot(y2, sr=sr)
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