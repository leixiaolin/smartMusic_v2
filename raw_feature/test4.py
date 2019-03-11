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

savepath = 'e:/test_image/'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/视唱1-02（90）.wav'

#y, sr = librosa.load(filename)
y, sr = load_and_trim(filename)
CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref = np.max)
#librosa.display.specshow(CQT)
librosa.display.specshow(CQT, y_axis='cqt_note',x_axis='time')
onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
onset_times = librosa.frames_to_time(onset_frames, sr=sr)
plt.vlines(onset_times, 0, y.max(), color='r', linestyle='--')
onset_samples = librosa.time_to_samples(onset_times)
print(onset_samples)
#plt.subplot(len(onset_times),1,1)
plt.show()

plt.figure(figsize=(5,80))
for i in range(0, len(onset_times)):
    start = onset_samples[i] - sr/2
    if start < 0:
        start =0
    end = onset_samples[i] + sr/2
    #y2 = [x if i> start and i<end else 0 for i,x in enumerate(y)]
    y2 = [x for i,x in enumerate(y) if i> start and i<end]
    y2[int(len(y2) / 2)] = np.max(y)  # 让图片展示归一化
    t = librosa.samples_to_time([int(len(y2) / 2)], sr=sr)
    plt.vlines(t, -1, 1, color='r', linestyle='--')
    #y2 = [0.03 if i> start and i<end else 0.02 for i,x in enumerate(y)]
    y2 = np.array(y2)
    print("len(y2) is {}".format(len(y2)))

    print("(end - start)*sr is {}".format((end - start)*sr))

    plt.subplot(len(onset_times),1,i+1)
    #y, sr = librosa.load(filename, offset=2.0, duration=3.0)
    CQT = librosa.amplitude_to_db(librosa.cqt(y2, sr=16000), ref=np.max)
    # librosa.display.specshow(CQT)
    librosa.display.specshow(CQT, y_axis='cqt_note', x_axis='time')

plt.show()