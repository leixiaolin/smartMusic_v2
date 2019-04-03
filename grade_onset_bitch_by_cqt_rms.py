# -*- coding: utf-8 -*-

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
import tensorflow as tf
import numpy as np
import pdb
import cv2
import os
import glob
import slim.nets.alexnet as alaxnet
import os
from create_tf_record import *
from cqt_rms import *
import tensorflow.contrib.slim as slim



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




if __name__ ==  '__main__':

    # filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏1.3(95).wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏1（二）(100).wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏1_40227（100）.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏1林(70).wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏2_40314（100）.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏2_40409（98）.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏2林(25).wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏2语(85).wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏10_40411（85）.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏10-04（80）.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏九（2）（95）.wav'


    savepath = './single_notes/data/test/'



    '''
    测试多个文件
    '''
    dir_list = ['F:/项目/花城音乐项目/样式数据/3.06MP3/节奏/']
    #dir_list = ['F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/']
    #dir_list = []
    total_accuracy = 0
    total_num = 0
    result_path = 'e:/test_image/t/'
    for dir in dir_list:
        file_list = os.listdir(dir)
        #file_list = ['节奏1（二）(100).wav','节奏1（四）(100).wav','节奏1（三）(95).wav']
        for filename in file_list:
            # wavname = re.findall(pattern,filename)[0]
            print(dir + filename)
            y, sr = librosa.load(dir + filename)
            CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref=np.max)
            onset_frames_cqt,best_threshold = get_onsets_by_cqt_rms_optimised(dir + filename)
            onset_frames_cqt_time = librosa.frames_to_time(onset_frames_cqt, sr=sr)

            type_index = get_onsets_index_by_filename(dir + filename)
            total_frames_number = get_total_frames_number(dir + filename)
            # 标准节拍时间点
            base_frames = onsets_base_frames(codes[type_index], total_frames_number)
            base_frames_time = librosa.frames_to_time(base_frames, sr=sr)
            min_d, best_y, onsets_frames = get_dtw_min(onset_frames_cqt, base_frames, 65)
            base_onsets = librosa.frames_to_time(best_y, sr=sr)

            # librosa.display.specshow(CQT)
            plt.figure(figsize=(10, 6))
            plt.subplot(3, 1, 1)  # 要生成两行两列，这是第一个图plt.subplot('行','列','编号')
            # plt.colorbar(format='%+2.0f dB')
            # plt.title('Constant-Q power spectrogram (note)')
            librosa.display.specshow(CQT, y_axis='cqt_note', x_axis='time')
            plt.vlines(onset_frames_cqt_time, 0, sr, color='y', linestyle='solid')
            # plt.vlines(base_onsets, 0,sr, color='r', linestyle='solid')

            # print(plt.figure)

            plt.subplot(3, 1, 2)  # 要生成两行两列，这是第一个图plt.subplot('行','列','编号')
            librosa.display.waveplot(y, sr=sr)
            plt.vlines(onset_frames_cqt_time, -1 * np.max(y), np.max(y), color='y', linestyle='solid')

            plt.subplot(3, 1, 3)

            rms = librosa.feature.rmse(y=y)[0]
            rms = [x / np.std(rms) for x in rms]
            max_rms = np.max(rms)
            # rms = np.diff(rms)
            times = librosa.frames_to_time(np.arange(len(rms)))
            rms_on_onset_frames_cqt = [rms[x] for x in onset_frames_cqt]
            min_rms_on_onset_frames_cqt = np.min(rms_on_onset_frames_cqt)
            # rms = [1 if x >=min_rms_on_onset_frames_cqt else 0 for x in rms]
            plt.plot(times, rms)
            # plt.axhline(min_rms_on_onset_frames_cqt)
            plt.axhline(max_rms * best_threshold)
            # plt.vlines(onsets_frames_rms_best_time, 0,np.max(rms), color='y', linestyle='solid')
            plt.vlines(onset_frames_cqt_time, 0, np.max(rms), color='r', linestyle='solid')
            plt.xlim(0, np.max(times))
            plt.savefig(result_path + filename.split(".wav")[0]+'.jpg', bbox_inches='tight', pad_inches=0)
            plt.clf()


