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
    # savepath = 'e:/test_image/'

    savepath = '../single_notes/data/test/'
    labels_filename = '../single_onsest/data/label.txt'
    models_path = '../single_notes/models/alex/model.ckpt-10000'

    # 清空文件夹

    # if not os.path.exists(image_dir):
    #     os.mkdir(image_dir)
    # clear_dir(image_dir)

    '''
    图片处理
    一次性自动生成所有图片
    '''
    # dir_list = ['./mp3/2.18WAV/','./mp3/2.27WAV/']
    # num = 200
    # process_all_pic(dir_list,num)


    '''
    测试单个文件
    '''
    import re
    image_dir = '../single_notes/data/test/'
    filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏1（三）(95).wav'
    #filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏1.2(100).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/节奏/节1.1(100).wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/视唱1-02（90）.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/视唱1-01（95）.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1.1(96).wav'

    clear_dir(image_dir)
    pattern = 'WAV/(.+)'
    #wavname = re.findall(pattern,filename)[0]
    wavname = ''
    curr_num = 1
    onset_frames, curr_num = get_single_notes(filename, curr_num,savepath,True)
    onsets_frames_strength = []
    #print("onset_frames,onsets_frames_strength is {},{}".format(onset_frames))
    if onset_frames:
        #onsets, onsets_strength,_ = predict(wavname,image_dir,onset_frames,onsets_frames_strength,models_path)
        onsets_strength = []
        print("onsets, onsets_strength is {},{}".format(onset_frames, onsets_strength))
        y, sr = librosa.load(filename)
        onset_frames = del_note_end_by_cqt(y,onset_frames)
        onset_frames = del_note_middle_by_cqt(y,onset_frames)
        if len(onset_frames) > 0:
            min_width = get_min_width_onsets(filename)
            print("min_width is {}".format(min_width))
            onset_frames = del_overcrowding(onset_frames, min_width/3)
        #librosa.display.waveplot(y, sr=sr)
        CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref=np.max)
        w, h = CQT.shape
        CQT[50:w, :] = np.min(CQT)
        CQT[0:20, :] = np.min(CQT)
        for i in range(w):
            for j in range(h):
                if CQT[i][j] > -20:
                    CQT[i][j] = np.max(CQT)
                # else:
                #     CQT[i][j] = np.min(CQT)
        librosa.display.specshow(CQT, y_axis='cqt_note', x_axis='time')
        onsets_time = librosa.frames_to_time(onset_frames, sr=sr)
        onset_frames_time = librosa.frames_to_time(onset_frames,sr = sr)
        #plt.vlines(onset_frames_time,0,sr, color='b', linestyle='dashed')
        plt.vlines(onsets_time, 0,sr, color='r', linestyle='solid')
        print("onsets, onsets_strength is {},{}".format(onset_frames, onsets_strength))
        plt.show()


    '''
    测试多个文件
    '''
    dir_list = ['F:/项目/花城音乐项目/样式数据/3.06MP3/节奏/']
    #dir_list = []
    total_accuracy = 0
    total_num = 0
    result_path = 'e:/test_image/t/'
    #clear_dir(result_path)
    # 要测试的数量
    test_num = 100
    for dir in dir_list:
        file_list = os.listdir(dir)
        #file_list = ['旋1.1(96).wav']
        for filename in file_list:
            clear_dir(image_dir)
            pattern = 'WAV/(.+)'
            # wavname = re.findall(pattern,filename)[0]
            print(dir + filename)
            wavname = ''
            curr_num = 1
            onset_frames, curr_num = get_single_notes(dir + filename, curr_num,savepath,modify_by_energy=True)
            onsets_frames_strength = []
            print("onset_frames,onsets_frames_strength is {},{}".format(onset_frames, onsets_frames_strength))
            if onset_frames:
                onsets, onsets_strength, _ = [],[],[]
                print("onsets, onsets_strength is {},{}".format(onset_frames, onsets_strength))
                y, sr = librosa.load(dir + filename)
                # onset_frames = del_note_end_by_cqt(y, onset_frames)
                # onset_frames = del_note_middle_by_cqt(y, onset_frames)
                #onset_frames = get_note_start_by_cqt(y, onset_frames)
                if len(onset_frames) > 0:
                    #min_width = get_min_width_onsets(dir + filename)
                    min_width = 2
                    print("min_width is {}".format(min_width))
                    onset_frames = del_overcrowding(onset_frames, min_width)
                # librosa.display.waveplot(y, sr=sr)
                CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref=np.max)
                w, h = CQT.shape
                CQT[50:w, :] = np.min(CQT)
                CQT[0:20, :] = np.min(CQT)
                for i in range(w):
                    for j in range(h):
                        if CQT[i][j] > -20:
                            CQT[i][j] = np.max(CQT)
                        # else:
                        #     CQT[i][j] = np.min(CQT)
                librosa.display.specshow(CQT, y_axis='cqt_note', x_axis='time')
                onsets_time = librosa.frames_to_time(onset_frames, sr=sr)
                onset_frames_time = librosa.frames_to_time(onset_frames, sr=sr)
                #plt.vlines(onset_frames_time, 0, sr, color='b', linestyle='dashed')
                plt.vlines(onsets_time, 0, sr, color='r', linestyle='solid')
                plt.savefig(result_path + filename.split(".wav")[0]+'.jpg', bbox_inches='tight', pad_inches=0)
                plt.clf()
