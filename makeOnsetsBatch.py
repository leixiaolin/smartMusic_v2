import numpy, wave,matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import librosa
import librosa.display
from PIL import Image
import re
import shutil
import os
from create_base import *
from create_labels_files import write_txt

#COOKED_DIR = 'F:\项目\花城音乐项目\样式数据\音乐样本2019-01-29\节奏二\\'
plotpath = 'F:\\specgram\\'
#savepath = 'F:\\mfcc_pic\\2\\'

score = 0
path_index = np.array(['一','二','三','四','五','六','七','八','九','十'])
codes = np.array(['[1000,1000;2000;1000,500,500;2000]',
                  '[2000;1000,1000;500,500,1000;2000]',
                  '[1000,1000;500,500,1000;1000,1000;2000]',
                  '[1000,--(1000);1000,--(1000);500,250,250,1000;--(1000),1000]',
                  '[500;1000,500,1000,500;500,500,500,250,250,500,500;250,250,500,500,1000]',
                  '[1000,--(1000);1000,--(1000);1000,-(500),500;1000,1000]',
                  '[750,250,500,500,500,-(500);500,1000,500,500,-(500);750,250,500,500,500,-(500)]',
                  '[500,1000,500,500,250,250;1000,500,750,250,500;3000]',
                  '[500,500,500;1000,500;500,500,500;1500;500,500,500;1000,500;500;1000;1500]',
                  '[500,500,1000;500,500;1000;375,125,250,250,375,125,250,250;500,500,1000]'])
def load_and_trim(path):
    audio, sr = librosa.load(path)
    energy = librosa.feature.rmse(audio)
    frames = np.nonzero(energy >= np.max(energy) / 10)
    indices = librosa.core.frames_to_samples(frames)[1]
    audio = audio[indices[0]:indices[-1]] if indices.size else audio[0:0]

    return audio, sr

tmp = ['A','B','C','D','E']
dis_dir = 'F:/项目/花城音乐项目/参考代码/tensorflow_models_nets-master/onsets/train'
scr_dir = 'F:/项目/花城音乐项目/参考代码/tensorflow_models_nets-master/onsets/val'
def clear_dir(dis_dir,scr_dir):
    for i in tmp:
        d_dir = dis_dir + '/' + i
        shutil.rmtree(d_dir)
        os.mkdir(d_dir)

        s_dir = scr_dir + '/' + i
        shutil.rmtree(s_dir)
        os.mkdir(s_dir)

#清空文件夹
clear_dir(dis_dir,scr_dir)

# 保存新文件名与原始文件的对应关系
files_list = []
new_old_txt = './onsets/new_and_old.txt'
for i in range(1,5):
    #COOKED_DIR = 'F:/项目/花城音乐项目/样式数据/ALL/节奏/节奏'+ path_index[i-1] + '\\'
    COOKED_DIR = 'F:/项目/花城音乐项目/参考代码/tensorflow_models_nets-master/raw_data/onsets/' + tmp[i - 1] + '/'
    #savepath = 'F:\\mfcc_pic\\'+ str(i) +'\\'
    for root, dirs, files in os.walk(COOKED_DIR):
        print("Root = ", root, "dirs = ", dirs, "files = ", files)

        base_onsets = []
        for filename in files:
            if "." in filename:
                Filename = filename.split(".")[0]
            if Filename.find('标准') > 0 or Filename.find('100') > 0:
                path_one = COOKED_DIR + filename
                y, sr = load_and_trim(path_one)
                base_onsets = librosa.onset.onset_detect(y)
                break

        index = 0
        for filename in files:
            print(filename)
            if filename.find('wav') <= 0:
                continue
            elif filename.find('shift') > 0:
                continue
            else:
                index = index + 1
            path_one = COOKED_DIR + filename
            print(path_one)
            y, sr = load_and_trim(path_one)
            # onsets_frames = librosa.onset.onset_detect(y)
            #
            # # 获取每个帧的能量
            # energy = librosa.feature.rmse(y)
            # energy_mean = np.mean(energy)  # 获取能量均值
            # energy_gap = energy_mean * 0.95
            # onsets_frames = [x for x in onsets_frames if energy[0][x] > energy_gap]  # 筛选能量过低的伪节拍点

            if len(y)>0:
                #onsets_frames = get_real_onsets_frames(y)
                onsets_frames = get_onsets_by_all(y,sr)
                if len(onsets_frames)<3:
                    continue

                print("onsets_frames is {}".format(onsets_frames))

                # 节拍时间点
                onstm = librosa.frames_to_time(onsets_frames, sr=sr)
                print("onstm is {}".format(onstm))
                duration = librosa.get_duration(y, sr=sr) # 获取音频时长
                print("duration is {}".format(duration))
                # 标准节拍时间点
                base_onsets = onsets_base(codes[i-1],duration,onstm[0])
                print(base_onsets)
                librosa.display.waveplot(y, sr=sr)
                #plt.show()
                #plt.vlines(onstm,  -1*np.max(y),np.max(y), color='b', linestyle='solid')
            plt.vlines(base_onsets[:-1],  -1*np.max(y),np.max(y), color='r', linestyle='dashed')
            plt.vlines(base_onsets[-1],  -1*np.max(y),np.max(y), color='white', linestyle='dashed')
            # CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref=np.max)
            # librosa.display.specshow(CQT)
            #plt.ylabel('Frequency')
            #plt.xlabel('Time(s)')
            #plt.show()
            fig = matplotlib.pyplot.gcf()
            fig.set_size_inches(4, 4)
            if "." in filename:
                Filename = filename.split(".")[0]
            plt.axis('off')
            plt.axes().get_xaxis().set_visible(False)
            plt.axes().get_yaxis().set_visible(False)
            #plt.rcParams['savefig.dpi'] = 300  # 图片像素
            #plt.figure(figsize=(10, 10))
            #plt.rcParams['figure.dpi'] = 300  # 分辨率
            if filename.find('标准') > 0:
                saveFileName = '100-A'
                savepath = 'F:/项目/花城音乐项目/参考代码/tensorflow_models_nets-master/onsets/train/A/'
            # elif filename.find('分') > 0:
            #     score = filename.split("分")[0].split(".")[1]  # 提取分值
            # elif filename.find('(') > 0:
            #     score = filename.split("(")[2].split(")")[0]  # 提取分值
            else:
                #score = filename.split("（")[2].split("）")[0]  # 提取分值
                score = re.sub("\D", "", filename)  # 筛选数字
            if str(score).find("100") > 0:
                score = 100
            else:
                score = int(score) % 100

            if int(score) >=90:
                grade = 'A'
                savepath = 'F:/项目/花城音乐项目/参考代码/tensorflow_models_nets-master/onsets/train/A/'
            elif int(score) >= 75:
                grade = 'B'
                savepath = 'F:/项目/花城音乐项目/参考代码/tensorflow_models_nets-master/onsets/train/B/'
            elif int(score) >=60:
                grade = 'C'
                savepath = 'F:/项目/花城音乐项目/参考代码/tensorflow_models_nets-master/onsets/train/C/'
            elif int(score) >=1:
                grade = 'D'
                savepath = 'F:/项目/花城音乐项目/参考代码/tensorflow_models_nets-master/onsets/train/D/'
            else:
                grade = 'E'
                savepath = 'F:/项目/花城音乐项目/参考代码/tensorflow_models_nets-master/onsets/train/E/'
            saveFileName = str(score) + '-' + grade
            file_sum = os.listdir(savepath)
            #saveFileName = str(len(file_sum)+1) + '-' + filename.split(".wav")[0] + '-' + saveFileName
            saveFileName = str(len(file_sum) + 1) + '-' + saveFileName
            plt.savefig(savepath + saveFileName + '.jpg',  bbox_inches='tight', pad_inches=0)
            plt.clf()
            files_list.append([savepath + saveFileName + '.jpg', filename])
            saveFileName = ''
write_txt(files_list, new_old_txt, mode='w')
