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

#COOKED_DIR = 'F:\项目\花城音乐项目\样式数据\音乐样本2019-01-29\节奏二\\'
plotpath = 'F:\\specgram\\'
#savepath = 'F:\\mfcc_pic\\2\\'

score = 0
path_index = np.array(['一','二','三','四','五','六','七','八','九','十'])
def load_and_trim(path):
    audio, sr = librosa.load(path)
    energy = librosa.feature.rmse(audio)
    frames = np.nonzero(energy >= np.max(energy) / 5)
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
for i in range(1,11):
    COOKED_DIR = 'F:/项目/花城音乐项目/样式数据/ALL/节奏/节奏'+ path_index[i-1] + '\\'
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
            else:
                index = index + 1
            path_one = COOKED_DIR + filename
            y, sr = load_and_trim(path_one)
            # onsets_frames = librosa.onset.onset_detect(y)
            # #D = librosa.stft(y)
            # # librosa.display.specshow(librosa.amplitude_to_db(D))
            # plt.vlines(onsets_frames, 0, sr, color='b', linestyle='solid')
            # plt.vlines(base_onsets, 0, sr, color='r', linestyle='dashed')
            CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref=np.max)
            librosa.display.specshow(CQT)
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
            saveFileName = str(len(file_sum)+1) + '-' + saveFileName
            plt.savefig(savepath + saveFileName + '.jpg',  bbox_inches='tight', pad_inches=0)
            plt.clf()
            saveFileName = ''
            '''
            infile = plotpath + Filename + '.jpg'

            outfile = savepath + saveFileName + '.jpg'
            im = Image.open(infile).convert("L")
            (x, y) = im.size  # read image size
            x_s = 224  # define standard width
            y_s = 224  # calc height based on standard width
            out = im.resize((x_s, y_s), Image.ANTIALIAS)  # resize image with high-quality
            out.save(outfile)
            print
            'original size: ', x, y
            print
            'adjust size: ', x_s, y_s
            '''

