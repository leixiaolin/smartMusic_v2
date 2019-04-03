
# coding: utf-8

# In[1]:


import sys
sys.path.append("..")
sys.path.append(".")
import os
import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
from create_base import *
from myDtw import *
from find_mismatch import *
from filters import *
from vocal_separation import *


codes = np.array(['[1000,1000;2000;1000,500,500;2000]',
                  '[2000;1000,1000;500,500,1000;2000]',
                  '[1000,1000;500,500,1000;1000,1000;2000]',
                  '[1000,--(1000);1000,--(1000);500,250,250,1000;--(1000),1000]',
                  '[500;1000,500,1000,500;500,500,500,250,250,500,500;250,250,500,500,1000]',
                  '[1000,--(1000);1000,--(1000);1000,-(500),500;1000,1000]',
                  '[750,250,500,500,500,-(500);500,1000,500,500,-(500);750,250,500,500,500,-(500)]',
                  '[500,1000,500,500,250,250;1000,500,750,250,500;3000]',
                  '[500,500,500;1000,500;500,500,500;1500;500,500,500;1000,500;500;1000;1500]',
                  '[500,500,1000;500,500;1000;375,125,250,250,375,125,250,250;500,500,1000]',
                  '[500,500,1000;l500,500,1000;500,500,750,250;2000]',
                  '[1000,1000;500,500,1000;1000,500,500;2000]',
                  '[1000,1000;500,500,1000;500,250,250,250;2000]',
                  '[500,1000,500;250,250,250,250,500,500;500,500,500,500;2000]'])
# 1. Get the file path to the included audio example
# Sonify detected beat events




filepath = 'F:\项目\花城音乐项目\样式数据\音乐样本2019-01-29\节奏九\\'
# filename = 'F:/项目/花城音乐项目/样式数据/ALL/节奏/节奏八/节奏八（标准音频）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/ALL/节奏/节奏八/节奏8.100分.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/ALL/旋律/1.31MP3/旋律1.100分.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/ALL/旋律/1.31MP3/旋律2.100分.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/ALL/节奏/节奏八/节奏八（1）(90).wav'
#filename = 'F:/项目/花城音乐项目/样式数据/ALL/节奏/节奏八/节奏八（2）（90分）.wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏4卢(65).wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏2-01（80）.wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏4-02（68）.wav'
filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/节奏/节奏二（4）（100）.wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏7-02（30）.wav'
filename = './single_onsets/样式数据/WAV/1.31/节奏9.65分.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/视唱1-02（90）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律2（四）(96).wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律1.1(95).wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律2.1(80).wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律2.3(55).wav'
#filename = 'F:/项目/花城音乐项目/样式数据/3.19MP3/节奏/节奏六1(10).wav'

#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律二（10）（75）.wav'
# filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律二（8）（100）.wav'
# filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律7_40218（20）.wav'
# filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律一（9）（100）.wav'
# filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律一（14）（95）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/节奏/节1罗（90）.wav'

# 2. Load the audio as a waveform `y`
#    Store the sampling rate as `sr`


# In[3]:

def detect_repeat_onset(filename,old_way,new_way):

    y,sr = librosa.load(old_way+filename)
    melspec = librosa.feature.melspectrogram(y, sr, n_fft=1024, hop_length=512)
    logmelspec = librosa.power_to_db(melspec)

    # In[4]:

    o_env = librosa.onset.onset_strength(y, sr=sr)
    times = librosa.frames_to_time(np.arange(len(o_env)), sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr)

    # ## 正常的CQT图，以及librosa的onset_frame

    # In[5]:

    melspec = librosa.feature.melspectrogram(y, sr, n_fft=1024, hop_length=512, n_mels=128)
    # convert to log scale
    logmelspec = librosa.power_to_db(melspec)
    logmelspec.shape
    CQT = librosa.power_to_db(melspec, ref=np.max)
    print(CQT.shape)
    print(CQT[:, 1].shape)
    print(onset_frames)

    # ## 去伪

    # ### ①计算CQT相邻帧之间的欧式距离，并正则化CQT
    # In[6]:

    dist = np.zeros((CQT.shape[1] - 1, 1))

    a = CQT[:, 1]
    b = CQT[:, 0]
    for i in range(dist.shape[0]):
        dist[i] = np.linalg.norm(CQT[:, i + 1] - CQT[:, i])

    # #### 归一化

    # In[7]:

    ##
    Normalized_CQT = CQT
    Normalized_CQT = Normalized_CQT / np.max(abs(CQT))
    Normalized_CQT

    # #### 设置阀值
    # In[8]:

    threhold = -0.5  ##低于-0.5的置为0
    for i in range(Normalized_CQT.shape[0]):
        for j in range(Normalized_CQT.shape[1]):
            if Normalized_CQT[i][j] < threhold:
                Normalized_CQT[i][j] = -1

    # #### 正则化后的CQT，很明显对比度更强了

    # In[9]:

    # ### ②把整个正则化后的cqt每一帧求取(竖直方向)求平均值

    # #### -1是那些黑色的区间，我们要做的就是分开这些区间

    average_CQT = []
    for i in range(CQT.shape[1]):
        average_CQT.append(np.sum(Normalized_CQT[:, i]) / CQT.shape[0])
    print(np.mean(average_CQT))
    average_CQT = np.array(average_CQT)
    average_CQT = np.array(
        [average_CQT[i] if average_CQT[i] >= np.mean(average_CQT) else -1 for i in range(average_CQT.shape[0])])

    itemindex = np.argwhere(average_CQT == -1)
    print(average_CQT)
    print(average_CQT.shape)

    # #### 判断黑色(-1连续)区间
    #
    # In[11]:

    space_index = []
    i = 1
    index = 0
    start_index = 0
    end_index = 0
    while (i < average_CQT.shape[0]):
        if (average_CQT[i] == -1):
            if (i != 0 and average_CQT[i - 1] != -1):  ##如果不是第一个，且前面一个帧不是-1，那么从当前点开始
                start_index = i
            elif (i == 0):  ##如果是第一个，起点为0
                start_index = 0
            if (i + 1 != average_CQT.shape[0]):  ##不是最后一个
                if (average_CQT[i + 1] != -1):
                    end_index = i
                    space_index.append((start_index, end_index))  ##终点
            elif (i + 1 == average_CQT.shape[0]):  ##结尾判定
                if (average_CQT[i] == -1):
                    end_index = i
                    space_index.append((start_index, end_index))
        i = i + 1

    # #### 初步检测街拍间的间距

    # In[12]:

    onset_space = [(space_index[i][1] + 1, space_index[i + 1][0] - 1) for i in range(len(space_index) - 1)]
    # onset_space.append(space_index[-1][0]-space_index[-2][1])
    # onset_space##表示区间范围

    # #### 判断-1区间间隔大小
    #

    # In[13]:

    ##判断空白间隔之间的距离
    exclude = []
    blank_space = [x[1] - x[0] + 1 for x in space_index]
    # print(len(blank_space))
    for i in range(len(blank_space)):
        if (i < len(blank_space) - 1):
            if (blank_space[i] <= 5 and space_index[i + 1][0] - space_index[i][1] < 5):  ##间隔太小了，而且前后两个黑色区间的间隔也小于一定区间
                exclude.append(i)  ##去除改帧

    # print(exclude)

    # In[14]:

    space_index = np.delete(space_index, exclude)  ##间隔，去掉了

    # In[15]:

    # ### ③检测节拍点
    #

    # In[16]:

    detect_onset = []
    # detect_onset = [x+1 for x in space_index]
    for x in space_index:  ##每一个符合要求的-1大区间的起点+1
        k = x + 1
        while (k < average_CQT.shape[0] - 1 and average_CQT[k] < average_CQT[k + 1]):  ##类似贪心，如果后面有一直递增的，渠道这个点
            k += 1
        detect_onset.append(k)
    # In[17]:

    detect_onset = np.unique(detect_onset)

    # #### 去除间隔相近的节拍，循环5次的原因是有一些相近而且比较多的伪街拍聚集在一起，一次可能无法完全去除

    # In[18]:

    for i in range(5):
        mean_diff = np.mean(np.diff(detect_onset))
        min_diff = np.min(np.diff(detect_onset))
        print(min_diff)
        if min_diff < 10:
            for i in range(detect_onset.shape[0] - 1):
                if detect_onset[i + 1] - detect_onset[i] <= 10 and detect_onset[i + 1] < detect_onset.shape[0]:
                    if average_CQT[detect_onset[i + 1]] - average_CQT[detect_onset[i + 1] - 1] > average_CQT[
                        detect_onset[i]] - average_CQT[detect_onset[i] - 1]:  ##从梯度来考虑
                        detect_onset[i] = detect_onset[i + 1]
                    else:
                        detect_onset[i + 1] = detect_onset[i]

                        #         if detect_onset[-1]-detect_onset[-2]<=10:
    #             if average_CQT[detect_onset[-1]]-average_CQT[detect_onset[-1]-1]>average_CQT[detect_onset[-2]]-average_CQT[detect_onset[-2]-1]:##从梯度来考虑
    #                 detect_onset[-2]=detect_onset[-1]
    #             else:
    #                 detect_onset[-1]=detect_onset[-2]
    detect_onset = np.unique(detect_onset)  ##去重


    # In[20]:

    # ### 调整最佳帧的位置，不断往峰值走,类似贪心

    # In[21]:

    # detect_onset = [x[0] for x in onset_space]
    onset_space = detect_onset
    detect_onset = []
    for x in onset_space:
        k = x + 1
        if (k < average_CQT.shape[0] - 1):
            while (k < average_CQT.shape[0] - 1 and average_CQT[k] <= (average_CQT[k + 1]) and average_CQT[k] -
                   average_CQT[k - 1] >= average_CQT[k + 1] - average_CQT[k]):
                k += 1
            detect_onset.append(k)

    # In[22]:

    detect_onset = np.unique(detect_onset)

    # ### 使用rms来判断

    # In[23]:

    # #### 类似于上面的cqt的贪心操作，有利于去伪和调整最佳位置

    # In[24]:

    # detect_onset = [x[0] for x in onset_space]
    rms = librosa.feature.rmse(y)[0]
    onset_space = detect_onset
    detect_onset = []
    for x in onset_space:
        k = x + 1
        if (k < rms.shape[0]):
            while (k < rms.shape[0] - 1 and rms[k] <= (rms[k + 1]) and rms[k] - rms[k - 1] >= rms[k + 1] - rms[k]):
                k += 1
            print(k)
            detect_onset.append(k)

    # In[25]:

    detect_onset = np.unique(detect_onset)

    # In[26]:

    # #### 由于上面的操作使得检测出来的街拍在rms的峰值了，接下来排序，低于均值2/3的去除

    # In[27]:

    onset_rms = rms[detect_onset]
    mean_onset_rms = np.mean(onset_rms) * 1 / 2  ##低于2/3的去除
    new_onset = []
    for x in detect_onset:

        if rms[x] >= mean_onset_rms:
            new_onset.append(x)

    # In[28]:

    new_onset = np.unique(new_onset)  ##去重

    # #### 最后一步
    #

    # In[30]:

    min_diff = np.min(np.diff(new_onset))
    print(min_diff)
    if min_diff < 10:
        for i in range(new_onset.shape[0] - 1):
            if new_onset[i + 1] - new_onset[i] <= 10:
                if rms[new_onset[i + 1]] - rms[new_onset[i + 1] - 1] > rms[new_onset[i]] - rms[
                    new_onset[i] - 1]:  ##从梯度来考虑
                    new_onset[i] = new_onset[i + 1]
                else:
                    new_onset[i + 1] = new_onset[i]

        if new_onset[-1] - new_onset[-2] <= 10:
            if rms[new_onset[-1]] - rms[new_onset[-1] - 1] > rms[new_onset[-2]] - rms[new_onset[-2] - 1]:  ##从梯度来考虑
                new_onset[-2] = new_onset[-1]
            else:
                new_onset[-1] = new_onset[-2]

                # In[31]:

    new_onset = np.unique(new_onset)  ##去重


    librosa.display.waveplot(y, sr=sr)
    plt.vlines(librosa.frames_to_time(new_onset), -np.max(y), np.max(y), color='r', linestyle='dashed')

    savename = filename.split('.wav')[0]
    plt.savefig(new_way + savename + ".png")
    plt.show()


if __name__=='__main__':
    fold = './single_onsets/样式数据/WAV/3.06/'
    save = './single_onsets/样式数据/WAV/test/'
    for file in os.listdir(fold):
        print(file)
        detect_repeat_onset(file,fold,save)
