import  numpy as np
import librosa
import matplotlib.pyplot as plt
import random
import os

def augmention(path,name,save_path):
    file_path = path + name + '.wav'
    y, sr = librosa.load(file_path)

    # 产生4个不相等的随机数
    shifts = []
    while(len(shifts)<1):
        x = random.uniform(1,1.3)
        x =round(x,2)
        if x not in shifts:
            shifts.append(x)
    while (len(shifts) < 1):
        x = random.uniform(0.7,1)
        x = round(x, 2)
        if x not in shifts:
            shifts.append(x)

    for shift in shifts:
        new_file = str(shift) + '-shift-' +  name
        save_path_file = save_path + new_file + '.wav'
        # 通过改变采样率来改变音速，相当于播放速度X2
        sr_shift = int(sr * shift)
        librosa.output.write_wav(save_path_file, y, sr_shift)

if __name__ == '__main__':
    path = 'F:/项目/花城音乐项目/样式数据/ALL/旋律/1.31MP3/'
    name = '旋律1.100分'
    #save_path = 'F:/项目/花城音乐项目/参考代码/tensorflow_models_nets-master/onsets/train/E/'
    save_path = 'F:/项目/花城音乐项目/参考代码/tensorflow_models_nets-master/raw_data/onsets/D/'
    #result = augmention(path,name,save_path)

    path_index = np.array(['一','二','三','四','五','六','七','八','九','十'])

    # 清空之前增扩的数据
    for i in range(1,2):
        #COOKED_DIR = 'F:/项目/花城音乐项目/样式数据/ALL/节奏/节奏' + path_index[i - 1] + '/'
        COOKED_DIR = 'F:/项目/花城音乐项目/参考代码/tensorflow_models_nets-master/raw_data/onsets/D/'
        # savepath = 'F:\\mfcc_pic\\'+ str(i) +'\\'
        for root, dirs, files in os.walk(COOKED_DIR):
            print("Root = ", root, "dirs = ", dirs, "files = ", files)

            index = 0
            for filename in files:
                print(filename)
                if filename.find('shift') > 0:
                    os.remove(root + filename)
    # 重新增扩数据
    for i in range(1,2):
        # COOKED_DIR = 'F:/项目/花城音乐项目/样式数据/ALL/节奏/节奏' + path_index[i - 1] + '/'
        COOKED_DIR = 'F:/项目/花城音乐项目/参考代码/tensorflow_models_nets-master/raw_data/onsets/D/'
        # savepath = 'F:\\mfcc_pic\\'+ str(i) +'\\'
        for root, dirs, files in os.walk(COOKED_DIR):
            print("Root = ", root, "dirs = ", dirs, "files = ", files)

            index = 0
            for filename in files:
                print(filename)
                augmention(root, filename.split('.wav')[0], root)
