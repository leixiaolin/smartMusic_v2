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
    while(len(shifts)<2):
        x = random.randint(1,5)
        if x not in shifts:
            shifts.append(x)
    while (len(shifts) < 4):
        x = random.randint(-4,-1)
        if x not in shifts:
            shifts.append(x)

    for shift in shifts:
        # 通过移动音调变声，14是上移14个半步，如果是-14，则是下移14个半步
        b = librosa.effects.pitch_shift(y, sr, n_steps=shift)
        new_file = name + '-shift-' + str(shift)
        save_path_file = save_path + new_file + '.wav'
        librosa.output.write_wav(save_path_file, b, sr)

if __name__ == '__main__':
    path = 'F:/项目/花城音乐项目/样式数据/ALL/旋律/1.31MP3/'
    name = '旋律1.100分'
    save_path = 'F:/项目/花城音乐项目/参考代码/tensorflow_models_nets-master/onsets/train/E/'
    #result = augmention(path,name,save_path)

    path_index = np.array(['1.31MP3','2.2MP3','2.18MP3','2019-01-29'])

    # 清空之前增扩的数据
    for i in range(1,5):
        COOKED_DIR = 'F:/项目/花城音乐项目/样式数据/ALL/旋律/' + path_index[i - 1] + '/'
        # savepath = 'F:\\mfcc_pic\\'+ str(i) +'\\'
        for root, dirs, files in os.walk(COOKED_DIR):
            print("Root = ", root, "dirs = ", dirs, "files = ", files)

            index = 0
            for filename in files:
                print(filename)
                if filename.find('shift') > 0:
                    os.remove(root + filename)
    # 重新增扩数据
    for i in range(1,5):
        COOKED_DIR = 'F:/项目/花城音乐项目/样式数据/ALL/旋律/' + path_index[i - 1] + '/'
        # savepath = 'F:\\mfcc_pic\\'+ str(i) +'\\'
        for root, dirs, files in os.walk(COOKED_DIR):
            print("Root = ", root, "dirs = ", dirs, "files = ", files)

            index = 0
            for filename in files:
                print(filename)
                augmention(root, filename.split('.wav')[0], root)
