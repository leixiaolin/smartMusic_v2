import  numpy as np
import librosa
import matplotlib.pyplot as plt
import random
import os
from scipy.io import wavfile
import cv2

def augmention(filename,number):
    y, sr = librosa.load(filename)
    filepath, fullflname = os.path.split(filename)
    # 产生4个不相等的随机数
    shifts = []
    # while(len(shifts)< int(number/2)):
    #     x = random.randint(1,3)
    #     if x not in shifts:
    #         shifts.append(x)
    while (len(shifts) < number):
        x = random.randint(-4,3)
        if x not in shifts:
            shifts.append(x)

    for shift in shifts:
        # 通过移动音调变声，14是上移14个半步，如果是-14，则是下移14个半步
        b = librosa.effects.pitch_shift(y, sr, n_steps=shift)
        new_file = fullflname.split(".wav")[0] + '-shift-' + str(shift)
        save_path_file = filepath + "/" + new_file + '.wav'
        librosa.output.write_wav(save_path_file, b, sr)
def add_noice(filename,number):
    y, sr = librosa.load(filename)
    filepath, fullflname = os.path.split(filename)
    for i in range(number):
        wn = np.random.randn(len(y))
        y = np.where(y != 0.0, y + 0.005 * wn, 0.0)  # 噪声不要添加到0上！
        #y = [x + 0.02*random.random() for x in y if x != 0.0]
        new_file = fullflname.split(".wav")[0] + '-add-' + str(i)
        save_path_file = filepath + "/" + new_file + '.wav'
        #librosa.output.write_wav(save_path_file, y, sr) # 写入音频
        wavfile.write(save_path_file, sr, y)  # 写入音频

def change_tune(filename,number):
    y, sr = librosa.load(filename)
    filepath, fullflname = os.path.split(filename)
    for i in range(number):
        ly = len(y)
        tmp = random.uniform(1, 1.5)
        tmp = round(tmp,2)
        y_tune = cv2.resize(y, (1, int(len(y) * tmp))).squeeze()
        lc = len(y_tune) - ly
        y_tune = y_tune[int(lc / 2):int(lc / 2) + ly]


        new_file = fullflname.split(".wav")[0] + '-tune-' + str(i)
        save_path_file = filepath + "/" + new_file + '.wav'
        # librosa.output.write_wav(save_path_file, y, sr) # 写入音频
        wavfile.write(save_path_file, sr, y_tune)  # 写入音频

if __name__ == '__main__':

    path = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/'
    filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律一（12）（95）.wav'
    savepath = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/'
    number1 = 2
    number2 = 3
    number3 = 1
    #augmention(filename,number)
    #add_noice(filename,number)
    #change_tune(filename,number2)

    dir_list = ['F:/项目/花城音乐项目/样式数据/3.06MP3/节奏/']
    dir_list = ['e:/test_image/m/D/']
    #dir_list = []
    # clear_dir(result_path)
    # 要测试的数量
    test_num = 100
    score = 0
    for dir in dir_list:
        file_list = os.listdir(dir)
        # shuffle(file_list)  # 将语音文件随机排列
        # file_list = ['视唱1-01（95）.wav']
        for filename in file_list:

            if filename.find("tune") > 0 or filename.find("add") > 0 or filename.find("shift") > 0:
                continue
            # clear_dir(image_dir)
            # wavname = re.findall(pattern,filename)[0]
            print(dir + filename)
            #add_noice(dir + filename,number1)
            augmention(dir + filename,number2)
            #change_tune(dir + filename, number3)
