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
from pitch_helper import get_all_myabe_starts


def clear_dir(dis_dir):
    # shutil.rmtree(dis_dir)
    # os.mkdir(dis_dir)
    delList = os.listdir(dis_dir)
    for f in delList:
        filePath = os.path.join(dis_dir, f)
        if os.path.isfile(filePath):
            os.remove(filePath)
            print
            filePath + " was removed!"


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

def cqt_split_and_save(filename,onset_frames,savepath):
    y, sr = librosa.load(filename)
    rms = librosa.feature.rmse(y=y)[0]
    time = librosa.get_duration(filename=filename)
    total_frames_number = len(rms)
    print("time is {}".format(time))
    CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref=np.max)
    w, h = CQT.shape
    for i in range(0, len(onset_frames)):
        half = 30
        start = onset_frames[i] - half
        if start < 0:
            start = 0
        end = onset_frames[i] + half
        if end >= total_frames_number:
            end = total_frames_number - 1
        # y2 = [x if i> start and i<end else 0 for i,x in enumerate(y)]
        CQT_sub = np.zeros(CQT.shape)
        middle = int(h / 2)
        offset = middle - onset_frames[i]
        for j in range(int(start), int(end)):
            CQT_sub[:, j + offset] = CQT[:, j]
        # CQT = CQT_T
        librosa.display.specshow(CQT_sub, y_axis='cqt_note', x_axis='time')
        # y2 = [x for i,x in enumerate(y) if i> start and i<end]
        # y2 = [0.03 if i> start and i<end else 0.02 for i,x in enumerate(y)]
        # y2[int(len(y2) / 2)] = np.max(y)  # 让图片展示归一化
        t = librosa.frames_to_time([middle], sr=sr)
        plt.vlines(t, 0, sr, color='y', linestyle='--')  # 标出节拍位置
        # y2 = np.array(y2)
        # print("len(y2) is {}".format(len(y2)))

        print("(end - start)*sr is {}".format((end - start) * sr))
        # plt.show()
        # plt.subplot(len(onset_times),1,i+1)
        # y, sr = librosa.load(filename, offset=2.0, duration=3.0)
        # librosa.display.waveplot(y2, sr=sr)
        fig = matplotlib.pyplot.gcf()
        # fig.set_size_inches(4, 4)
        if "." in filename:
            Filename = filename.split(".")[0]
        plt.axis('off')
        plt.axes().get_xaxis().set_visible(False)
        plt.axes().get_yaxis().set_visible(False)
        # plt.savefig(savepath + str(i + 1) + '.jpg', bbox_inches='tight', pad_inches=0)
        plt.savefig(savepath + str(onset_frames[i]) + '.jpg', bbox_inches='tight', pad_inches=0)
        plt.clf()

def init_data(filename, rhythm_code,savepath):
    clear_dir(savepath)
    change_points, starts_from_rms_maybe, starts_on_highest_point = get_all_myabe_starts(filename, rhythm_code)
    if len(starts_from_rms_maybe) > 0:
        for s in starts_from_rms_maybe:
            change_points.append(s)
    if len(starts_on_highest_point) > 0:
        for s in starts_on_highest_point:
            change_points.append(s)
    cqt_split_and_save(filename, change_points, savepath)

def get_starts_by_overage(onset_frames):
    select_starts = []
    select_starts.append(onset_frames[0])
    for i in range(1,len(onset_frames)):
        if onset_frames[i] - onset_frames[i-1] < 8:
            select_starts[-1] = int((select_starts[-1] + onset_frames[i])/2)
        else:
            select_starts.append(onset_frames[i])
    return select_starts


savepath = 'e:/test_image/t/'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏1.3(95).wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏1（二）(100).wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏1_40227（100）.wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏1林(70).wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏2_40314（100）.wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏2_40409（98）.wav'
# filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏2林(25).wav'
# filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏2语(85).wav'
#filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/节奏/节2罗（75）.wav'
# filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/视唱1-01（95）.wav'
# filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律二（11）（60）.wav'
# filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/视唱1-02（90）.wav'
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/两只老虎20190624-1089.wav', '[500,500,500,500;500,500,500,500;500,500,1000;500,500;1000]', '[1,2,3,1,1,2,3,1,3,4,5,3,4,5]'  # =======================
# filename,rhythm_code,pitch_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/两只老虎20190624-1328.wav','[500,500,500,500;500,500,500,500;500,500,1000;500,500;1000]','[1,2,3,1,1,2,3,1,3,4,5,3,4,5]'       #
# filename,rhythm_code,pitch_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/两只老虎20190624-1586.wav','[500,500,500,500;500,500,500,500;500,500,1000;500,500;1000]','[1,2,3,1,1,2,3,1,3,4,5,3,4,5]'      #
# filename,rhythm_code,pitch_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/两只老虎20190624-2939.wav','[500,500,500,500;500,500,500,500;500,500,1000;500,500;1000]','[1,2,3,1,1,2,3,1,3,4,5,3,4,5]'      #81
# filename,rhythm_code,pitch_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/两只老虎20190624-7881.wav','[500,500,500,500;500,500,500,500;500,500,1000;500,500;1000]','[1,2,3,1,1,2,3,1,3,4,5,3,4,5]'          #100
# filename,rhythm_code,pitch_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/两只老虎20190624-8973.wav','[500,500,500,500;500,500,500,500;500,500,1000;500,500;1000]','[1,2,3,1,1,2,3,1,3,4,5,3,4,5]'       #100


# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/小学8题20190624-3898-6.wav', '[1000,500,500;2000;250,250,500,500,500;2000]', '[6,5,3,6,3,5,3,2,1,6-]'  #72
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/小学8题20190624-3898-7.wav', '[2000;250,250,250,250,1000;2000;500,500,1000]', '[6,5,6,3,5,6,3,2,1,6-]'  #82
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/小学8题20190624-3898-8.wav', '[1000,250,250,250,250;2000;1000,500,500;2000]', '[1,3,5,1+,6,5,1,3,2,1]'  #100

# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.12MP3/旋律/小学8题20190702-2647-5.wav', '[1000,1000;500,250,250,1000;1000,500,500;2000]', '[3,1,5,5,6,5,1+,6,3,5]'  #86
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.12MP3/旋律/小学8题20190702-2647-6.wav', '[1000,500,500;2000;250,250,500,500,500;2000]', '[6,5,3,6,3,5,3,2,1,6]'  # 84
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.12MP3/旋律/小学8题20190702-2647-7.wav', '[2000;250,250,250,250,1000;2000;500,500,1000]', '[6,5,6,3,5,6,3,2,1,6]'  #74
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.12MP3/旋律/小学8题20190702-2647-8.wav', '[1000,250,250,250,250;2000;1000,500,500;2000]', '[1,3,5,1+,6,5,1,3,2,1]'  # 94


# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-2776-5.wav', '[1000,1000;500,250,250,1000;1000,500,500;2000]', '[3,1,5,5,6,5,1+,6,3,5]'  # 100
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-2776-6.wav', '[1000,500,500;2000;250,250,500,500,500;2000]', '[6,5,3,6,3,5,3,2,1,6-]'  #94
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-2776-7.wav', '[2000;250,250,250,250,1000;2000;500,500,1000]', '[6,5,6,3,5,6,3,2,1,6-]'  #100
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-2776-8.wav', '[1000,250,250,250,250;2000;1000,500,500;2000]', '[1,3,5,1+,6,5,1,3,2,1]'  #84

# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-5668-5.wav', '[1000,1000;500,250,250,1000;1000,500,500;2000]', '[3,1,5,5,6,5,1+,6,3,5]'  #100
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-5668-6.wav', '[1000,500,500;2000;250,250,500,500,500;2000]', '[6,5,3,6,3,5,3,2,1,6-]'  #100
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-5668-7.wav', '[2000;250,250,250,250,1000;2000;500,500,1000]', '[6,5,6,3,5,6,3,2,1,6-]'  #100
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-5668-8.wav', '[1000,250,250,250,250;2000;1000,500,500;2000]', '[1,3,5,1+,6,5,1,3,2,1]'  # 88

# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-6249-5.wav', '[1000,1000;500,250,250,1000;1000,500,500;2000]', '[3,1,5,5,6,5,1+,6,3,5]'  #76
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-6249-6.wav', '[1000,500,500;2000;250,250,500,500,500;2000]', '[6,5,3,6,3,5,3,2,1,6-]'  #94
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-6249-7.wav', '[2000;250,250,250,250,1000;2000;500,500,1000]', '[6,5,6,3,5,6,3,2,1,6-]'  #100
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-6249-8.wav', '[1000,250,250,250,250;2000;1000,500,500;2000]', '[1,3,5,1+,6,5,1,3,2,1]'  #50

# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-6285-5.wav', '[1000,1000;500,250,250,1000;1000,500,500;2000]', '[3,1,5,5,6,5,1+,6,3,5]'  #100
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-6285-6.wav', '[1000,500,500;2000;250,250,500,500,500;2000]', '[6,5,3,6,3,5,3,2,1,6-]'  #  90
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-6285-7.wav', '[2000;250,250,250,250,1000;2000;500,500,1000]', '[6,5,6,3,5,6,3,2,1,6]'  #90
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-6285-8.wav', '[1000,250,250,250,250;2000;1000,500,500;2000]', '[1,3,5,1+,6,5,1,3,2,1]'  #8

# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.12MP3/旋律/小学8题20190712-4290-5.wav', '[1000,1000;500,250,250,1000;1000,500,500;2000]', '[3,1,5,5,6,5,1+,6,3,5]'  #100
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.12MP3/旋律/小学8题20190712-4290-6.wav', '[1000,500,500;2000;250,250,500,500,500;2000]', '[6,5,3,6,3,5,3,2,1,6]'  #84
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.12MP3/旋律/小学8题20190712-4290-7.wav','[2000;250,250,250,250,1000;2000;500,500,1000]', '[6,5,6,3,5,6,3,2,1,6]'  # 88
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.12MP3/旋律/小学8题20190712-4290-8.wav', '[1000,250,250,250,250;2000;1000,500,500;2000]', '[1,3,5,1+,6,5,1,3,2,1]'  #100

# filename, rhythm_code, pitch_code  = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋3罗（80）.wav', '[1000,1000;500,500,1000;500,250,250,500,500;2000]', '[5,5,3,2,1,2,2,3,2,6-,5-]' # 67 背景噪声造成音高不准
# filename, rhythm_code, pitch_code  = 'F:/项目/花城音乐项目/样式数据/6.18MP3/旋律/01，98.wav', '[500,250,250,500,500;250,250,250,250,500,500;500,250,250,500,500;500,250,250,1000]', '[5,5,6,5,3,4,5,4,5,4,2,3,3,4,3,1,2,3,5,1]'  # 25 背景噪声太大

# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/test.wav','[1000,250,250,250,250;2000;1000,500,500;2000]', '[1,3,5,1+,6,5,1,3,2,1]'  #94
# filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋4.4(0).wav'
# filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋2.3(95).wav'
# filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋2.4(50).wav'
# filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋2录音1(90).wav'
# type_index = get_onsets_index_by_filename_rhythm(filename)
# rhythm_code = get_code(type_index, 2)
# pitch_code = get_code(type_index, 3)
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/8.05MP3/旋律/小学8题20190805-9112-5.wav', '[1000,1000;500,250,250,1000;1000,500,500;2000]', '[3,1,5,5,6,5,1+,6,3,5]'  #38?????????????????????????????
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/8.05MP3/旋律/小学8题20190805-9112-6.wav', '[1000,500,500;2000;250,250,500,500,500;2000]', '[6,5,3,6,3,5,3,2,1,6]'  #72???????????????????????? 倍频问题
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/8.05MP3/旋律/小学8题20190805-9112-7.wav','[2000;250,250,250,250,1000;2000;500,500,1000]', '[6,5,6,3,5,6,3,2,1,6]'  #62????????????????????????
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/8.05MP3/旋律/小学8题20190805-9112-8.wav', '[1000,250,250,250,250;2000;1000,500,500;2000]', '[1,3,5,1+,6,5,1,3,2,1]'  #94????????????

# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/8.05MP3/旋律/中学8题20190805-6858-5.wav', '[500,250,250,500,500;1500,500;1000,1000;2000]', '[3,3,1,3,4,5,5,6,7,1+]'  #100
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/8.05MP3/旋律/中学8题20190805-6858-6.wav', '[1000,1000;1500,500;500,250,250,500,500;2000]', '[1+,7,6,5,4,5,4,3,2,1]'  #100
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/8.05MP3/旋律/中学8题20190805-6858-7.wav','[500,1000,500;2000;500,250,250,500,500;2000]', '[1,3,4,5,6,6,1+,7,6,1+]'  #100
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/8.05MP3/旋律/中学8题20190805-6858-8.wav', '[500,1000,500;2000;500,500,500,250,250;2000]', '[1+,7,6,5,6,5,4,3,2,1]'  #94


# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/8.12MP3/旋律/小学8题20190809-3492-5.wav', '[1000,1000;500,250,250,1000;1000,500,500;2000]', '[3,1,5,5,6,5,1+,6,3,5]'  #74 静默区起始点的伪节拍引起的
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/8.12MP3/旋律/小学8题20190809-3492-6.wav', '[1000,500,500;2000;250,250,500,500,500;2000]', '[6,5,3,6,3,5,3,2,1,6]'  # 84
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/8.12MP3/旋律/小学8题20190809-3492-7.wav','[2000;250,250,250,250,1000;2000;500,500,1000]', '[6,5,6,3,5,6,3,2,1,6]'  # 72
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/8.12MP3/旋律/小学8题20190809-3492-8.wav', '[1000,250,250,250,250;2000;1000,500,500;2000]', '[1,3,5,1+,6,5,1,3,2,1]'  #68????????????????????????????? 将1000也判断为静默区

# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/8.12MP3/旋律/小学8题20190812-6117-5.wav', '[1000,1000;500,250,250,1000;1000,500,500;2000]', '[3,1,5,5,6,5,1+,6,3,5]'  #94
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/8.12MP3/旋律/小学8题20190812-6117-6.wav', '[1000,500,500;2000;250,250,500,500,500;2000]', '[6,5,3,6,3,5,3,2,1,6]'  # 90
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/8.12MP3/旋律/小学8题20190812-6117-7.wav','[2000;250,250,250,250,1000;2000;500,500,1000]', '[6,5,6,3,5,6,3,2,1,6]'  #88
# filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/8.12MP3/旋律/小学8题20190812-6117-8.wav', '[1000,250,250,250,250;2000;1000,500,500;2000]', '[1,3,5,1+,6,5,1,3,2,1]'  #94


if __name__ == "__main__":
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

    savepath = 'F:/项目/花城音乐项目/参考代码/tensorflow_models_nets-master/single_notes/data/test/'
    cqt_split_and_save(filename, onset_frames,savepath)
    # clear_dir(savepath)
    # change_points, starts_from_rms_maybe, starts_on_highest_point = get_all_myabe_starts(filename, rhythm_code)
    # if len(starts_from_rms_maybe) > 0:
    #     for s in starts_from_rms_maybe:
    #         change_points.append(s)
    # if len(starts_on_highest_point) > 0:
    #     for s in starts_on_highest_point:
    #         change_points.append(s)
    # cqt_split_and_save(filename, change_points,savepath)

    # init_data(filename, rhythm_code, savepath)
    # cqt_split_and_save(filename, starts_from_rms_maybe)
    # cqt_split_and_save(filename, starts_on_highest_point)
#plt.show()