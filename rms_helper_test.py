# -*- coding: UTF-8 -*-
import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
from rms_helper import *
import os

# 1. Get the file path to the included audio example
# Sonify detected beat events
# 定义加载语音文件并去掉两端静音的函数

test_codes = np.array(['[1000,1000;2000;1000,500,500;2000]',
                       '[2000;1000,1000;500,500,1000;2000]',
                       '[1000,1000;500,500,1000;1000,1000;2000]',
                       '[1000,--(1000);1000,--(1000);500,250,250,1000;--(1000),1000]',
                       '[500;1000,500,1000,500;500,500,500,250,250,500,500;250,250,500,500,1000]',
                       '[1000,--(1000);1000,--(1000);1000,-(500),500;1000,1000]',
                       '[750,250,500,500,500,-(500);500,1000,500,500,-(500);750,250,500,500,500,-(500)]',
                       '[500,1000,500,500,250,250;1000,500,750,250,500;3000]',
                       '[500,500,500;1000,500;500,500,500;1500;500,500,500;1000,500;500;1000;1500]',
                       '[500,500,1000;500,500;1000;375,125,250,250,375,125,250,250;500,500,1000]'])
test_note_codes = np.array(['[3,3,3,3,3,3,3,5,1,2,3]',
                            '[5,5,3,2,1,2,5,3,2]',
                            '[5,5,3,2,1,2,2,3,2,6-,5-]',
                            '[5,1+,7,1+,2+,1+,7,6,5,2,4,3,6,5]',
                            '[3,6,7,1+,2+,1+,7,6,3]',
                            '[1+,7,1+,2+,3+,2+,1+,7,6,7,1+,2+,7,1+,7,1+,2+,1+]',
                            '[5,6,1+,6,2,3,1,6-,5-]',
                            '[5,5,6,5,6,5,1,3,0,2,2,5-,2,1]',
                            '[3,2,1,2,1,1,2,3,4,5,3,6,5,5,3]',
                            '[3,4,5,1+,7,6,5]'])
test_rhythm_codes = np.array(['[500,500,1000;500,500,1000;500,500,750,250;2000]',
                              '[1000,1000;500,500,1000;1000,500,500; 2000]',
                              '[1000,1000;500,500,1000;500,250,250,500,500;2000]',
                              '[500,1000,500;250,250,250,250,500,500;500,500,500,500;2000]',
                              '[1000;500,500,1000;500,500,500,500;2000]',
                              '[500;500,500,500,500;500,500,500,500;500,500,500,500;250,250,250,250,500]',
                              '[1000,750,250,2000;500,500,500,500,2000]',
                              '[1000,1000,1000,500,500;1000,1000,1000,--(1000);1000,1000,1000;1000,4000]',
                              '[1500,500,500,500;2500,500;1000,500,500,500,500;2500,500]',
                              '[500,500;1500,500,500,500;2000]'])


def get_code(index, type):
    if type == 1:
        code = test_codes[index]
    if type == 2:
        code = test_rhythm_codes[index]
    if type == 3:
        code = test_note_codes[index]
    # code = code.replace(";", ',')
    # code = code.replace("[", '')
    # code = code.replace("]", '')
    # code = [x for x in code.split(',')]
    return code


def get_onsets_index_by_filename(filename):
    if filename.find("节奏10") >= 0 or filename.find("节奏十") >= 0 or filename.find("节奏题十") >= 0 or filename.find(
            "节奏题10") >= 0 or filename.find("节10") >= 0:
        return 9
    elif filename.find("节奏1") >= 0 or filename.find("节奏一") >= 0 or filename.find("节奏题一") >= 0 or filename.find(
            "节奏题1") >= 0 or filename.find("节1") >= 0:
        return 0
    elif filename.find("节奏2") >= 0 or filename.find("节奏二") >= 0 or filename.find("节奏题二") >= 0 or filename.find(
            "节奏题2") >= 0 or filename.find("节2") >= 0:
        return 1
    elif filename.find("节奏3") >= 0 or filename.find("节奏三") >= 0 or filename.find("节奏题三") >= 0 or filename.find(
            "节奏题3") >= 0 or filename.find("节3") >= 0:
        return 2
    elif filename.find("节奏4") >= 0 or filename.find("节奏四") >= 0 or filename.find("节奏题四") >= 0 or filename.find(
            "节奏题4") >= 0 or filename.find("节4") >= 0:
        return 3
    elif filename.find("节奏5") >= 0 or filename.find("节奏五") >= 0 or filename.find("节奏题五") >= 0 or filename.find(
            "节奏题5") >= 0 or filename.find("节5") >= 0:
        return 4
    elif filename.find("节奏6") >= 0 or filename.find("节奏六") >= 0 or filename.find("节奏题六") >= 0 or filename.find(
            "节奏题6") >= 0 or filename.find("节6") >= 0:
        return 5
    elif filename.find("节奏7") >= 0 or filename.find("节奏七") >= 0 or filename.find("节奏题七") >= 0 or filename.find(
            "节奏题7") >= 0 or filename.find("节7") >= 0:
        return 6
    elif filename.find("节奏8") >= 0 or filename.find("节奏八") >= 0 or filename.find("节奏题八") >= 0 or filename.find(
            "节奏题8") >= 0 or filename.find("节8") >= 0:
        return 7
    elif filename.find("节奏9") >= 0 or filename.find("节奏九") >= 0 or filename.find("节奏题九") >= 0 or filename.find(
            "节奏题9") >= 0 or filename.find("节9") >= 0:
        return 8
    else:
        return -1


def get_onsets_index_by_filename_rhythm(filename):
    if filename.find("旋律10") >= 0 or filename.find("旋律十") >= 0 or filename.find("视唱十") >= 0 or filename.find(
            "视唱10") >= 0 or filename.find("旋10") >= 0:
        return 9
    elif filename.find("旋律1") >= 0 or filename.find("旋律一") >= 0 or filename.find("视唱一") >= 0 or filename.find(
            "视唱1") >= 0 or filename.find("旋1") >= 0:
        return 0
    elif filename.find("旋律2") >= 0 or filename.find("旋律二") >= 0 or filename.find("视唱二") >= 0 or filename.find(
            "视唱2") >= 0 or filename.find("旋2") >= 0:
        return 1
    elif filename.find("旋律3") >= 0 or filename.find("旋律三") >= 0 or filename.find("视唱三") >= 0 or filename.find(
            "视唱3") >= 0 or filename.find("旋3") >= 0:
        return 2
    elif filename.find("旋律4") >= 0 or filename.find("旋律四") >= 0 or filename.find("视唱四") >= 0 or filename.find(
            "视唱4") >= 0 or filename.find("旋4") >= 0:
        return 3
    elif filename.find("旋律5") >= 0 or filename.find("旋律五") >= 0 or filename.find("视唱五") >= 0 or filename.find(
            "视唱5") >= 0 or filename.find("旋5") >= 0:
        return 4
    elif filename.find("旋律6") >= 0 or filename.find("旋律六") >= 0 or filename.find("视唱六") >= 0 or filename.find(
            "视唱6") >= 0 or filename.find("旋6") >= 0:
        return 5
    elif filename.find("旋律7") >= 0 or filename.find("旋律七") >= 0 or filename.find("视唱七") >= 0 or filename.find(
            "视唱7") >= 0 or filename.find("旋7") >= 0:
        return 6
    elif filename.find("旋律8") >= 0 or filename.find("旋律八") >= 0 or filename.find("视唱八") >= 0 or filename.find(
            "视唱8") >= 0 or filename.find("旋8") >= 0:
        return 7
    elif filename.find("旋律9") >= 0 or filename.find("旋律九") >= 0 or filename.find("视唱九") >= 0 or filename.find(
            "视唱9") >= 0 or filename.find("旋9") >= 0:
        return 8
    else:
        return -1

def test_batch_samples():
    files,onset_codes = [],[]
    filename, onset_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/小学8题20190624-3898-1.wav', '[1000,1000;500,250,250,500;1000,500,500;2000]'  # 第1条 这个可以给满分                       95/90
    files.append(filename)
    onset_codes.append(onset_code)
    filename, onset_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/小学8题20190624-3898-2.wav', '[1000,500,500;2000;250,250,500,500,500;2000]'  # 第2条 基本上可以是满分                      100
    files.append(filename)
    onset_codes.append(onset_code)
    filename, onset_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/小学8题20190624-3898-3.wav', '[2000;250,250,250,250,1000;2000;500,500,1000]'  # 第3条 故意错一个，扣一分即可               89?86
    files.append(filename)
    onset_codes.append(onset_code)
    filename, onset_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/小学8题20190624-3898-4.wav', '[1000,250,250,250,250;2000;1000,500,500;2000]'  # 第4条 故意错了两处，应该扣两分左右即可     94?87
    files.append(filename)
    onset_codes.append(onset_code)
    filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.01MP3/旋律/小学8题20190625-2251 节拍题一.wav', '[1000,1000;500,250,250,500;1000,500,500;2000]'  # 应该有七分左右                     78
    files.append(filename)
    onset_codes.append(onset_code)
    filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.01MP3/旋律/小学8题20190625-2251 节拍题三.wav', '[2000;250,250,250,250,1000;2000;500,500,1000]'  # 应该接近满分                       98
    files.append(filename)
    onset_codes.append(onset_code)
    filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.01MP3/旋律/中学8题20190701-4154 节拍题二.wav', '[1000,1000;1500,500;500,250,250,500,500;2000]'  # 可给满分                           100
    files.append(filename)
    onset_codes.append(onset_code)
    filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.01MP3/旋律/中学8题20190701-4154 节拍题三.wav', '[500,1000,500;2000;500,250,250,500,500;2000]'  # 可给接近满分                        100
    files.append(filename)
    onset_codes.append(onset_code)
    filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.01MP3/旋律/录音题E20190701-9528 第一题.wav', '[1000,1000;500,250,250,1000;500,500,500,500;2000]'  # 可给满分                         100
    files.append(filename)
    onset_codes.append(onset_code)
    filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.01MP3/旋律/录音题E20190701-9528 第二题.wav', '[1000,500,500;500,250,250,500;500,500,1000;2000]'  # 可给接近满分                      90
    files.append(filename)
    onset_codes.append(onset_code)
    filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.01MP3/旋律/中学8题20190701-1547 节奏一.wav', '[500,250,250,500,500;1500,500;1000,1000;2000]'  # 可给接近满分                         94
    files.append(filename)
    onset_codes.append(onset_code)
    filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.01MP3/旋律/中学8题20190701-1547 节奏二.wav', '[1000,1000;1500,500;500,250,250,500,500;2000]'  # 可给接近满分                         97 ?????
    files.append(filename)
    onset_codes.append(onset_code)
    filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.01MP3/旋律/中学8题20190701-1547 节奏三.wav', '[500,1000,500;2000;500,250,250,500,500;2000]'  # 可给接近满分                          100
    files.append(filename)
    onset_codes.append(onset_code)
    filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.01MP3/旋律/中学8题20190701-1547 节奏四.wav', '[500,1000,500;2000;500,500,500,250,250;2000]'  # 应该给接近九分                        93 ????
    files.append(filename)
    onset_codes.append(onset_code)
    filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.01MP3/旋律/中学8题20190701-1547 节奏四.wav', '[500,1000,500;2000;500,500,500,250,250;2000]'  # 应该给接近九分                        93
    files.append(filename)
    onset_codes.append(onset_code)

    filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.12MP3/旋律/小学8题20190702-2647-1.wav', '[1000,1000;500,250,250,1000;1000,500,500;2000]'  # 可给满分                        100
    files.append(filename)
    onset_codes.append(onset_code)
    filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.12MP3/旋律/小学8题20190702-2647-2.wav', '[1000,500,500;2000;250,250,500,500,500;2000]'  # 可给满分                       100
    files.append(filename)
    onset_codes.append(onset_code)
    filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.12MP3/旋律/小学8题20190702-2647-3.wav', '[2000,250,250,250,250,1000;2000;500,500,1000]'  # 可给满分                       100
    files.append(filename)
    onset_codes.append(onset_code)
    filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.12MP3/旋律/小学8题20190702-2647-4.wav', '[1000,250,250,250,250;2000;1000,500,500;2000]'  # 可给满分                        100
    files.append(filename)
    onset_codes.append(onset_code)

    filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-2776-1.wav', '[1000,1000;500,250,250,1000;1000,500,500;2000]'  # 100
    files.append(filename)
    onset_codes.append(onset_code)
    filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-2776-2.wav', '[1000,500,500;2000;250,250,500,500,500;2000]'  # 100
    files.append(filename)
    onset_codes.append(onset_code)
    filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-2776-3.wav', '[2000;250,250,250,250,1000;2000;500,500,1000]'  # 100
    files.append(filename)
    onset_codes.append(onset_code)
    filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-2776-4.wav', '[1000,250,250,250,250;2000;1000,500,500;2000]'  # 100
    files.append(filename)
    onset_codes.append(onset_code)

    filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-5668-1.wav', '[1000,1000;500,250,250,1000;1000,500,500;2000]'  # 68
    files.append(filename)
    onset_codes.append(onset_code)
    filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-5668-2.wav', '[1000,500,500;2000;250,250,500,500,500;2000]'  # 65
    files.append(filename)
    onset_codes.append(onset_code)
    filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-5668-3.wav', '[2000;250,250,250,250,1000;2000;500,500,1000]'  # 87
    files.append(filename)
    onset_codes.append(onset_code)

    filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-5668-4.wav', '[1000,250,250,250,250;2000;1000,500,500;2000]'  # 100 ?????
    files.append(filename)
    onset_codes.append(onset_code)
    filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-6249-1.wav', '[1000,1000;500,250,250,1000;1000,500,500;2000]'  # 100
    files.append(filename)
    onset_codes.append(onset_code)
    filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-6249-2.wav', '[1000,500,500;2000;250,250,500,500,500;2000]'  # 100
    files.append(filename)
    onset_codes.append(onset_code)
    filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-6249-3.wav', '[2000;250,250,250,250,1000;2000;500,500,1000]'  # 100
    files.append(filename)
    onset_codes.append(onset_code)
    filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-6249-4.wav', '[1000,250,250,250,250;2000;1000,500,500;2000]'  # 100
    files.append(filename)
    onset_codes.append(onset_code)

    filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-6285-1.wav', '[1000,1000;500,250,250,1000;1000,500,500;2000]'  # 100
    files.append(filename)
    onset_codes.append(onset_code)
    filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-6285-2.wav', '[1000,500,500;2000;250,250,500,500,500;2000]'  # 100
    files.append(filename)
    onset_codes.append(onset_code)
    filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-6285-3.wav', '[2000;250,250,250,250,1000;2000;500,500,1000]'  # 100
    files.append(filename)
    onset_codes.append(onset_code)
    filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-6285-4.wav', '[1000,250,250,250,250;2000;1000,500,500;2000]'  # 100
    files.append(filename)
    onset_codes.append(onset_code)

    filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.18MP3/旋律/小学8题20190717-4634-1.wav', '[1000,1000;500,250,250,1000;1000,500,500;2000]'  # 54
    files.append(filename)
    onset_codes.append(onset_code)
    filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.18MP3/旋律/小学8题20190717-4634-2.wav', '[1000,500,500;2000;250,250,500,500,500;2000]'  # 47
    files.append(filename)
    onset_codes.append(onset_code)
    filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.18MP3/旋律/小学8题20190717-4634-3.wav', '[2000;250,250,250,250,1000;2000;500,500,1000]'  # 42
    files.append(filename)
    onset_codes.append(onset_code)
    filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.18MP3/旋律/小学8题20190717-4634-4.wav', '[1000,250,250,250,250;2000;1000,500,500;2000]'  # 30
    files.append(filename)
    onset_codes.append(onset_code)

    filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.18MP3/旋律/小学8题20190717-4856-1.wav', '[1000,1000;500,250,250,1000;1000,500,500;2000]'  # 92
    files.append(filename)
    onset_codes.append(onset_code)
    filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.18MP3/旋律/小学8题20190717-4856-2.wav', '[1000,500,500;2000;250,250,500,500,500;2000]'  # 65
    files.append(filename)
    onset_codes.append(onset_code)
    filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.18MP3/旋律/小学8题20190717-4856-3.wav', '[2000;250,250,250,250,1000;2000;500,500,1000]'  # 81
    files.append(filename)
    onset_codes.append(onset_code)
    filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.18MP3/旋律/小学8题20190717-4856-4.wav', '[1000,250,250,250,250;2000;1000,500,500;2000]'  # 85
    files.append(filename)
    onset_codes.append(onset_code)

    filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.18MP3/旋律/小学8题20190718-4074-1.wav', '[1000,1000;500,250,250,1000;1000,500,500;2000]'  # 60
    files.append(filename)
    onset_codes.append(onset_code)
    filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.18MP3/旋律/小学8题20190718-4074-2.wav', '[1000,500,500;2000;250,250,500,500,500;2000]'  # 67
    files.append(filename)
    onset_codes.append(onset_code)
    filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.18MP3/旋律/小学8题20190718-4074-3.wav', '[2000;250,250,250,250,1000;2000;500,500,1000]'  # 49
    files.append(filename)
    onset_codes.append(onset_code)
    filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.18MP3/旋律/小学8题20190718-4074-4.wav', '[1000,250,250,250,250;2000;1000,500,500;2000]'  # 86
    files.append(filename)
    onset_codes.append(onset_code)

    filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.18MP3/旋律/小学8题20190718-7649-1.wav', '[1000,1000;500,250,250,1000;1000,500,500;2000]'  # 66
    files.append(filename)
    onset_codes.append(onset_code)
    filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.18MP3/旋律/小学8题20190718-7649-2.wav', '[1000,500,500;2000;250,250,500,500,500;2000]'  # 89
    files.append(filename)
    onset_codes.append(onset_code)
    filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.18MP3/旋律/小学8题20190718-7649-3.wav', '[2000;250,250,250,250,1000;2000;500,500,1000]'  # 100
    files.append(filename)
    onset_codes.append(onset_code)
    filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.18MP3/旋律/小学8题20190718-7649-4.wav', '[1000,250,250,250,250;2000;1000,500,500;2000]'  # 100
    files.append(filename)
    onset_codes.append(onset_code)

    filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.18MP3/旋律/小学8题20190718-9728-1.wav', '[1000,1000;500,250,250,1000;1000,500,500;2000]'  # 100
    files.append(filename)
    onset_codes.append(onset_code)
    filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.18MP3/旋律/小学8题20190718-9728-2.wav', '[1000,500,500;2000;250,250,500,500,500;2000]'  # 100
    files.append(filename)
    onset_codes.append(onset_code)
    filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.18MP3/旋律/小学8题20190718-9728-3.wav', '[2000;250,250,250,250,1000;2000;500,500,1000]'  # 100
    files.append(filename)
    onset_codes.append(onset_code)
    filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.18MP3/旋律/小学8题20190718-9728-4.wav', '[1000,250,250,250,250;2000;1000,500,500;2000]'  # 100
    files.append(filename)
    onset_codes.append(onset_code)

    for filename,onset_code in zip(files,onset_codes):
        print(filename)
        rms, rms_diff, sig_ff, max_indexs = get_rms_max_indexs_for_onset(filename, onset_code)
        max_indexs = get_best_max_index(filename, onset_code)
        start, end, total_length = get_start_end_length_by_max_index(max_indexs, filename)
        total_score, detail = calculate_score(max_indexs, onset_code,end)
        print("total_score is {}".format(total_score))
        print("detail is {}".format(detail))
        print("")
        plt = draw_plt(filename,onset_code,rms,sig_ff,max_indexs,start,end)
        #plt.savefig(result_path  + filename.split('/')[-1].split('.wav')[0] + '-' + str(total_score) + '-' + str(lost_score) + '-' + str(ex_score) + '-' + str(min_d) + '.jpg', bbox_inches='tight', pad_inches=0)
        plt.savefig(result_path  + filename.split('/')[-1].split('.wav')[0] + '-' + str(total_score) + '分.jpg', bbox_inches='tight', pad_inches=0)
        plt.clf()


def test_all(dir):
    file_list = os.listdir(dir)
    for filename in file_list:
        filename = dir + filename
        print(filename)
        if filename.find('中') >= 0:
            if filename.find('1.wav') >= 0:
                onset_code = '[500,250,250,500,500;1500,500;1000,1000;2000]'
            if filename.find('2.wav') >= 0:
                onset_code =  '[1000,1000;1500,500;500,250,250,500,500;2000]'
            if filename.find('3.wav') >= 0:
                onset_code = '[500,1000,500;2000;500,250,250,500,500;2000]'
            if filename.find('4.wav') >= 0:
                onset_code = '[500,1000,500;2000;500,500,500,250,250;2000]'
        else:
            if filename.find('1.wav') >= 0:
                onset_code = '[1000,1000;500,250,250,1000;1000,500,500;2000]'
            if filename.find('2.wav') >= 0:
                onset_code =  '[1000,500,500;2000;250,250,500,500,500;2000]'
            if filename.find('3.wav') >= 0:
                onset_code = '[2000;250,250,250,250,1000;2000;500,500,1000]'
            if filename.find('4.wav') >= 0:
                onset_code = '[1000,250,250,250,250;2000;1000,500,500;2000]'
        rms, rms_diff, sig_ff, max_indexs = get_rms_max_indexs_for_onset(filename, onset_code)
        max_indexs = get_best_max_index(filename, onset_code)
        # print("0 max_indexs is {},size is {}".format(max_indexs, len(max_indexs)))
        start, end, total_length = get_start_end_length_by_max_index(max_indexs, filename)
        max_indexs = [x for x in max_indexs if x > start - 5 and x < end - 5]
        max_indexs.append(end)
        max_indexs.sort()
        total_score, detail = calculate_score(max_indexs, onset_code,end)
        print("total_score is {}".format(total_score))
        print("detail is {}".format(detail))
        print("")
        plt = draw_plt(filename,onset_code,rms,sig_ff,max_indexs,start,end)
        #plt.savefig(result_path  + filename.split('/')[-1].split('.wav')[0] + '-' + str(total_score) + '-' + str(lost_score) + '-' + str(ex_score) + '-' + str(min_d) + '.jpg', bbox_inches='tight', pad_inches=0)
        plt.savefig(result_path  + filename.split('/')[-1].split('.wav')[0] + '-' + str(total_score) + '分.jpg', bbox_inches='tight', pad_inches=0)
        plt.clf()

if __name__ == "__main__":
    # y, sr = load_and_trim('F:/项目/花城音乐项目/样式数据/ALL/旋律/1.31MP3/旋律1.100分.wav')
    filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律2.1(80).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/ALL/旋律/1.31MP3/旋律3.100分.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律一（9）（100）.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律一（14）（95）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋律五（3）（63）.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏一（4）（96）.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋3王（80）.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋4谭（95）.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋4文(75).wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋8录音1(80).wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1.3(93).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋3罗（80）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律1_40312（95）.wav'
    # filename = 'e:/test_image/m1/A/旋律1_40312（95）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋3罗（80）.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋律十（2）（80）.wav'

    # filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律8录音3(95).wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1王（98）.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/Archive/dada1.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1王（98）.wav'

    # filename = 'F:/项目/花城音乐项目/样式数据/1-2/旋律mp3/旋律8.100分.mp3'
    # filename = 'F:/项目/花城音乐项目/样式数据/1-2/旋律mp3/旋律7.100分.mp3'
    # filename = 'F:/项目/花城音乐项目/样式数据/1-2/旋律mp3/旋律5.100分.mp3'
    # filename = 'F:/项目/花城音乐项目/样式数据/1-2/旋律mp3/旋律六.5（100）.mp3'
    # filename = 'F:/项目/花城音乐项目/样式数据/1-2/旋律mp3/旋律6.75分.mp3'
    # filename =  'F:/项目/花城音乐项目/样式数据/1-2/旋律mp3/旋律1.40分.mp3'

    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1.2(92).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1谭（98）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1王（98）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋3.3(96).wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋4谭（95）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋律8录音3(95).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1谭（98）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1王（98）.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋8文(58).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋律四（1）（20）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋4王（56）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋4欧(25).wav'

    # filename = 'F:/项目/花城音乐项目/样式数据/1-2/旋律/旋律八（9）(90).wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/1-2/旋律/旋律二（2）（90分）.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/1-2/旋律/旋律九（4）(95).wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/1-2/旋律/旋律三（2）（90分）.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/1-2/旋律/旋律四.1（100）.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/1-2/旋律/旋律四.3（100）.wav'
    # # filename = 'F:/项目/花城音乐项目/样式数据/1-2/旋律/旋律十（5）(50).wav'
    # # filename = 'F:/项目/花城音乐项目/样式数据/1-2/旋律/旋律七(5)（55）.wav'
    # # filename = 'F:/项目/花城音乐项目/样式数据/1-2/旋律/旋律1.90分.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/1-2/旋律/旋律四.10（100）.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/1-2/旋律/旋律三（3）（80分）.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/1-2/旋律/旋律三（8）(80).wav'
    # # filename = 'F:/项目/花城音乐项目/样式数据/1-2/旋律/旋律二（2）（90分）.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/1-2/旋律/旋律三.10（100）.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/1-2/旋律/旋律一.6（100）.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/1-2/旋律/旋律九（6）(50).wav'

    filename = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/两只老虎20190624-2939.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/小学8题20190624-3898-2.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/节奏/节10.4(60).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/节奏/节1文(95).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/节奏/节奏一（3）（90）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/节奏/节奏十（1）（100）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/节奏/节1谭（96）.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/节奏/节4.1(95).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/节奏/节奏三（1）（100）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/节奏/节10.4(60).wav'
    filename, onset_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/小学8题20190624-3898-1.wav', '[1000,1000;500,250,250,500;1000,500,500;2000]'  # 第1条
    filename, onset_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/小学8题20190624-3898-2.wav', '[1000,500,500;2000;250,250,500,500,500;2000]'  # 第2条
    # filename,onset_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/小学8题20190624-3898-3.wav','[2000;250,250,250,250,1000;2000;500,500,1000]'  # 第3条
    # filename,onset_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/小学8题20190624-3898-4.wav','[1000,250,250,250,250;2000;1000,500,500;2000]'  # 第4条

    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/节奏/节10.1(97).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/节奏/节4熙(95).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/6.18MP3/节奏/12；98.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/6.18MP3/节奏/1；100.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/6.18MP3/节奏/10；84.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/6.18MP3/节奏/节奏3，90.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/6.18MP3/节奏/节奏3，90.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/6.18MP3/节奏/节奏3，78.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/6.18MP3/节奏/节奏3，80.wav'

    # filename,onset_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/小学8题20190624-3898-1.wav','[1000,1000;500,250,250,500;1000,500,500;2000]'  # 第1条 这个可以给满分 90
    # filename,onset_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/小学8题20190624-3898-2.wav','[1000,500,500;2000;250,250,500,500,500;2000]'  # 第2条 基本上可以是满分  97
    # filename,onset_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/小学8题20190624-3898-3.wav','[2000;250,250,250,250,1000;2000;500,500,1000]'  # 第3条 故意错一个，扣一分即可 89
    # filename,onset_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/小学8题20190624-3898-4.wav','[1000,250,250,250,250;2000;1000,500,500;2000]'  # 第4条 故意错了两处，应该扣两分左右即可  85
    # filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.01MP3/旋律/小学8题20190625-2251 节拍题一.wav', '[1000,1000;500,250,250,500;1000,500,500;2000]'  #应该有七分左右 74
    filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.01MP3/旋律/小学8题20190625-2251 节拍题三.wav', '[2000;250,250,250,250,1000;2000;500,500,1000]'  #应该接近满分 97
    # filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.01MP3/旋律/中学8题20190701-4154 节拍题二.wav', '[1000,1000;1500,500;500,250,250,500,500;2000]'  #可给满分 100
    # filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.01MP3/旋律/中学8题20190701-4154 节拍题三.wav', '[500,1000,500;2000;500,250,250,500,500;2000]'  #可给接近满分 100
    # filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.01MP3/旋律/录音题E20190701-9528 第一题.wav', '[1000,1000;500,250,250,1000;500,500,500,500;2000]'  #可给满分 89
    # filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.01MP3/旋律/录音题E20190701-9528 第二题.wav', '[1000,500,500;500,250,250,500;500,500,1000;2000]'  # 可给接近满分 90
    # filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.01MP3/旋律/中学8题20190701-1547 节奏一.wav', '[500,250,250,500,500;1500,500;1000,1000;2000]'  # 可给接近满分 71
    # filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.01MP3/旋律/中学8题20190701-1547 节奏二.wav', '[1000,1000;1500,500;500,250,250,500,500;2000]'  # 可给接近满分 8?
    # filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.01MP3/旋律/中学8题20190701-1547 节奏三.wav', '[500,1000,500;2000;500,250,250,500,500;2000]'  #可给接近满分 94
    # filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.01MP3/旋律/中学8题20190701-1547 节奏四.wav', '[500,1000,500;2000;500,500,500,250,250;2000]'  #应该给接近九分87

    result_path = 'e:/test_image/n/'
    plt.close()
    type_index = get_onsets_index_by_filename(filename)
    #onset_code = get_code(type_index, 1)
    rhythm_code = get_code(type_index, 2)
    pitch_code = get_code(type_index, 3)

    # filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-5668-4.wav', '[1000,250,250,250,250;2000;1000,500,500;2000]'
    # filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.01MP3/旋律/中学8题20190701-1547 节奏四.wav', '[500,1000,500;2000;500,500,500,250,250;2000]'  # 应该给接近九分
    # filename, onset_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/小学8题20190624-3898-2.wav', '[1000,500,500;2000;250,250,500,500,500;2000]'  # 第2条 基本上可以是满分                      100
    # filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.01MP3/旋律/小学8题20190625-2251 节拍题一.wav', '[1000,1000;500,250,250,500;1000,500,500;2000]'  # 应该有七分左右 74

    filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.18MP3/旋律/小学8题20190717-4634-1.wav', '[1000,1000;500,250,250,1000;1000,500,500;2000]'  # 54
    # filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-6285-4.wav', '[1000,250,250,250,250;2000;1000,500,500;2000]'  # 100
    # rhythm_code = '[1000,1000;500,500,1000;500,250,250,500,500;2000]'
    # melody_code = '[5,5,3,2,1,2,2,3,2,6-,5-]'
    # filename,onset_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/小学8题20190624-3898-1.wav','[1000,1000;500,250,250,500;1000,500,500;2000]'  # 第1条 这个可以给满分                       95/90
    filename,onset_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/小学8题20190624-3898-2.wav','[1000,500,500;2000;250,250,500,500,500;2000]'  # 第2条 基本上可以是满分                      100
    filename,onset_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/小学8题20190624-3898-3.wav','[2000;250,250,250,250,1000;2000;500,500,1000]'  # 第3条 故意错一个，扣一分即可               89?86
    filename,onset_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/小学8题20190624-3898-4.wav','[1000,250,250,250,250;2000;1000,500,500;2000]'  # 第4条 故意错了两处，应该扣两分左右即可     94?87
    # filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.01MP3/旋律/小学8题20190625-2251 节拍题一.wav', '[1000,1000;500,250,250,500;1000,500,500;2000]'  #应该有七分左右                     78
    # filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.01MP3/旋律/小学8题20190625-2251 节拍题三.wav', '[2000;250,250,250,250,1000;2000;500,500,1000]'  #应该接近满分                       98
    # filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.01MP3/旋律/中学8题20190701-4154 节拍题二.wav', '[1000,1000;1500,500;500,250,250,500,500;2000]'  #可给满分                           100
    # filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.01MP3/旋律/中学8题20190701-4154 节拍题三.wav', '[500,1000,500;2000;500,250,250,500,500;2000]'  #可给接近满分                        100
    # filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.01MP3/旋律/录音题E20190701-9528 第一题.wav', '[1000,1000;500,250,250,1000;500,500,500,500;2000]'  #可给满分                         100
    # filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.01MP3/旋律/录音题E20190701-9528 第二题.wav', '[1000,500,500;500,250,250,500;500,500,1000;2000]'  #可给接近满分                      90
    # filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.01MP3/旋律/中学8题20190701-1547 节奏一.wav', '[500,250,250,500,500;1500,500;1000,1000;2000]'  #可给接近满分                         94
    # filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.01MP3/旋律/中学8题20190701-1547 节奏二.wav', '[1000,1000;1500,500;500,250,250,500,500;2000]'  #可给接近满分                         97 ?????
    # filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.01MP3/旋律/中学8题20190701-1547 节奏三.wav', '[500,1000,500;2000;500,250,250,500,500;2000]'  #可给接近满分                          100
    # filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.01MP3/旋律/中学8题20190701-1547 节奏四.wav', '[500,1000,500;2000;500,500,500,250,250;2000]'  #应该给接近九分                        93

    # filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.12MP3/旋律/小学8题20190702-2647-1.wav', '[1000,1000;500,250,250,1000;1000,500,500;2000]'  # 可给满分                        100
    # filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.12MP3/旋律/小学8题20190702-2647-2.wav', '[1000,500,500;2000;250,250,500,500,500;2000]'  # 可给满分                       100
    # filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.12MP3/旋律/小学8题20190702-2647-3.wav', '[2000,250,250,250,250,1000;2000;500,500,1000]'  # 可给满分                       100
    # filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.12MP3/旋律/小学8题20190702-2647-4.wav', '[1000,250,250,250,250;2000;1000,500,500;2000]'  # 可给满分                        100

    # filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-2776-1.wav', '[1000,1000;500,250,250,1000;1000,500,500;2000]'  # 100
    # filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-2776-2.wav', '[1000,500,500;2000;250,250,500,500,500;2000]'  # 100
    # filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-2776-3.wav', '[2000;250,250,250,250,1000;2000;500,500,1000]'  # 100
    # filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-2776-4.wav', '[1000,250,250,250,250;2000;1000,500,500;2000]'  # 100

    # filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-5668-1.wav', '[1000,1000;500,250,250,1000;1000,500,500;2000]'  # 68
    # filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-5668-2.wav', '[1000,500,500;2000;250,250,500,500,500;2000]'  # 65
    # filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-5668-3.wav', '[2000;250,250,250,250,1000;2000;500,500,1000]'  # 87
    # filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-5668-4.wav', '[1000,250,250,250,250;2000;1000,500,500;2000]'  # 100 ?????

    # filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-6249-1.wav', '[1000,1000;500,250,250,1000;1000,500,500;2000]'  # 100
    filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-6249-2.wav', '[1000,500,500;2000;250,250,500,500,500;2000]'  # 100
    # filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-6249-3.wav', '[2000;250,250,250,250,1000;2000;500,500,1000]'  # 100
    # filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-6249-4.wav', '[1000,250,250,250,250;2000;1000,500,500;2000]'  # 100

    # filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-6285-1.wav', '[1000,1000;500,250,250,1000;1000,500,500;2000]'  # 100
    # filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-6285-2.wav', '[1000,500,500;2000;250,250,500,500,500;2000]'  # 100
    # filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-6285-3.wav', '[2000;250,250,250,250,1000;2000;500,500,1000]'  # 100
    # filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.17MP3/旋律/小学8题20190717-6285-4.wav', '[1000,250,250,250,250;2000;1000,500,500;2000]'  # 100

    filename,onset_code = 'F:/项目/花城音乐项目/样式数据/3.06MP3/节奏/节1.2(100).wav','[1000,1000;2000;1000,500,500;2000]'
    # filename, onset_code = 'F:/项目/花城音乐项目/样式数据/8.28MP3/节奏/5.wav', '[1000,1000;500,250,250,1000;1000,500,500;2000]'
    # filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.18MP3/旋律/小学8题20190718-9728-4.wav', '[1000,250,250,250,250;2000;1000,500,500;2000]'  # 100
    filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.18MP3/旋律/小学8题20190718-9728-2.wav', '[1000,500,500;2000;250,250,500,500,500;2000]'  # 100
    filename, onset_code = 'F:/项目/花城音乐项目/样式数据/2019MP3/节奏/小学8题20190718-9728-1.wav', '[1000,1000;500,250,250,1000;1000,500,500;2000]'  # 100
    filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.18MP3/旋律/小学8题20190718-7649-4.wav', '[1000,250,250,250,250;2000;1000,500,500;2000]'
    filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.18MP3/旋律/小学8题20190718-7649-3.wav', '[2000;250,250,250,250,1000;2000;500,500,1000]'
    filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.18MP3/旋律/小学8题20190718-7649-2.wav', '[1000,500,500;2000;250,250,500,500,500;2000]'
    filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.18MP3/旋律/小学8题20190717-4856-3.wav', '[2000;250,250,250,250,1000;2000;500,500,1000]'  # 通过前后比例修正的例子
    # filename, onset_code = 'F:/项目/花城音乐项目/样式数据/2019MP3/节奏/小学8题20190702-2647-1.wav', '[1000,1000;500,250,250,1000;1000,500,500;2000]'  #
    # filename, onset_code = 'F:/项目/花城音乐项目/样式数据/2019MP3/节奏/中学8题20190701-1547-1.wav', '[1000,1000;500,250,250,1000;1000,500,500;2000]'
    # filename, onset_code = 'F:/项目/花城音乐项目/样式数据/3.06MP3/节奏/节1.2(100).wav', '[1000,1000;2000;1000,500,500;2000]'
    # filename, onset_code = 'F:/项目/花城音乐项目/样式数据/9.04MP3/节奏/1.wav', '[500,250,250,500,500;1500,500;1000,1000;2000]'
    # filename, onset_code = 'F:/项目/花城音乐项目/样式数据/9.04MP3/节奏/3.wav', '[500,1000,500;2000;500,250,250,500,500;2000]'
    # filename, onset_code = 'F:/项目/花城音乐项目/样式数据/9.04MP3/节奏/4.wav', '[500,1000,500;2000;500,500,500,250,250;2000]'
    # filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.18MP3/旋律/小学8题20190717-4856-1.wav', '[1000,1000;500,250,250,1000;1000,500,500;2000]'
    # filename, onset_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/小学8题20190624-3898-3.wav', '[2000;250,250,250,250,1000;2000;500,500,1000]'
    # filename, onset_code = 'F:/项目/花城音乐项目/样式数据/9.08MP3/节奏/xj2.wav', '[1000,500,500;2000;250,250,500,500,500;2000]'
    # filename, onset_code = 'F:/项目/花城音乐项目/样式数据/9.08MP3/节奏/zj4.wav', '[500,1000,500;2000;500,500,500,250,250;2000]'

    print("rhythm_code is {}".format(rhythm_code))
    print("pitch_code is {}".format(pitch_code))
    # plt, total_score, onset_score, note_scroe, detail_content = draw_plt(filename, rhythm_code, pitch_code)
    # plt.show()
    # plt.clf()
    code = parse_onset_code(onset_code)
    code = [int(x) for x in code]
    rms,rms_diff, sig_ff, _ = get_rms_max_indexs_for_onset(filename,onset_code)
    max_indexs = get_best_max_index(filename, onset_code)
    # print("0 max_indexs is {},size is {}".format(max_indexs, len(max_indexs)))
    start, end, total_length = get_start_end_length_by_max_index(max_indexs, filename)
    max_indexs = [x for x in max_indexs if x > start - 5 and x < end - 5]
    max_indexs.append(end)
    max_indexs.sort()
    # print("end is{},total_length is {} v is {}".format(end,total_length,8000/total_length))
    print("1 max_indexs is {},size is {}".format(max_indexs, len(max_indexs)))
    print("1 max_indexs_diff is {},size is {}".format(np.diff(max_indexs), len(max_indexs)-1))
    # types, real_types = get_offset_for_each_onsets_by_speed(max_indexs, onset_code)
    # offset_indexs = [i for i in range(len(types)) if np.abs(types[i] - real_types[i]) > 125]
    # types = get_onset_type(max_indexs, onset_code)
    # index_diff = np.diff(max_indexs)
    # print("index_diff is {},size is {}".format(index_diff, len(index_diff)))
    # vs = [int(types[i])/index_diff[i] for i in range(len(index_diff))]
    # print("vs is {},size is {}".format(vs, len(vs)))
    # print("vs mean is {}".format(np.mean(vs)))
    # print("types is {},size is {}".format(types, len(types)))
    # types_tmp = [int(d * np.mean(vs)) for d in index_diff]
    # print("types_tmp is {},size is {}".format(types_tmp, len(types_tmp)))
    # print("code is {},size is {}".format(code, len(code)))


    print("========================================================================")
    total_score, detail = calculate_score(max_indexs, onset_code,end)
    print("total_score is {}".format(total_score))
    print("detail is {}".format(detail))
    plt.clf()
    draw_pic = draw_plt(filename,onset_code,rms,sig_ff,max_indexs,start,end)

    draw_pic.show()

    # test_batch_samples()
    dir = 'F:/项目/花城音乐项目/样式数据/2019MP3/节奏/'
    test_all(dir)
