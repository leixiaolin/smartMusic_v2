# -*- coding: UTF-8 -*-
import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
from create_base import *
from myDtw import *
import scipy.signal as signal
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


def write_txt(content, filename, mode='w'):
    """保存txt数据
    :param content:需要保存的数据,type->list
    :param filename:文件名
    :param mode:读写模式:'w' or 'a'
    :return: void
    """
    with open(filename, mode) as f:
        f.write(content)
def get_rms_max_indexs_for_onset(filename):
    y, sr = librosa.load(filename)
    rms = librosa.feature.rmse(y=y)[0]
    rms = [x / np.std(rms) for x in rms]
    rms_bak = rms.copy();
    rms = list(np.diff(rms))
    rms.insert(0,0)

    # b, a = signal.butter(4, 0.1, analog=False)
    # sig_ff = signal.filtfilt(b, a, rms)

    # Savitzky-Golay filter 平滑
    from scipy.signal import savgol_filter
    sig_ff = savgol_filter(rms, 5, 1)  # window size 51, polynomial order 3
    sig_ff = [x/np.std(sig_ff) for x in sig_ff]
    max_indexs = [i for i in range(1,len(sig_ff)-1) if sig_ff[i]>sig_ff[i-1] and sig_ff[i]>sig_ff[i+1] and sig_ff[i] > np.max(sig_ff)*0.15]
    return rms_bak,rms,sig_ff,max_indexs

def get_start_end_length_by_max_index(max_indexs,onset_code):
    code = parse_onset_code(onset_code)
    code = [int(x) for x in code]
    last_length = int((max_indexs[-1] - max_indexs[0])*code[-1]/(np.sum(code[:-1])))
    start = max_indexs[0]
    end = max_indexs[-1] + last_length
    total_length = end - start
    return start,end,total_length


def get_topN_rms_max_indexs_for_onset(filename,topN):
    rms,rms_diff, sig_ff, max_indexs = get_rms_max_indexs_for_onset(filename)
    rms_on_max_indexs = [rms[x] for x in max_indexs]
    topN_max_indexs = find_n_largest(rms_on_max_indexs, topN)

    result = []
    for x in topN_max_indexs:
        rms_tmp = rms_on_max_indexs[x]
        result.append(rms.index(rms_tmp))
    return result
def find_n_largest(a,topN):
    import heapq

    a = list(a)
    #a = [43, 5, 65, 4, 5, 8, 87]

    re1 = heapq.nlargest(topN, a)  # 求最大的三个元素，并排序
    re1.sort()
    #re2 = map(a.index, heapq.nlargest(total, a))  # 求最大的三个索引    nsmallest与nlargest相反，求最小
    re2 = [i for i,x in enumerate(a) if x in re1]

    #print(re1)
    #print(list(re2))  # 因为re1由map()生成的不是list，直接print不出来，添加list()就行了
    return list(re2)

def parse_onset_code(onset_code):
    code = onset_code
    indexs = []
    code = code.replace(";", ',')
    code = code.replace("[", '')
    code = code.replace("]", '')
    tmp = [x for x in code.split(',')]
    for i in range(len(tmp)):
        if tmp[i].find("(") >= 0:
            indexs.append(i)
    while code.find("(") >= 0:
        code = code.replace("(", '')
        code = code.replace(")", '')
        code = code.replace("-", '')
        code = code.replace("--", '')
    code = [x for x in code.split(',')]
    result = []
    for i in range(len(code)):
        if i in indexs:
            continue
        elif i+1 not in indexs:
            result.append(code[i])
        else:
            t = int(code[i]) + int(code[i+1])
            result.append(t)
    return result

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
    # filename, onset_code = 'F:/项目/花城音乐项目/样式数据/7.01MP3/旋律/小学8题20190625-2251 节拍题三.wav', '[2000;250,250,250,250,1000;2000;500,500,1000]'  #应该接近满分 97
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
    onset_code = get_code(type_index, 1)
    rhythm_code = get_code(type_index, 2)
    pitch_code = get_code(type_index, 3)
    # rhythm_code = '[1000,1000;500,500,1000;500,250,250,500,500;2000]'
    # melody_code = '[5,5,3,2,1,2,2,3,2,6-,5-]'
    print("rhythm_code is {}".format(rhythm_code))
    print("pitch_code is {}".format(pitch_code))
    # plt, total_score, onset_score, note_scroe, detail_content = draw_plt(filename, rhythm_code, pitch_code)
    # plt.show()
    # plt.clf()
    rms,rms_diff, sig_ff, max_indexs = get_rms_max_indexs_for_onset(filename)
    print("max_indexs is {},size is {}".format(max_indexs, len(max_indexs)))

    start, end, total_length = get_start_end_length_by_max_index(max_indexs, onset_code)
    #max_indexs = get_topN_rms_max_indexs_for_onset(filename, 10)
    times = librosa.frames_to_time(np.arange(len(rms)))
    plt.subplot(3,1,1)
    plt.plot(times, rms)
    plt.plot(times, rms_diff)
    plt.plot(times, sig_ff)
    plt.xlim(0, np.max(times))
    max_index_times = librosa.frames_to_time(max_indexs)
    plt.vlines(max_index_times, 0, np.max(rms), color='r', linestyle='dashed')
    start_time = librosa.frames_to_time(start)
    end_time = librosa.frames_to_time(end)
    plt.vlines(start_time, 0, np.max(rms)/2, color='b', linestyle='solid')
    plt.vlines(end_time, 0, np.max(rms)/2, color='b', linestyle='solid')

    plt.subplot(3,1,2)
    base_frames = onsets_base_frames(onset_code, total_length)
    base_frames = [x - (base_frames[0] - max_indexs[0]) for x in base_frames]
    base_frames_times = librosa.frames_to_time(base_frames)
    plt.vlines(base_frames_times, 0, 10, color='r', linestyle='dashed')
    plt.xlim(0, np.max(times))

    plt.subplot(3, 1,3)
    xc, yc = get_matched_onset_frames_compared(max_indexs, base_frames)
    xc_times = librosa.frames_to_time(xc)
    plt.vlines(xc_times, 0, 10, color='r', linestyle='dashed')
    plt.xlim(0, np.max(times))


    plt.show()
