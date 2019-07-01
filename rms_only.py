# -*- coding: UTF-8 -*-
import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
from note_lines_helper import *
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

def get_code(index,type):

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
    if filename.find("节奏10") >= 0 or filename.find("节奏十") >= 0 or filename.find("节奏题十") >= 0 or filename.find("节奏题10") >= 0 or filename.find("节10") >= 0:
        return 9
    elif filename.find("节奏1") >= 0 or filename.find("节奏一") >= 0 or filename.find("节奏题一") >= 0 or filename.find("节奏题1") >= 0 or filename.find("节1") >= 0:
        return 0
    elif filename.find("节奏2") >= 0 or filename.find("节奏二") >= 0 or filename.find("节奏题二") >= 0 or filename.find("节奏题2") >= 0 or filename.find("节2") >= 0:
        return 1
    elif filename.find("节奏3") >= 0 or filename.find("节奏三") >= 0 or filename.find("节奏题三") >= 0 or filename.find("节奏题3") >= 0 or filename.find("节3") >= 0:
        return 2
    elif filename.find("节奏4") >= 0 or filename.find("节奏四") >= 0 or filename.find("节奏题四") >= 0 or filename.find("节奏题4") >= 0 or filename.find("节4") >= 0:
        return 3
    elif filename.find("节奏5") >= 0 or filename.find("节奏五") >= 0 or filename.find("节奏题五") >= 0 or filename.find("节奏题5") >= 0 or filename.find("节5") >= 0:
        return 4
    elif filename.find("节奏6") >= 0 or filename.find("节奏六") >= 0 or filename.find("节奏题六") >= 0 or filename.find("节奏题6") >= 0 or filename.find("节6") >= 0:
        return 5
    elif filename.find("节奏7") >= 0 or filename.find("节奏七") >= 0 or filename.find("节奏题七") >= 0 or filename.find("节奏题7") >= 0 or filename.find("节7") >= 0:
        return 6
    elif filename.find("节奏8") >= 0 or filename.find("节奏八") >= 0 or filename.find("节奏题八") >= 0 or filename.find("节奏题8") >= 0 or filename.find("节8") >= 0:
        return 7
    elif filename.find("节奏9") >= 0 or filename.find("节奏九") >= 0 or filename.find("节奏题九") >= 0 or filename.find("节奏题9") >= 0 or filename.find("节9") >= 0:
        return 8
    else:
        return -1

def get_onsets_index_by_filename_rhythm(filename):
    if filename.find("旋律10") >= 0 or filename.find("旋律十") >= 0 or filename.find("视唱十") >= 0 or filename.find("视唱10") >= 0 or filename.find("旋10") >= 0:
        return 9
    elif filename.find("旋律1") >= 0 or filename.find("旋律一") >= 0 or filename.find("视唱一") >= 0 or filename.find("视唱1") >= 0 or filename.find("旋1") >= 0:
        return 0
    elif filename.find("旋律2") >= 0 or filename.find("旋律二") >= 0 or filename.find("视唱二") >= 0 or filename.find("视唱2") >= 0 or filename.find("旋2") >= 0:
        return 1
    elif filename.find("旋律3") >= 0 or filename.find("旋律三") >= 0 or filename.find("视唱三") >= 0 or filename.find("视唱3") >= 0 or filename.find("旋3") >= 0:
        return 2
    elif filename.find("旋律4") >= 0 or filename.find("旋律四") >= 0 or filename.find("视唱四") >= 0 or filename.find("视唱4") >= 0 or filename.find("旋4") >= 0:
        return 3
    elif filename.find("旋律5") >= 0 or filename.find("旋律五") >= 0 or filename.find("视唱五") >= 0 or filename.find("视唱5") >= 0 or filename.find("旋5") >= 0:
        return 4
    elif filename.find("旋律6") >= 0 or filename.find("旋律六") >= 0 or filename.find("视唱六") >= 0 or filename.find("视唱6") >= 0 or filename.find("旋6") >= 0:
        return 5
    elif filename.find("旋律7") >= 0 or filename.find("旋律七") >= 0 or filename.find("视唱七") >= 0 or filename.find("视唱7") >= 0 or filename.find("旋7") >= 0:
        return 6
    elif filename.find("旋律8") >= 0 or filename.find("旋律八") >= 0 or filename.find("视唱八") >= 0 or filename.find("视唱8") >= 0 or filename.find("旋8") >= 0:
        return 7
    elif filename.find("旋律9") >= 0 or filename.find("旋律九") >= 0 or filename.find("视唱九") >= 0 or filename.find("视唱9") >= 0 or filename.find("旋9") >= 0:
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

def get_start_and_max_indexs_by_rms(filename):
    y, sr = librosa.load(filename)
    rms = librosa.feature.rmse(y=y)[0]
    rms = [x / np.std(rms) for x in rms]
     # Savitzky-Golay filter 平滑
    from scipy.signal import savgol_filter

    rms_filtered = savgol_filter(rms, 7, 1)  # window size 51, polynomial order 3
    rms_filtered = [x / np.max(rms_filtered) for x in rms_filtered]
    rms_max = [i for i in range(1,len(rms_filtered)-1) if rms_filtered[i] > rms_filtered[i-1] and rms_filtered[i] > rms_filtered[i+1] and rms_filtered[i] > 0.2]
    starts = [i for i in range(1,len(rms_filtered)-1) if rms_filtered[i] < rms_filtered[i-1] and rms_filtered[i] < rms_filtered[i+1]]
    return rms,rms_filtered,rms_max,starts

def draw_rms_max(filename,rhythm_code,pitch_code):
    y, sr = librosa.load(filename)
    rms = librosa.feature.rmse(y=y)[0]
    rms = [x / np.std(rms) for x in rms]
    time = librosa.get_duration(filename=filename)
    print("time is {}".format(time))
    CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref=np.max)
    CQT = np.where(CQT > -22, np.max(CQT), np.min(CQT))
    w, h = CQT.shape
    print("w.h is {},{}".format(w, h))


    times = librosa.frames_to_time(np.arange(len(rms)))


    # Savitzky-Golay filter 平滑
    from scipy.signal import savgol_filter

    rms_filtered = savgol_filter(rms, 7, 1)  # window size 51, polynomial order 3
    rms_filtered = [x / np.max(rms_filtered) for x in rms_filtered]
    rms_max = [i for i in range(1,len(rms_filtered)-1) if rms_filtered[i] > rms_filtered[i-1] and rms_filtered[i] > rms_filtered[i+1] and rms_filtered[i] > 0.2]
    rms_max_time = librosa.frames_to_time(rms_max, sr=sr)

    plt.plot(times, rms)
    plt.plot(times, rms_filtered)
    plt.xlim(0, np.max(times))
    plt.vlines(rms_max_time, 0, np.max(rms), color='b', linestyle='dashed')

    plt.show()


if __name__ == "__main__":
    #y, sr = load_and_trim('F:/项目/花城音乐项目/样式数据/ALL/旋律/1.31MP3/旋律1.100分.wav')
    filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律2.1(80).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/ALL/旋律/1.31MP3/旋律3.100分.wav'
    #filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律一（9）（100）.wav'
    #filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律一（14）（95）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋律五（3）（63）.wav'
    #filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏一（4）（96）.wav'
    #filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋3王（80）.wav'
    #filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋4谭（95）.wav'
    #filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋4文(75).wav'
    #filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋8录音1(80).wav'
    #filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1.3(93).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋3罗（80）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律1_40312（95）.wav'
    #filename = 'e:/test_image/m1/A/旋律1_40312（95）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋3罗（80）.wav'
    #filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋律十（2）（80）.wav'

    #filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律8录音3(95).wav'
    #filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1王（98）.wav'
    #filename = 'F:/项目/花城音乐项目/样式数据/Archive/dada1.wav'
    #filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1王（98）.wav'

    # filename = 'F:/项目/花城音乐项目/样式数据/1-2/旋律mp3/旋律8.100分.mp3'
    # filename = 'F:/项目/花城音乐项目/样式数据/1-2/旋律mp3/旋律7.100分.mp3'
    # filename = 'F:/项目/花城音乐项目/样式数据/1-2/旋律mp3/旋律5.100分.mp3'
    # filename = 'F:/项目/花城音乐项目/样式数据/1-2/旋律mp3/旋律六.5（100）.mp3'
    #filename = 'F:/项目/花城音乐项目/样式数据/1-2/旋律mp3/旋律6.75分.mp3'
    #filename =  'F:/项目/花城音乐项目/样式数据/1-2/旋律mp3/旋律1.40分.mp3'

    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1.2(92).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1谭（98）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1王（98）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋3.3(96).wav'
    #filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋4谭（95）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋律8录音3(95).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1谭（98）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1王（98）.wav'
    #filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋8文(58).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋律四（1）（20）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋4王（56）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋4欧(25).wav'


    filename = 'F:/项目/花城音乐项目/样式数据/1-2/旋律/旋律八（9）(90).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/1-2/旋律/旋律二（2）（90分）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/1-2/旋律/旋律九（4）(95).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/1-2/旋律/旋律三（2）（90分）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/1-2/旋律/旋律四.1（100）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/1-2/旋律/旋律四.3（100）.wav'
    #filename = 'F:/项目/花城音乐项目/样式数据/1-2/旋律/旋律十（5）(50).wav'
    #filename = 'F:/项目/花城音乐项目/样式数据/1-2/旋律/旋律七(5)（55）.wav'
    #filename = 'F:/项目/花城音乐项目/样式数据/1-2/旋律/旋律1.90分.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/1-2/旋律/旋律四.10（100）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/1-2/旋律/旋律三（3）（80分）.wav'
    #filename = 'F:/项目/花城音乐项目/样式数据/1-2/旋律/旋律三（8）(80).wav'
    #filename = 'F:/项目/花城音乐项目/样式数据/1-2/旋律/旋律二（2）（90分）.wav'
    #filename = 'F:/项目/花城音乐项目/样式数据/1-2/旋律/旋律三.10（100）.wav'
    #filename = 'F:/项目/花城音乐项目/样式数据/1-2/旋律/旋律一.6（100）.wav'
    #filename = 'F:/项目/花城音乐项目/样式数据/1-2/旋律/旋律九（6）(50).wav'

    filename = 'F:/项目/花城音乐项目/样式数据/6.21MP3/旋律/两只老虎20190621-7278.wav' #只是节奏基本对，旋律全错 score, onset_score, note_scroe is 57,37,20
    #filename = 'F:/项目/花城音乐项目/样式数据/6.21MP3/旋律/两只老虎20190621-6314.wav' #唱的不是同一首歌，旋律也相差较大 score, onset_score, note_scroe is 50,32,18
    #filename = 'F:/项目/花城音乐项目/样式数据/6.21MP3/旋律/20190621-3533.wav' #节奏对，旋律全错  score, onset_score, note_scroe is 45,32,13
    #filename = 'F:/项目/花城音乐项目/样式数据/6.21MP3/旋律/两只老虎20190621-9983.wav' #节奏对，旋律错 score, onset_score, note_scroe is 60,37,23
    #filename = 'F:/项目/花城音乐项目/样式数据/6.21MP3/旋律/20190621-3858.wav'# 速度偏快，节奏旋律基本正确 score, onset_score, note_scroe is 61,36,25
    #filename = 'F:/项目/花城音乐项目/样式数据/6.21MP3/旋律/两只老虎20190621-6264.wav' #唱速不稳定，旋律正确，节奏错误 score, onset_score, note_scroe is 44,33,11
    #filename = 'F:/项目/花城音乐项目/样式数据/6.21MP3/旋律/20190621-3192.wav' #只是节奏基本对，旋律全错 score, onset_score, note_scroe is 50,32,18
    #filename = 'F:/项目/花城音乐项目/样式数据/6.21MP3/旋律/两只老虎20190621-6143.wav' #速度偏快，节奏旋律基本正确 score, onset_score, note_scroe is 10,4,6
    #filename = 'F:/项目/花城音乐项目/样式数据/6.21MP3/旋律/两只老虎20190621-9805.wav' #节奏对，旋律错，score, onset_score, note_scroe is 52,32,20

    #filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1谭（98）.wav'
    #filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1.1(96).wav'
    #filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋7卉(55).wav'
    #filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋3谭（98）.wav'

    # filename = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/两只老虎20190624-2939.wav'
    #
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1录音3(90).wav'
    #filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋9.1(73).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/小学8题20190624-3898-2.wav'

    result_path = 'e:/test_image/n/'
    plt.close()
    rhythm_code = '[500,500,500,500;500,500,500,500;500,500,1000;500,500;1000]'
    pitch_code = '[1,2,3,1,1,2,3,1,3,4,5,3,4,5]'
    type_index = get_onsets_index_by_filename_rhythm(filename)
    rhythm_code = get_code(type_index, 2)
    pitch_code = get_code(type_index, 3)
    #rhythm_code = '[1000,1000;500,500,1000;500,250,250,500,500;2000]'
    #melody_code = '[5,5,3,2,1,2,2,3,2,6-,5-]'
    print("rhythm_code is {}".format(rhythm_code))
    print("pitch_code is {}".format(pitch_code))
    #draw_rms_max(filename,rhythm_code,pitch_code)

    rms, rms_filtered, rms_max,starts = get_start_and_max_indexs_by_rms(filename)
    times = librosa.frames_to_time(np.arange(len(rms)))
    rms_max_time = librosa.frames_to_time(rms_max)
    starts_time = librosa.frames_to_time(starts)
    plt.plot(times, rms)
    plt.plot(times, rms_filtered)
    plt.xlim(0, np.max(times))
    plt.vlines(rms_max_time, 0, np.max(rms), color='b', linestyle='dashed')
    plt.vlines(starts_time, 0, np.max(rms), color='r', linestyle='dashed')

    plt.show()