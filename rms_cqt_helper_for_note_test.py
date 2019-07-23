# -*- coding: UTF-8 -*-
import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import os
from rms_helper import *
from cqt_helper import *
from rms_cqt_helper_for_note import *

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
def draw_detail(filename,rhythm_code,pitch_code):
    score, onset_score, note_score,standard_y, start_indexs, detail_content = cal_score_onset_and_note(filename, rhythm_code,pitch_code)
    #print("start_indexs is {},size is {}".format(start_indexs, len(start_indexs)))


    plt.subplot(4, 1, 1)
    plt.title(filename)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    y, sr = librosa.load(filename)
    CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref=np.max)
    CQT = np.where(CQT > -22, np.max(CQT), np.min(CQT))
    CQT = signal.medfilt(CQT, (3, 3))  # 二维中值滤波
    librosa.display.specshow(CQT, x_axis='time')
    # start_indexs = check_starts_with_max_index(filename)
    start_indexs_time = librosa.frames_to_time(start_indexs)
    plt.vlines(start_indexs_time, 0, 84, color='b', linestyle='dashed')
    start, end, length = get_onset_frame_length(filename,rhythm_code)
    start_time = librosa.frames_to_time(start)
    end_time = librosa.frames_to_time(end)
    plt.vlines(start_time, 0, 40, color='r', linestyle='dashed')
    plt.vlines(end_time, 0, 40, color='r', linestyle='dashed')


    print("score, onset_score, note_score is {},{},{}".format(score, onset_score, note_score))
    print("detail_content is {}".format(detail_content))
    if len(start_indexs) == 0:
        return plt

    plt.subplot(4, 1, 2)
    rms, sig_ff, max_indexs = get_cqt_diff(filename)
    base_frames = onsets_base_frames(rhythm_code, length)
    base_frames = [x - (base_frames[0] - start_indexs[0] - 2) for x in base_frames]
    base_frames_time = librosa.frames_to_time(base_frames)
    plt.vlines(base_frames_time, 0, 84, color='r', linestyle='solid')
    times = librosa.frames_to_time(np.arange(len(rms)))
    plt.xlim(0, np.max(times))


    plt.subplot(4, 1, 3)
    y, sr = librosa.load(filename)
    rms = librosa.feature.rmse(y=y)[0]
    rms = [x / np.std(rms) for x in rms]
    times = range(len(rms))
    plt.plot(times, rms)

    plt.subplot(4, 1, 4)
    rms,rms_diff, sig_ff, max_indexs = get_rms_max_indexs_for_onset(filename)
    #print("=====max_indexs is {},size is {}".format(max_indexs, len(max_indexs)))
    times = librosa.frames_to_time(np.arange(len(rms)))
    plt.plot(times, rms)
    plt.plot(times, sig_ff)
    plt.xlim(0, np.max(times))
    max_index_times = librosa.frames_to_time(max_indexs)
    plt.vlines(max_index_times, 0, np.max(rms), color='r', linestyle='dashed')

    return plt
    #plt.show()

def batch_test(dir_list):
    date = '3.06'
    src_path = 'F:/项目/花城音乐项目/样式数据/3.19MP3/节奏/'
    new_old_txt = './onsets/' + date + 'best_dtw.txt'
    # 保存新文件名与原始文件的对应关系
    files_list = []
    files_list_a = []
    files_list_b = []
    files_list_c = []
    files_list_d = []
    #dir_list = ['F:/项目/花城音乐项目/样式数据/3.06MP3/节奏/']
    #dir_list = ['F:/项目/花城音乐项目/样式数据/1-2/节奏/']
    #dir_list = ['F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/']
    #dir_list = ['e:/test_image/m1/A/']
    #dir_list = []
    total_accuracy = 0
    total_num = 0
    result_path = 'e:/test_image/n/'
    # clear_dir(result_path)
    # 要测试的数量
    test_num = 100
    score = 0
    file_total = 0
    total_10 = 0
    total_15 = 0
    total_20 = 0
    for dir in dir_list:
        file_list = os.listdir(dir)
        # shuffle(file_list)  # 将语音文件随机排列
        #file_list = ['节2罗（75）.wav']
        file_total = len(file_list)
        for filename in file_list:
            print(filename)
            if filename.find('wav') <= 0:
                continue
            elif filename.find('shift') > 0:
                continue

            if filename.find("tune") > 0 or filename.find("add") > 0 or filename.find("shift") > 0:
                score = re.sub("\D", "", filename.split("-")[0])  # 筛选数字
            else:
                score = re.sub("\D", "", filename)  # 筛选数字

            if str(score).find("100") > 0:
                score = 100
            else:
                score = int(score) % 100

            type_index = get_onsets_index_by_filename_rhythm(dir + filename)
            rhythm_code = get_code(type_index, 2)
            pitch_code = get_code(type_index, 3)
            rhythm_code, pitch_code =  '[500,250,250,500,500;250,250,250,250,500,500;500,250,250,500,500;500,250,250,1000]', '[5,5,6,5,3,4,5,4,5,4,2,3,3,4,3,1,2,3,5,1]'
            total_score, onset_score, note_score, standard_y, start_indexs, detail_content = cal_score_onset_and_note(dir + filename,rhythm_code,pitch_code)
            print("score, onset_score, note_score is {},{},{}".format(total_score, onset_score, note_score))
            print("detail_content is {}".format(detail_content))

            if int(score) < 60:
                if total_score > 70:
                    files_list_d.append([filename, total_score, onset_score, note_score, ""])
                    plt = draw_detail(dir + filename,rhythm_code,pitch_code)
                    plt.savefig(result_path + "/" + filename + '-' + str(total_score) + '-' + str(onset_score) + '-' + str(note_score) + '.jpg', bbox_inches='tight', pad_inches=0)
                    plt.clf()
            else:
                if np.abs(total_score - int(score)) > 15:
                    files_list_a.append([filename, total_score, onset_score, note_score, ""])
                    plt = draw_detail(dir + filename, rhythm_code,pitch_code)
                    plt.savefig(result_path + "/" + filename + '-' + str(total_score) + '-' + str(onset_score) + '-' + str(note_score) + '.jpg', bbox_inches='tight', pad_inches=0)
                    plt.clf()

            # if int(score) >= 90:
            #     grade = 'A'
            #     files_list_a.append([filename + ' - ' + grade, score, lost_score, ex_score, min_d])
            # elif int(score) >= 75:
            #     grade = 'B'
            #     files_list_b.append([filename + ' - ' + grade, score, lost_score, ex_score, min_d])
            # elif int(score) >= 60:
            #     grade = 'C'
            #     files_list_c.append([filename + ' - ' + grade, score, lost_score, ex_score, min_d])
            # elif int(score) >= 1:
            #     grade = 'D'
            #     files_list_d.append([filename + ' - ' + grade, score, lost_score, ex_score, min_d])
            # else:
            #     grade = 'E'

        t1 = np.append(files_list_a, files_list_b).reshape(len(files_list_a) + len(files_list_b), 5)
        t2 = np.append(files_list_c, files_list_d).reshape(len(files_list_c) + len(files_list_d), 5)
        files_list = np.append(t1, t2).reshape(len(t1) + len(t2), 5)

        write_txt(files_list, new_old_txt, mode='w')

if __name__ == "__main__":
    # y, sr = load_and_trim('F:/项目/花城音乐项目/样式数据/ALL/旋律/1.31MP3/旋律1.100分.wav')
    filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律2.1(80).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/ALL/旋律/1.31MP3/旋律3.100分.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律一（9）（100）.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律一（14）（95）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋律五（3）（63）.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋4王（56）.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1谭（98）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1王（98）.wav'
    # filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋4欧(25).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋4.4(0).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋2.3(95).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋2.4(50).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋2录音1(90).wav'

    result_path = 'e:/test_image/n/'
    plt.close()
    type_index = get_onsets_index_by_filename_rhythm(filename)
    rhythm_code = get_code(type_index, 2)
    pitch_code = get_code(type_index, 3)

    filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/两只老虎20190624-1089.wav', '[500,500,500,500;500,500,500,500;500,500,1000;500,500;1000]', '[1,2,3,1,1,2,3,1,3,4,5,3,4,5]'  # 音准节奏均正确，给分偏低  94
    filename,rhythm_code,pitch_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/两只老虎20190624-1328.wav','[500,500,500,500;500,500,500,500;500,500,1000;500,500;1000]','[1,2,3,1,1,2,3,1,3,4,5,3,4,5]'       #音准节奏均正确，给分偏低  95
    # filename,rhythm_code,pitch_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/两只老虎20190624-1586.wav','[500,500,500,500;500,500,500,500;500,500,1000;500,500;1000]','[1,2,3,1,1,2,3,1,3,4,5,3,4,5]'      #音准节奏均正确，给分偏低  95
    # filename,rhythm_code,pitch_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/两只老虎20190624-2939.wav','[500,500,500,500;500,500,500,500;500,500,1000;500,500;1000]','[1,2,3,1,1,2,3,1,3,4,5,3,4,5]'      #音准节奏均正确，给分偏低 90
    # filename,rhythm_code,pitch_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/两只老虎20190624-7881.wav','[500,500,500,500;500,500,500,500;500,500,1000;500,500;1000]','[1,2,3,1,1,2,3,1,3,4,5,3,4,5]'          # 音准节奏均正确，给分偏低 90
    # filename,rhythm_code,pitch_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/两只老虎20190624-8973.wav','[500,500,500,500;500,500,500,500;500,500,1000;500,500;1000]','[1,2,3,1,1,2,3,1,3,4,5,3,4,5]'       #音准节奏均正确，给分偏低 98
    # filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/小学8题20190624-3898-5.wav', '[1000,1000;500,250,250,1000;1000,500,500;2000]', '[3,1,5,5,6,5,1+,6,3,5]'  #这个给九分多是可以的 86
    # filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/小学8题20190624-3898-6.wav', '[1000,500,500;2000;250,250,500,500,500;2000]', '[6,5,3,6,3,5,3,2,1,6-]'  #故意把最后一个音唱错了，节奏全对,扣0.5左右即可 85
    # filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/小学8题20190624-3898-7.wav', '[2000;250,250,250,250,1000;2000;500,500,1000]', '[6,5,6,3,5,6,3,2,1,6-]'  # 这一条故意唱错了两个音，节奏是对的，这个扣一分即可 72
    # filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/6.24MP3/旋律/小学8题20190624-3898-8.wav', '[1000,250,250,250,250;2000;1000,500,500;2000]', '[1,3,5,1+,6,5,1,3,2,1]'  # 这一条节奏不是太稳，但音高基本正确,9.5分是没问题的 93

    # filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.12MP3/旋律/小学8题20190702-2647-5.wav', '[1000,1000;500,250,250,500;1000,500,500;2000]', '[3,1,5,5,6,5,1+,6,3,5]'  #  准确 96
    # filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.12MP3/旋律/小学8题20190702-2647-6.wav', '[1000,500,500;2000;250,250,500,500,500;2000]', '[6,5,3,6,3,5,3,2,1,6]'  #基本可给满分 59
    # filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.12MP3/旋律/小学8题20190702-2647-7.wav', '[2000;250,250,250,250,1000;2000;500,500,1000]', '[6,5,6,3,5,6,3,2,1,6]'  #第一个节奏应该扣分，最后一个音没唱，应该没分  79
    # filename, rhythm_code, pitch_code = 'F:/项目/花城音乐项目/样式数据/7.12MP3/旋律/小学8题20190702-2647-8.wav', '[1000,250,250,250,250;2000;1000,500,500;2000]', '[1,3,5,1+,6,5,1,3,2,1]'  #节奏全对，旋律错最后一个  86
    # rhythm_code = '[1000,1000;500,500,1000;500,250,250,500,500;2000]'
    # melody_code = '[5,5,3,2,1,2,2,3,2,6-,5-]'
    # print("rhythm_code is {}".format(rhythm_code))
    # print("pitch_code is {}".format(pitch_code))
    # plt, total_score, onset_score, note_scroe, detail_content = draw_plt(filename, rhythm_code, pitch_code)
    # plt.show()
    # plt.clf()
    plt = draw_detail(filename,rhythm_code,pitch_code)
    plt.show()

    dir_list = ['F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/']
    dir_list = ['F:/项目/花城音乐项目/样式数据/6.18MP3/旋律/']
    #batch_test(dir_list)