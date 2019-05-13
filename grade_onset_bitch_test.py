import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
from find_mismatch import *
from create_labels_files import *
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


if __name__ == "__main__":

    filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏8_40210（30）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏1卢(100).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/节奏/节4欧(95).wav'




    plt.close()
    type_index = get_onsets_index_by_filename(filename)
    onset_code = get_code(type_index, 1)
    print("onset_code is {}".format(onset_code))
    score, lost_score, ex_score, min_d = get_score_jz(filename,onset_code)
    print("score, lost_score, ex_score, min_d is {},{},{},{}".format(score, lost_score, ex_score, min_d))

    date = '3.06'
    src_path = 'F:/项目/花城音乐项目/样式数据/3.19MP3/节奏/'
    new_old_txt = './onsets/' + date + 'best_dtw.txt'
    # 保存新文件名与原始文件的对应关系
    files_list = []
    files_list_a = []
    files_list_b = []
    files_list_c = []
    files_list_d = []
    dir_list = ['F:/项目/花城音乐项目/样式数据/3.06MP3/节奏/']
    #dir_list = ['F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/']
    #dir_list = ['e:/test_image/m1/A/']
    dir_list = []
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
        # file_list = ['视唱1-01（95）.wav']
        file_total = len(file_list)
        for filename in file_list:
            print(filename)
            if filename.find('wav') <= 0:
                continue
            elif filename.find('shift') > 0:
                continue
            type_index = get_onsets_index_by_filename(dir + filename)
            onset_code = get_code(type_index, 1)
            score, lost_score, ex_score, min_d = get_score_jz(dir + filename,onset_code)
            print("score, lost_score, ex_score, min_d is {},{},{},{}".format(score, lost_score, ex_score, min_d))

            if int(score) >= 90:
                grade = 'A'
                files_list_a.append([filename + ' - ' + grade, score, lost_score, ex_score, min_d])
            elif int(score) >= 75:
                grade = 'B'
                files_list_b.append([filename + ' - ' + grade, score, lost_score, ex_score, min_d])
            elif int(score) >= 60:
                grade = 'C'
                files_list_c.append([filename + ' - ' + grade, score, lost_score, ex_score, min_d])
            elif int(score) >= 1:
                grade = 'D'
                files_list_d.append([filename + ' - ' + grade, score, lost_score, ex_score, min_d])
            else:
                grade = 'E'

        t1 = np.append(files_list_a, files_list_b).reshape(len(files_list_a) + len(files_list_b), 5)
        t2 = np.append(files_list_c, files_list_d).reshape(len(files_list_c) + len(files_list_d), 5)
        files_list = np.append(t1, t2).reshape(len(t1) + len(t2), 5)

        write_txt(files_list, new_old_txt, mode='w')
    print("file_total,yes_total is {},{},{},{},{}".format(file_total, total_10, total_15, total_20,
                                                          total_10 / file_total))