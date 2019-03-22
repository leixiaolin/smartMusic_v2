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
from find_mismatch import *
from grade import *
from vocal_separation import *
from viterbi import *
from create_base import *
import re
score = 0
# save_path = 'F:/项目/花城音乐项目/参考代码/tensorflow_models_nets-master/onsets/test/'
# src_path = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/'
save_path = './onsets/test/'
src_path = './mp3/1.31WAV/'
error_info_txt = './onsets/1.31error_info.txt'

# save_path = ''
tmp = ['A','B','C','D','E']
# dis_dir = 'F:/项目/花城音乐项目/参考代码/tensorflow_models_nets-master/onsets/test'
# scr_dir = 'F:/项目/花城音乐项目/参考代码/tensorflow_models_nets-master/onsets/test'
dis_dir = ''

new_old_txt = './rhythm/new_and_old.txt'
codes = np.array(['[1000,1000;2000;1000,500,500;2000]',
                  '[2000;1000,1000;500,500,1000;2000]',
                  '[1000,1000;500,500,1000;1000,1000;2000]',
                  '[1000,--(1000);1000,--(1000);500,250,250,1000;--(1000),1000]',
                  '[500;1000,500,1000,500;500,500,500,250,250,500,500;250,250,500,500,1000]',
                  '[1000,--(1000);1000,--(1000);1000,-(500),500;1000,1000]',
                  '[750,250,500,500,500,-(500);500,1000,500,500,-(500);750,250,500,500,500,-(500)]',
                  '[500,1000,500,500,250,250;1000,500,750,250,500;3000]',
                  '[500,500,500;1000,500;500,500,500;1500;500,500,500;1000,500;500;1000;1500]',
                  '[500,500,1000;500,500;1000;375,125,250,250,375,125,250,250;500,500,1000]'])


def load_and_trim(path):
    audio, sr = librosa.load(path)
    energy = librosa.feature.rmse(audio)
    frames = np.nonzero(energy >= np.max(energy) / 5)
    indices = librosa.core.frames_to_samples(frames)[1]
    audio = audio[indices[0]:indices[-1]] if indices.size else audio[0:0]

    return audio, sr

def get_total_frames_number(path):
    audio, sr = librosa.load(path)
    energy = librosa.feature.rmse(audio)
    frames = np.nonzero(energy >= np.max(energy) / 5)

    total = frames[1][-1]

    return total



def list_all_files(rootdir):
    import os
    _files = []
    list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
    for i in range(0,len(list)):
           path = os.path.join(rootdir,list[i])
           if os.path.isdir(path):
              _files.extend(list_all_files(path))
           if os.path.isfile(path):
              _files.append(path)
    return _files


#将第一阶段的文件遍历出来
#_k = filter(lambda x:re.compile(r'val.txt').search(x),_fs)


def clear_dir(dis_dir,scr_dir):
    for i in tmp:
        d_dir = dis_dir + '/' + i
        shutil.rmtree(d_dir)
        os.mkdir(d_dir)

        s_dir = scr_dir + '/' + i
        shutil.rmtree(s_dir)
        os.mkdir(s_dir)

#清空文件夹
clear_dir(save_path,save_path)

# 保存新文件名与原始文件的对应关系
files_list = []
files_list_a = []
files_list_b = []
files_list_c = []
files_list_d = []
new_old_txt = './onsets/best_dtw.txt'
files = list_all_files(src_path)

print(files)
index = 0
# 用于分析人工打分与算法打分误差
total_error = 0   # 误差分数
higher_num = 0 # 打分偏高数量
lower_num = 0 # 打分偏低数量
higher_error = 0 # 打分高的误差
lower_error = 0 # 打分低的误差

# 测试单个文件
#files = ['F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏六（5）（80）.wav']
#files = ['F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏8_40210（30）.wav']
for filename in files:
    print(filename)
    if filename.find('wav') <= 0:
        continue
    elif filename.find('shift') > 0:
        continue
    else:
        index = index + 1

    # 获取人工打分
    manu_score = re.sub("\D", "", filename)
    if str(manu_score).find("100") > 0:
        manu_score = 100
    else:
        manu_score = int(manu_score) % 100

    type_index = get_onsets_index_by_filename(filename)
    y, sr = load_and_trim(filename)
    #y, sr = librosa.load(filename)
    #y, sr = get_foreground(y, sr) # 分离前景音
    silence_threshold = 0.2
    need_vocal_separation = check_need_vocal_separation(y, silence_threshold)
    # if need_vocal_separation:
    #     y, sr = get_foreground(y, sr)  # 分离前景音
    total_frames_number = get_total_frames_number(filename)
    if len(y) > 0:

        # 识别的实唱节拍
        onsets_frames = get_onsets_frames_for_jz(filename)
        # 在此处赋值防止后面实线被移动找不到强度
        recognize_y = onsets_frames.copy()
        onsets_frames_strength = librosa.onset.onset_strength(y=y, sr=sr)
        #onsets_frames = get_onsets_by_all_v2(y, sr,len(codes[type_index])+2)
        if len(onsets_frames) < 3:
            continue


        #print("onsets_frames is {}".format(onsets_frames))



        # 标准节拍时间点
        base_frames = onsets_base_frames(codes[type_index],total_frames_number)
        #print("base_frames is {}".format(base_frames))

        min_d, best_y,onsets_frames = get_dtw_min(onsets_frames, base_frames, 65)
        base_onsets = librosa.frames_to_time(best_y, sr=sr)
        #print("base_onsets is {}".format(base_onsets))

        # 节拍时间点
        onstm = librosa.frames_to_time(onsets_frames, sr=sr)
        #print("onstm is {}".format(onstm))
        duration = librosa.get_duration(y, sr=sr)  # 获取音频时长
        #print("duration is {}".format(duration))

        #节拍数之差
        diff_real_base = len(onsets_frames) - len(base_frames)


        #librosa.display.waveplot(y, sr=sr)
        # plt.show()

        plt.vlines(onstm, -1 * np.max(y), np.max(y), color='b', linestyle='solid')
    plt.vlines(base_onsets,  -1*np.max(y),np.max(y), color='r', linestyle='dashed')
    #plt.vlines(base_onsets[-1],  -1*np.max(y),np.max(y), color='white', linestyle='dashed')

    standard_y = best_y.copy()

    code = get_code(type_index, 1)
    modify_recognize_y = recognize_y
    ex_recognize_y = []
    # 多唱的情况
    if len(standard_y) < len(recognize_y):
        _, ex_recognize_y = get_mismatch_line(standard_y.copy(), recognize_y.copy())
        modify_recognize_y = [x for x in recognize_y if x not in ex_recognize_y]
        min_d = get_deviation(standard_y, modify_recognize_y, code)
    # 漏唱的情况
    if len(standard_y) > len(recognize_y):
        _, lost_standard_y = get_mismatch_line(recognize_y.copy(), standard_y.copy())
        modify_standard_y = [x for x in standard_y if x not in lost_standard_y]
        min_d = get_deviation(modify_standard_y, recognize_y, code)

    # 打印多唱的节拍
    if len(ex_recognize_y)>0:
        ex_recognize_y_time = librosa.frames_to_time(ex_recognize_y)
        plt.vlines(ex_recognize_y_time, -1 * np.max(y), np.max(y), color='black', linestyle='solid')
    # # 计算成绩测试
    print('偏移分值为：{}'.format(min_d))
    #plt.text(0.2, 20, 'test：{}'.format(min_d))
    #plt.text(0.2, 0.2, '偏移分值为:' + str(round(min_d, 2)))
    score, lost_score, ex_score, min_d = get_score1(standard_y, recognize_y, len(base_frames), onsets_frames_strength, min_d)
    print('最终得分为：{}'.format(score))
    standard_y, recognize_y = get_mismatch_line(standard_y, recognize_y)
    lost_num, ex_frames = get_wrong(standard_y, recognize_y)

    if lost_num:
        print('漏唱了' + str(lost_num) + '句')
    elif len(ex_frames) > 1:
        print('多唱的帧 is {}'.format(ex_frames))
        ex_frames_time = librosa.frames_to_time(ex_frames, sr=sr)
        plt.vlines(ex_frames_time, -1 * np.max(y), np.max(y), color='black', linestyle='solid')
    else:
        print('节拍数一致')
    lost_score, ex_score = get_scores(standard_y, recognize_y, len(base_frames), onsets_frames_strength)
    # print("lost_score, ex_score is : {},{}".format(lost_score, ex_score))

    print("lost_score, ex_score,min_d is : {},{},{}".format(lost_score, ex_score, min_d))

    #plt.show()
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(4, 4)
    if "." in filename:
        Filename = filename.split(".")[0]
    plt.axis('off')
    plt.axes().get_xaxis().set_visible(False)
    plt.axes().get_yaxis().set_visible(False)
    #plt.rcParams['savefig.dpi'] = 300  # 图片像素
    #plt.figure(figsize=(10, 10))
    #plt.rcParams['figure.dpi'] = 300  # 分辨率
    dirname, filename = os.path.split(filename)

    if str(score).find("100") > 0:
        score = 100
    else:
        score = int(score) % 100

    if int(score) >=90:
        grade = 'A'
        savepath = save_path + 'A/'
        files_list_a.append([filename.split('.wav')[0] + '-' + grade,score])
    elif int(score) >= 75:
        grade = 'B'
        savepath = save_path + 'B/'
        files_list_b.append([filename.split('.wav')[0] + '-' + grade, score])
    elif int(score) >=60:
        grade = 'C'
        savepath = save_path + 'C/'
        files_list_c.append([filename.split('.wav')[0] + '-' + grade, score])
    elif int(score) >=1:
        grade = 'D'
        savepath = save_path + 'D/'
        files_list_d.append([filename.split('.wav')[0] + '-' + grade, score])
    else:
        grade = 'E'
        savepath = save_path + 'E/'


    saveFileName = filename.split('.wav')[0] + '-' + grade
    #saveFileName = str(score) + '-' + grade
    file_sum = os.listdir(savepath)
    #saveFileName = str(len(file_sum)+1) + '-' + filename.split(".wav")[0] + '-' + saveFileName
    #saveFileName = str(len(file_sum) + 1) + '-' + saveFileName
    saveFileName = saveFileName
    plt.savefig(savepath + saveFileName+ '_'+ str(score) + '.png',  bbox_inches='tight', pad_inches=0)
    plt.clf()
    saveFileName = ''

    '''
    分析人工与算法打分的误差
    '''
    # 计算总误差
    total_error += abs(manu_score-score)

    # 计算分数偏高或偏低的情况
    if score-manu_score > 7:
        higher_num += 1
        higher_error += score-manu_score
    elif manu_score-score > 7:
        lower_num += 1
        lower_error += manu_score-score

# 写入误差信息
error_info = '人工与算法打分总误差为：'+str(total_error)+'分  打分偏高的有：'+str(higher_num)+'个 打分偏低的有：'+str(lower_num)+'个'
error_info += '\n分数偏高误差为：'+str(higher_error)+'分数偏低误差为'+str(lower_error)
error_info += '\n平均误差为：'+str(total_error/len(files))


f = open(error_info_txt,'w')
f.write(error_info)

if len(files_list_a) == 0 and len(files_list_b) == 0:
    files_list = np.vstack((files_list_c,files_list_d))
elif len(files_list_c) == 0 and len(files_list_d) == 0:
    files_list = np.vstack((files_list_a,files_list_b))
else:
    t1 = np.vstack((files_list_a,files_list_b))
    t2 = np.vstack((files_list_c,files_list_d))
    files_list = np.vstack((t1,t2))

write_txt(files_list, new_old_txt, mode='w')

# 先获取多唱漏唱的情况




