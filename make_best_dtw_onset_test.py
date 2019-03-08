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


score = 0
save_path = 'F:/项目/花城音乐项目/参考代码/tensorflow_models_nets-master/onsets/test/'
src_path = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/'
#save_path = './onsets/test'
#src_path = './onsets/mp3/2.27节奏'

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

def check_three_hit(onsets_frames,base_frames):
    result = 0
    all = np.hstack((onsets_frames,base_frames))
    all.sort()
    for i in range(1,len(all)-1):
        if (all[i] in onsets_frames and all[i-1] in onsets_frames and all[i+1] in onsets_frames \
            and all[i] not in base_frames and all[i-1] not in base_frames and all[i+1] not in base_frames)\
            or (all[i]  in base_frames and all[i-1] in base_frames and all[i+1] in base_frames\
            and all[i] not in onsets_frames and all[i-1] not in onsets_frames and all[i+1] not in onsets_frames):
            result = 1
            break
    return str(result)

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
# 测试单个文件
#files = ['F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏六（5）（80）.wav']
for filename in files:
    print(filename)
    if filename.find('wav') <= 0:
        continue
    elif filename.find('shift') > 0:
        continue
    else:
        index = index + 1

    type_index = get_onsets_index_by_filename(filename)
    y, sr = load_and_trim(filename)
    total_frames_number = get_total_frames_number(filename)
    if len(y) > 0:
        #onsets_frames = get_real_onsets_frames(y)
        onsets_frames,onsets_frames_strength = get_onsets_by_all(y,sr)
        #onsets_frames = get_onsets_by_all_v2(y, sr,len(codes[type_index])+2)
        if len(onsets_frames) < 3:
            continue


        print("onsets_frames is {}".format(onsets_frames))



        # 标准节拍时间点
        base_frames = onsets_base_frames(codes[type_index],total_frames_number)
        print("base_frames is {}".format(base_frames))

        min_d, best_y,onsets_frames = get_dtw_min(onsets_frames, base_frames, 65)
        base_onsets = librosa.frames_to_time(best_y, sr=sr)
        print("base_onsets is {}".format(base_onsets))

        # 节拍时间点
        onstm = librosa.frames_to_time(onsets_frames, sr=sr)
        print("onstm is {}".format(onstm))
        duration = librosa.get_duration(y, sr=sr)  # 获取音频时长
        #print("duration is {}".format(duration))

        #节拍数之差
        diff_real_base = len(onsets_frames) - len(base_frames)

        #判断是否有未吻合的三连节拍,1是有，0是无
        three_hit = check_three_hit(onsets_frames, best_y)

        #librosa.display.waveplot(y, sr=sr)
        # plt.show()
        plt.vlines(onstm, -1 * np.max(y), np.max(y), color='b', linestyle='solid')
    plt.vlines(base_onsets,  -1*np.max(y),np.max(y), color='r', linestyle='dashed')
    #plt.vlines(base_onsets[-1],  -1*np.max(y),np.max(y), color='white', linestyle='dashed')
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

    if filename.find('标准') > 0:
        saveFileName = '100-A'
        savepath = save_path + 'A/'
    # elif filename.find('分') > 0:
    #     score = filename.split("分")[0].split(".")[1]  # 提取分值
    # elif filename.find('(') > 0:
    #     score = filename.split("(")[2].split(")")[0]  # 提取分值
    else:
        #score = filename.split("（")[2].split("）")[0]  # 提取分值
        score = re.sub("\D", "", filename)  # 筛选数字
    if str(score).find("100") > 0:
        score = 100
    else:
        score = int(score) % 100

    if int(score) >=90:
        grade = 'A'
        savepath = save_path + 'A/'
        files_list_a.append([filename.split('.wav')[0] + '-' + grade,min_d,diff_real_base,three_hit])
    elif int(score) >= 75:
        grade = 'B'
        savepath = save_path + 'B/'
        files_list_b.append([filename.split('.wav')[0] + '-' + grade, min_d,diff_real_base,three_hit])
    elif int(score) >=60:
        grade = 'C'
        savepath = save_path + 'C/'
        files_list_c.append([filename.split('.wav')[0] + '-' + grade, min_d,diff_real_base,three_hit])
    elif int(score) >=1:
        grade = 'D'
        savepath = save_path + 'D/'
        files_list_d.append([filename.split('.wav')[0] + '-' + grade, min_d,diff_real_base,three_hit])
    else:
        grade = 'E'
        savepath = save_path + 'E/'


    saveFileName = filename.split('.wav')[0] + '-' + grade
    #saveFileName = str(score) + '-' + grade
    file_sum = os.listdir(savepath)
    #saveFileName = str(len(file_sum)+1) + '-' + filename.split(".wav")[0] + '-' + saveFileName
    #saveFileName = str(len(file_sum) + 1) + '-' + saveFileName
    saveFileName = saveFileName
    plt.savefig(savepath + saveFileName+ '_'+ str(min_d) + '.jpg',  bbox_inches='tight', pad_inches=0)
    plt.clf()
    saveFileName = ''

t1 = np.vstack((files_list_a,files_list_b))
t2 = np.vstack((files_list_c,files_list_d))

files_list = np.vstack((t1,t2))
write_txt(files_list, new_old_txt, mode='w')

# 先获取多唱漏唱的情况




