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


score = 0
save_path = 'F:/项目/花城音乐项目/参考代码/tensorflow_models_nets-master/onsets/test/'
src_path = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/'
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

def get_max_strength(chromagram):
    c_max = np.argmax(chromagram, axis=0)
    #print(c_max.shape[0])
    #print(c_max)
   # print(np.diff(c_max))
    # chromagram_diff = np.diff(chromagram,axis=0)
    # print(chromagram_diff)
    # sum_chromagram_diff = chromagram_diff.sum(axis=0)
    # test = np.array(sum_chromagram_diff)
    # plt.plot(test)

    img = np.zeros(chromagram.shape, dtype=np.float32)
    w, h = chromagram.shape
    for x in range(h):
        # img.item(x, c_max[x], 0)
        img.itemset((c_max[x], x), 1)
    return img

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
new_old_txt = '../onsets/best_dtw.txt'
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

    y, sr = load_and_trim(filename)
    CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref=np.max)
    w, h = CQT.shape
    CQT[40:w, :] = -100
    CQT[0:20, :] = -100

    rms = librosa.feature.rmse(y=y)[0]
    rms = rms / np.std(rms)
    rms_diff = np.diff(rms)
    #print("rms_diff is {}".format(rms_diff))
    # 标准节拍时间点
    type_index = get_onsets_index_by_filename_rhythm(filename)
    total_frames_number = get_total_frames_number(filename)
    base_frames = onsets_base_frames_rhythm(type_index, total_frames_number)
    base_onsets = librosa.frames_to_time(base_frames, sr=sr)
    print("rms max is {}".format(np.max(rms)))
    # all_peak_points = get_all_onsets_starts(rms,0.7)
    all_peak_points = get_onsets_by_cqt_rms(y, 16000, base_frames, 0.7)
    # all_peak_points = get_all_onsets_starts_for_beat(rms,0.6)
    # all_trough_points = get_all_onsets_ends(rms,-0.4)
    # want_all_points = np.hstack((all_peak_points, all_trough_points))
    # want_all_points = list(set(want_all_points))
    # want_all_points.sort()
    # want_all_points_diff = np.diff(want_all_points)
    # #去掉挤在一起的线
    # result = [want_all_points[0]]
    # for i,v in enumerate(want_all_points_diff):
    #     if v > 4:
    #         result.append(want_all_points[i+1])
    #     else:
    #        pass
    want_all_points = all_peak_points
    # want_all_points = [x for i,x in enumerate(all_points) if i < len(all_points)-1 and (peak_trough_rms_diff[i]>1)]
    print("want_all_points is {}".format(want_all_points))
    want_all_points_time = librosa.frames_to_time(want_all_points)

    # librosa.display.specshow(CQT)
    plt.figure(figsize=(10, 6))
    plt.subplot(5, 1, 1)  # 要生成两行两列，这是第一个图plt.subplot('行','列','编号')
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Constant-Q power spectrogram (note)')
    librosa.display.specshow(CQT, y_axis='cqt_note', x_axis='time')
    # onsets_frames =  librosa.onset.onset_detect(y)
    # onsets_frames = get_real_onsets_frames_rhythm(y)
    # onsets_frames = get_onsets_frames_by_cqt_for_rhythm(y,sr)

    print(np.max(y))
    # onstm = librosa.frames_to_time(onsets_frames, sr=sr)
    plt.vlines(want_all_points_time, 0, sr, color='y', linestyle='solid')
    print(CQT.shape)
    q1, q2 = CQT.shape
    print(plt.figure)

    plt.subplot(5, 1, 2)  # 要生成两行两列，这是第一个图plt.subplot('行','列','编号')
    librosa.display.waveplot(y, sr=sr)
    plt.vlines(want_all_points_time, -1 * np.max(y), np.max(y), color='y', linestyle='solid')

    # duration = librosa.get_duration(filename=filename)
    # # 标准节拍时间点
    # base_onsets = onsets_base(codes[11], duration, onstm[0])
    # plt.vlines(base_onsets[:-1], -1*np.max(y),np.max(y), color='r', linestyle='dashed')
    # plt.vlines(base_onsets[-1], -1*np.max(y),np.max(y), color='white', linestyle='dashed')
    plt.subplot(5, 1, 3)
    times = librosa.frames_to_time(np.arange(len(rms)))
    plt.plot(times, rms)
    # plt.axhline(0.02, color='r', alpha=0.5)
    plt.xlabel('Time')
    plt.ylabel('RMS')
    plt.axis('tight')
    plt.xlim(0, np.max(times))
    plt.vlines(want_all_points_time, 0, np.max(rms), color='y', linestyle='solid')
    # 标准节拍时间点
    base_frames = onsets_base_frames_rhythm(type_index, total_frames_number)
    print("base_frames is {}".format(base_frames))

    # min_d, best_y, onsets_frames = get_dtw_min(onsets_frames, base_frames, 65,move=False)
    if base_frames[0] < want_all_points[0]:
        best_y = [x + (want_all_points[0] - base_frames[0]) for x in base_frames]
    else:
        best_y = base_frames
    base_onsets = librosa.frames_to_time(best_y, sr=sr)
    plt.vlines(base_onsets, 0, np.max(rms), color='r', linestyle='dashed')
    # 找出漏唱的线的帧
    standard_y = want_all_points.copy()
    recognize_y = best_y.copy()
    miss_onsets = get_mismatch_line(standard_y, recognize_y)
    miss_onsets_time = librosa.frames_to_time(miss_onsets[1], sr=sr)
    plt.vlines(miss_onsets_time, 0, np.max(rms), color='black', linestyle='dashed')

    plt.subplot(5, 1, 4)
    chromagram = librosa.feature.chroma_cqt(y, sr=sr)

    c_max = np.argmax(chromagram, axis=0)
    # print("c_max is {}".format(c_max))
    c_max_diff = np.diff(c_max)  # 一阶差分
    img = np.zeros(chromagram.shape, dtype=np.float32)
    w, h = chromagram.shape
    for x in range(len(c_max_diff)):
        # img.item(x, c_max[x], 0)
        if x > 0 and (c_max_diff[x] == 1 or c_max_diff[x] == -1):
            c_max[x] = c_max[x - 1]

    for x in range(h):
        # img.item(x, c_max[x], 0)
        img.itemset((c_max[x], x), 1)
        img.itemset((c_max[x], x), 1)
        img.itemset((c_max[x], x), 1)
    # 最强音色图
    img = get_max_strength(CQT)
    librosa.display.specshow(img, x_axis='time', y_axis='chroma', cmap='coolwarm')
    # plt.vlines(base_onsets, 0, sr, color='y', linestyle='solid')
    plt.vlines(want_all_points_time, 0, sr, color='y', linestyle='solid')

    plt.subplot(5, 1, 5)
    c_max = np.argmax(CQT, axis=0)
    # c_max = deburring(c_max, 3)
    # note_start,note_end,note_number = find_note_number(c_max,all_peak_points[0],all_peak_points[1])
    # note_start_time = librosa.frames_to_time([note_start])
    # print("note_number is {}".format(note_number))
    c_max_diff = np.diff(c_max)
    plt.plot(times, c_max)
    plt.xlim(0, np.max(times))
    onsets_frames = get_onsets_by_cqt_rms(y, 16000, base_frames, 0.7)
    want_all_points_time = librosa.frames_to_time(onsets_frames)
    plt.vlines(want_all_points_time, np.min(c_max), np.max(c_max), color='y', linestyle='solid')
    # plt.text(note_start_time, note_number, note_number)
    for i in range(len(onsets_frames) - 1):
        note_start, note_end, note_number = find_note_number_by_range(c_max, onsets_frames[i], onsets_frames[i + 1])
        note_start_time = librosa.frames_to_time([note_start])
        plt.text(note_start_time, note_number - 4, note_number)
        if i == 0:
            first_note_number = note_number
    print(all_peak_points[-1])
    note_start, note_end, note_number = find_note_number_by_range(c_max, onsets_frames[-1], len(c_max) - 1)
    note_start_time = librosa.frames_to_time([note_start])
    find_note = find_note_number(note_number, 3)
    note_number_gap = first_note_number - find_note[0]
    plt.text(note_start_time, note_number - 4, note_number)
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
        files_list_a.append([filename.split('.wav')[0] + '-' + grade])
    elif int(score) >= 75:
        grade = 'B'
        savepath = save_path + 'B/'
        files_list_b.append([filename.split('.wav')[0] + '-' + grade])
    elif int(score) >=60:
        grade = 'C'
        savepath = save_path + 'C/'
        files_list_c.append([filename.split('.wav')[0] + '-' + grade])
    elif int(score) >=1:
        grade = 'D'
        savepath = save_path + 'D/'
        files_list_d.append([filename.split('.wav')[0] + '-' + grade])
    else:
        grade = 'E'
        savepath = save_path + 'E/'


    saveFileName = filename.split('.wav')[0] + '-' + grade
    #saveFileName = str(score) + '-' + grade
    file_sum = os.listdir(savepath)
    #saveFileName = str(len(file_sum)+1) + '-' + filename.split(".wav")[0] + '-' + saveFileName
    #saveFileName = str(len(file_sum) + 1) + '-' + saveFileName
    saveFileName = saveFileName
    plt.savefig(savepath + saveFileName+ '.jpg',  bbox_inches='tight', pad_inches=0)
    plt.clf()
    saveFileName = ''

t1 = np.vstack((files_list_a,files_list_b))
t2 = np.vstack((files_list_c,files_list_d))

files_list = np.vstack((t1,t2))
write_txt(files_list, new_old_txt, mode='w')

# 先获取多唱漏唱的情况




