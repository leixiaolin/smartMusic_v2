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
#files = ['F:/项目/花城音乐项目/样式数据/3.06MP3/节奏/节4录音4(80).wav']
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
    #y, sr = librosa.load(filename)
    #y, sr = get_foreground(y, sr) # 分离前景音
    silence_threshold = 0.2
    need_vocal_separation = check_need_vocal_separation(y, silence_threshold)
    # if need_vocal_separation:
    #     y, sr = get_foreground(y, sr)  # 分离前景音
    total_frames_number = get_total_frames_number(filename)
    if len(y) > 0:
        #onsets_frames = get_real_onsets_frames(y)
        #_,onsets_frames = get_onset_rmse_viterbi(y,0.35)
        CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref=np.max)
        w, h = CQT.shape
        CQT[50:w, :] = -100
        CQT[0:20, :] = -100

        # 标准节拍时间点
        type_index = get_onsets_index_by_filename(filename)
        total_frames_number = get_total_frames_number(filename)
        # base_frames = onsets_base_frames_rhythm(type_index,total_frames_number)
        base_frames = onsets_base_frames(codes[type_index], total_frames_number)
        base_onsets = librosa.frames_to_time(base_frames, sr=sr)

        first_frame = base_frames[1] - base_frames[0]
        rms = librosa.feature.rmse(y=y)[0]
        rms = [x / np.std(rms) for x in rms]
        min_waterline = find_min_waterline(rms, 8)
        first_frame_rms = rms[0:first_frame]
        first_frame_rms_max = np.max(first_frame_rms)

        if first_frame_rms_max == np.max(rms):
            print("=====================================")
            threshold = first_frame_rms_max * 0.35
            rms = rms_smooth(rms, threshold, 6)
            #rms = [x if x > first_frame_rms_max * 0.35 else 0 for x in rms]
        else:
            threshold = first_frame_rms_max * 0.6
            rms = rms_smooth(rms, threshold, 6)
            #rms = [x if x > first_frame_rms_max * 0.6 else 0 for x in rms]
        # rms = [x / np.std(rms) if x / np.std(rms) > first_frame_rms_max*0.8 else 0 for x in rms]
        # rms = rms/ np.std(rms)
        rms_diff = np.diff(rms)
        # print("rms_diff is {}".format(rms_diff))
        print("rms max is {}".format(np.max(rms)))
        # all_peak_points = get_all_onsets_starts(rms,0.7)
        # all_peak_points = get_onsets_by_cqt_rms(y,16000,base_frames,0.7)
        topN = len(base_frames)
        waterline = 0
        if len(min_waterline) > 0:
            waterline = min_waterline[0][1]
            waterline *= 1.5
            waterline = find_best_waterline(rms, 4, topN) + 0.3
            if waterline < 0.6:
                waterline = 0.6
            # waterline = 0.8
            print("waterline is {}".format(waterline))
        all_peak_points, rms,threshold = get_topN_peak_by_denoise(rms, first_frame_rms_max * 0.8, topN, waterline)
        #all_peak_points,_ = get_topN_peak_by_denoise(rms, first_frame_rms_max * 0.8, topN)
        #onsets_frames = get_real_onsets_frames_rhythm(y)
        #_, onsets_frames = get_onset_rmse_viterbi(y, 0.35)
        #onsets_frames = get_all_onsets_starts_for_beat(rms, 0.6)
        onsets_frames = []

        # all_peak_points = get_all_onsets_starts_for_beat(rms,0.6)
        # all_trough_points = get_all_onsets_ends(rms,-0.4)
        want_all_points = np.hstack((all_peak_points, onsets_frames))
        want_all_points = list(set(want_all_points))
        want_all_points.sort()
        want_all_points_diff = np.diff(want_all_points)
        if len(want_all_points)>0:
            # 去掉挤在一起的线
            result = [want_all_points[0]]
            for i, v in enumerate(want_all_points_diff):
                if v > 4:
                    result.append(want_all_points[i + 1])
                else:
                    pass
            onsets_frames = result
        #onsets_frames = get_onsets_by_all_v2(y, sr,len(codes[type_index])+2)
        if len(onsets_frames) < 3:
            continue


        print("onsets_frames is {}".format(onsets_frames))



        # 标准节拍时间点
        base_frames = onsets_base_frames(codes[type_index],total_frames_number)
        print("base_frames is {}".format(base_frames))

        min_d, best_y,onsets_frames = get_dtw_min(onsets_frames, base_frames, 65,move=False)
        base_onsets = librosa.frames_to_time(best_y, sr=sr)
        print("base_onsets is {}".format(base_onsets))

        # 节拍时间点
        onstm = librosa.frames_to_time(onsets_frames, sr=sr)
        print("onstm is {}".format(onstm))
        duration = librosa.get_duration(y, sr=sr)  # 获取音频时长
        #print("duration is {}".format(duration))

        #节拍数之差
        diff_real_base = len(onsets_frames) - len(base_frames)


        librosa.display.waveplot(y, sr=sr)
        # plt.show()
        plt.vlines(onstm, -1 * np.max(y), np.max(y), color='b', linestyle='solid')
    #plt.vlines(base_onsets,  -1*np.max(y),np.max(y), color='r', linestyle='dashed')
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
        files_list_a.append([filename.split('.wav')[0] + '-' + grade,min_d,diff_real_base,need_vocal_separation])
    elif int(score) >= 75:
        grade = 'B'
        savepath = save_path + 'B/'
        files_list_b.append([filename.split('.wav')[0] + '-' + grade, min_d,diff_real_base,need_vocal_separation])
    elif int(score) >=60:
        grade = 'C'
        savepath = save_path + 'C/'
        files_list_c.append([filename.split('.wav')[0] + '-' + grade, min_d,diff_real_base,need_vocal_separation])
    elif int(score) >=1:
        grade = 'D'
        savepath = save_path + 'D/'
        files_list_d.append([filename.split('.wav')[0] + '-' + grade, min_d,diff_real_base,need_vocal_separation])
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




