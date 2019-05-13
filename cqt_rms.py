import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
from create_base import *
from myDtw import *
from find_mismatch import *
from filters import *
from vocal_separation import *

filepath = 'F:\项目\花城音乐项目\样式数据\音乐样本2019-01-29\节奏九\\'
# filename = 'F:/项目/花城音乐项目/样式数据/ALL/节奏/节奏八/节奏八（标准音频）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/ALL/节奏/节奏八/节奏8.100分.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/ALL/旋律/1.31MP3/旋律1.100分.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/ALL/旋律/1.31MP3/旋律2.100分.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/ALL/节奏/节奏八/节奏八（1）(90).wav'
#filename = 'F:/项目/花城音乐项目/样式数据/ALL/节奏/节奏八/节奏八（2）（90分）.wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏4卢(65).wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏2-01（80）.wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏4-02（68）.wav'
filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/节奏/节奏二（4）（100）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏五（6）（100）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/视唱1-02（90）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律2（四）(96).wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律1.1(95).wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律2.1(80).wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律2.3(55).wav'
#filename = 'F:/项目/花城音乐项目/样式数据/3.19MP3/节奏/节奏六1(10).wav'

#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律二（10）（75）.wav'
# filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律二（8）（100）.wav'
# filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律7_40218（20）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律一（9）（100）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律一（14）（95）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏2.2(95).wav'
#filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/节奏/节1.3(95).wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏一（14）（100）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏1.1(100).wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏1.2(100).wav'
filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋1.1(96).wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏1.3(100).wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏10_40320（60）.wav'

# 2. Load the audio as a waveform `y`
#    Store the sampling rate as `sr`

codes = np.array(['[1000,1000;2000;1000,500,500;2000]',
                  '[2000;1000,1000;500,500,1000;2000]',
                  '[1000,1000;500,500,1000;1000,1000;2000]',
                  '[1000,--(1000);1000,--(1000);500,250,250,1000;--(1000),1000]',
                  '[500;1000,500,1000,500;500,500,500,250,250,500,500;250,250,500,500,1000]',
                  '[1000,--(1000);1000,--(1000);1000,-(500),500;1000,1000]',
                  '[750,250,500,500,500,-(500);500,1000,500,500,-(500);750,250,500,500,500,-(500)]',
                  '[500,1000,500,500,250,250;1000,500,750,250,500;3000]',
                  '[500,500,500;1000,500;500,500,500;1500;500,500,500;1000,500;500;1000;1500]',
                  '[500,500,1000;500,500;1000;375,125,250,250,375,125,250,250;500,500,1000]',
                  '[500,500,1000;500,500,1000;500,500,750,250;2000]',
                  '[1000,1000;500,500,1000;1000,500,500;2000]',
                  '[1000,1000;500,500,1000;500,250,250,250;2000]',
                  '[500,1000,500;250,250,250,250,500,500;500,500,500,500;2000]'])
# 1. Get the file path to the included audio example
# Sonify detected beat events
# 定义加载语音文件并去掉两端静音的函数
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

def get_miss_onsets_rms(y,onset_frames_cqt,threshold):
    rms = librosa.feature.rmse(y=y)[0]
    rms = [x / np.std(rms) for x in rms]
    raw_rms = rms.copy()
    rms = np.diff(rms)
    rms_on_onset_frames_cqt = [rms[x] for x in onset_frames_cqt if x < len(rms)]
    min_rms_on_onset_frames_cqt = np.min(rms_on_onset_frames_cqt)
    rms = [1 if x >= min_rms_on_onset_frames_cqt else 0 for x in rms]
    for i in range(1,len(rms)):
        tmp = [np.abs(i - x) for x in onset_frames_cqt]
        min_gap = np.min(tmp)
        sub_onset_frames_cqt = [x for x in onset_frames_cqt if x<i]
        if len(sub_onset_frames_cqt) > 0:
            last = sub_onset_frames_cqt[-1]
            start = last
            end = i
            threshold_rms = threshold * np.max(raw_rms)
            if raw_rms[start]<threshold_rms and raw_rms[start+1]>threshold_rms:
                start += 1
            if start <= end:
                continue
            sub_rms = raw_rms[start:end]
            min_rms = np.min(sub_rms)
            if rms[i] == 1 and rms[i-1] == 0 and min_gap > 5 and raw_rms[i] > threshold_rms and min_rms< threshold_rms:
                #print("start,end,min_rms,threshold_rms  is {},{},{},{}".format(start,end,min_rms,threshold_rms))
                is_onset = check_onset_by_cqt(y, onset_frames_cqt, i)
                if is_onset:
                    onset_frames_cqt.append(i+1)
        else:
            if rms[i] == 1 and rms[i-1] == 0 and min_gap > 5 and raw_rms[i] > np.max(raw_rms) * threshold:
                is_onset = check_onset_by_cqt(y, onset_frames_cqt, i)
                if is_onset:
                    onset_frames_cqt.append(i + 1)
        onset_frames_cqt.sort()
    return onset_frames_cqt

def get_miss_onsets_by_cqt(y,onset_frames_cqt):
    if len(onset_frames_cqt) < 0:
        return onset_frames_cqt
    cqt = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref=np.max)
    rms = librosa.feature.rmse(y=y)[0]
    rms = [x / np.std(rms) for x in rms]
    cqt[0:30, :] = -100
    w,h = cqt.shape
    max_cqt = [np.max(cqt[:, x+2]) for x in onset_frames_cqt if x < h-2]
    mean_max_cqt = np.mean(max_cqt)
    global_before_cqt = [np.max(cqt[:, x-6:x - 3]) for x in onset_frames_cqt if x > 6]
    mean_global_before_cqt = np.mean(global_before_cqt)
    #print("mean_max_cqt is {}".format(mean_max_cqt))
    step = 4
    w,h = cqt.shape
    result = []
    for i in range(step,h-4,3):
        before_cqt = [np.max(cqt[:, x]) for x in range(i-step,i)]
        #before_cqt = [cqt[:, x] for x in range(i - step, i)]
        mean_before_cqt = np.mean(before_cqt)
        after_cqt = [np.max(cqt[:, x]) for x in range(i, i + step)]
        #after_cqt = [cqt[:, x] for x in range(i, i + step)]
        mean_after_cqt = np.mean(after_cqt)
        #if np.abs(mean_after_cqt - mean_max_cqt) < 5:
        if np.abs(mean_after_cqt - mean_max_cqt) < 10 and mean_before_cqt < mean_max_cqt and mean_before_cqt < mean_global_before_cqt + 7 and np.abs(mean_before_cqt - mean_global_before_cqt) < 10 and rms[i] < rms[i+1] and rms[i] > np.max(rms)*0.2:
            #print("mean_before_cqt,mean_global_before_cqt,mean_after_cqt,mean_max_cqt,i is {},{},{},{},{}".format(mean_before_cqt,mean_global_before_cqt,mean_after_cqt,mean_max_cqt,i))
            # cqt上半部存在亮的水平线
            #if i + 2 < h and ( np.abs(np.max(cqt[30:, i + 2]) - mean_max_cqt) < 10 or np.max(cqt[30:, i + 2]) > mean_max_cqt):
            result.append(i)
    if result:
        min_width = 5
        # print("min_width is {}".format(min_width))
        result = del_overcrowding(result, min_width)
    for i in result:
        tmp = [np.abs(i - x) for x in onset_frames_cqt]
        min_gap = np.min(tmp)
        if min_gap > 5:
            onset_frames_cqt.append(i)
    onset_frames_cqt.sort()
    return onset_frames_cqt

def check_onset_by_cqt_v2(y,onset_frames_cqt):
    cqt = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref=np.max)
    rms = librosa.feature.rmse(y=y)[0]
    rms = [x / np.std(rms) for x in rms]
    cqt[0:30, :] = -100
    w, h = cqt.shape
    max_cqt = [np.max(cqt[:, x + 2]) for x in onset_frames_cqt if x < h - 2]
    mean_max_cqt = np.mean(max_cqt)
    result = []
    for x in onset_frames_cqt:
        # cqt上半部存在亮的水平线
        print("real,mean_max_cqt,end is {},{},{}".format(np.max(cqt[30:, x + 2]), mean_max_cqt, x))
        #if x + 2 < h and (np.abs(np.max(cqt[30:, x + 2]) - mean_max_cqt) < 5 or np.max(cqt[30:, x + 2]) > mean_max_cqt):
        result.append(x)
    return result

def check_onset_by_cqt(y,onset_frames_cqt,onset_frame):
    cqt = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref=np.max)
    rms = librosa.feature.rmse(y=y)[0]
    rms = [x / np.std(rms) for x in rms]
    cqt[0:30, :] = -100
    max_cqt = [np.max(cqt[:, x+2]) for x in onset_frames_cqt]
    mean_max_cqt = np.mean(max_cqt)
    #print("mean_max_cqt is {}".format(mean_max_cqt))
    global_before_cqt = [np.max(cqt[:, x-5:x - 2]) for x in onset_frames_cqt]
    mean_global_before_cqt = np.mean(global_before_cqt)
    #print("mean_global_before_cqt is {}".format(mean_global_before_cqt))
    step = 4
    w,h = cqt.shape
    result = False
    i = onset_frame
    before_cqt = [np.max(cqt[:, x]) for x in range(i-step,i)]
    #before_cqt = [cqt[:, x] for x in range(i - step, i)]
    mean_before_cqt = np.mean(before_cqt)
    after_cqt = [np.max(cqt[:, x]) for x in range(i, i + step)]
    #after_cqt = [cqt[:, x] for x in range(i, i + step)]
    mean_after_cqt = np.mean(after_cqt)
    #if np.abs(mean_after_cqt - mean_max_cqt) < 5:
    if np.abs(mean_after_cqt - mean_max_cqt) < 10 and mean_before_cqt < mean_max_cqt and np.abs(mean_before_cqt - mean_global_before_cqt) < 5 and rms[i] < rms[i+1] and rms[i] > np.max(rms)*0.2:
        #print("mean_before_cqt,mean_global_before_cqt,mean_after_cqt,mean_max_cqt,i is {},{},{},{}".format(mean_before_cqt,mean_global_before_cqt,mean_after_cqt,mean_max_cqt,i))
        result = True
    return result

def find_false_onsets_rms(y,onset_frames_cqt,threshold):
    cqt = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref=np.max)
    cqt[0:30, :] = -100
    w,h =cqt.shape
    max_cqt = [np.max(cqt[:, x + 2]) for x in onset_frames_cqt if x < h-2]
    mean_max_cqt = np.mean(max_cqt)
    global_max = np.max(cqt)
    rms = librosa.feature.rmse(y=y)[0]
    rms = [x / np.std(rms) for x in rms]
    min_waterline = find_min_waterline(rms, 8)
    gap = 0
    if len(min_waterline) > 0:
        waterline = min_waterline[0][1]
        gap = (np.max(rms) - waterline) * 0.2

    # 关于第一个节拍
    # 条件一：节拍点的前后有高差，前小后大
    #print("checking the first")
    condation1 = rms[onset_frames_cqt[0] + 1] - rms[onset_frames_cqt[0] - 1] > 0.12 * np.max(rms)
    #print("condation1 is {},{},{}".format(condation1,rms[onset_frames_cqt[0] + 1],rms[onset_frames_cqt[0] - 1]))
    # 条件三：后面的cqt上半部存在亮的水平线
    #condation3 = np.abs(np.max(cqt[30:, onset_frames_cqt[0] + 2:onset_frames_cqt[0] - 6]) - global_max) < 10
    condation3 = True
    #print("condation3 is {}".format(condation3))
    # 条件四：前面的cqt上半部与后面的cqt上半部有差别
    condation4 = np.abs(np.mean(cqt[30:, onset_frames_cqt[0] - 6:onset_frames_cqt[0] - 2]) - np.mean(cqt[30:, onset_frames_cqt[0] + 2:onset_frames_cqt[0] + 6])) > 4
    #condation4 = True
    tmp1 = onset_frames_cqt[0] - 10 if onset_frames_cqt[0] - 10 >= 0 else 0
    tmp2 = onset_frames_cqt[0] - 2 if onset_frames_cqt[0] - 2 >= 0 else 0
    tmp3 = onset_frames_cqt[0] + 2 if onset_frames_cqt[0] + 2 <= len(rms) else len(rms)
    tmp4 = onset_frames_cqt[0] + 8 if onset_frames_cqt[0] + 8 <= len(rms) else len(rms)
    if tmp1 >= tmp2:
        condation4 = False
    else:
        condation4 = np.abs(np.mean(cqt[30:, tmp1:tmp2]) - np.mean(cqt[30:, tmp3:tmp4])) > 3
    #print("condation4 is {}".format(condation4))
    if condation1 and condation3 and condation4:
        result = [onset_frames_cqt[0]]
    else:
        result = []
    for i in range(1,len(onset_frames_cqt)):
        start = onset_frames_cqt[i-1]
        end = onset_frames_cqt[i]
        sub_rms = rms[start:end]
        #print("checking {}".format(end))
        # 条件一：节拍点的前后有高差，前小后大
        #print("len(rms),end is {},{}".format(len(rms),end))
        tmp1 = len(rms)-1 if end >=len(rms)-1 else end - 1
        tmp2 = len(rms)-1 if end >=len(rms)-2 else end + 1
        condation1 = rms[tmp2] - rms[tmp1] > 0.12 * np.max(rms)
        #print("condation1 is {}".format(condation1))
        # 条件二：跟前一节拍之间有波谷
        #condation2 = np.min(sub_rms) < np.max(rms) * 0.4
        condation2 = True
        #print("condation2 is {}".format(condation2))
        #条件三：后面的cqt上半部存在亮的水平线
        tmp1 = len(rms)-1 if end + 2 >= len(rms) else end + 2
        tmp2 = len(rms)-1 if end + 6 >= len(rms) else end + 6
        if tmp1 >= tmp2:
            condation3 = False
        else:
            condation3 = np.abs(np.max(cqt[30:, tmp1:tmp2]) - global_max) < 10
        #     print("condation3,np.max(cqt[30:, end + 2:end + 6]) - global_max is {},{},{}".format(np.max(cqt[30:, tmp1:tmp2]),global_max,condation3))
        # 条件四：前面的cqt上半部与后面的cqt上半部有差别
        tmp1 = len(rms)-1 if end + 2 >= len(rms) else end + 2
        tmp2 = len(rms)-1 if end + 6 >= len(rms) else end + 6
        if tmp1 >= tmp2:
            condation4 = False
        else:
            condation4 = np.abs(np.mean(cqt[30:, end - 6:end - 2]) - np.mean(cqt[30:, tmp1:tmp2])) > 3
        #print("condation4 is {}".format(condation4))
        #if end < len(rms) -6 and np.min(sub_rms) < waterline + gap and (rms[end + 1]>threshold * np.max(rms) or rms[end + 2]>threshold * np.max(rms)) and rms[end + 1] > np.max(rms) * 0.15  :
        if end < len(rms) - 6 and condation1 and condation2 and condation4:
            # cqt上半部存在亮的水平线
            #print("real,mean_max_cqt,end is {},{},{}".format(np.max(cqt[30:, end + 2]),mean_max_cqt,end))
            #if np.abs(np.max(cqt[30:, end + 2]) - mean_max_cqt) < 4:
            # is_onset = check_onset_by_cqt(y, onset_frames_cqt, end)
            # if is_onset:
            result.append(end)
    return result

def find_false_onsets_rms_secondary_optimised(y,onset_frames_cqt,threshold1,threshold2):

    rms = librosa.feature.rmse(y=y)[0]
    rms = [x / np.std(rms) for x in rms]
    result = []
    for x in onset_frames_cqt:
        # 较大上升沿 或 波谷点
        if rms[x+1] - rms[x] > threshold1 or (rms[x-1] - rms[x] > threshold2 and rms[x+1] - rms[x] > threshold2):
            result.append(x)
    return result

def get_best_threshod(y):
    onsets_frames = get_real_onsets_frames_rhythm(y, modify_by_energy=True, gap=0.1)
    rms = librosa.feature.rmse(y=y)[0]
    rms = [x / np.std(rms) for x in rms]
    rms_on_onsets = [rms[x] for x in onsets_frames]
    mean_rms = np.mean(rms_on_onsets)
    best_threshod = mean_rms/np.max(rms)*0.4
    #best_threshod = np.min(rms_on_onsets)
    return best_threshod

def get_missing_by_best_threshod(y,onsets_frames,best_threshod):
    rms = librosa.feature.rmse(y=y)[0]
    rms = [x / np.std(rms) for x in rms]
    base_line = np.max(rms) * best_threshod
    onsets_on_best_threshod = [i for i in range(1,len(rms)-1) if rms[i] <= base_line and rms[i+1] >= base_line]

    for x in onsets_on_best_threshod:
        offset = [np.abs(x - i) for i in onsets_frames]
        min_gap = np.min(offset)
        if min_gap > 5:
            onsets_frames.append(x + onsets_frames[0])

    onsets_frames.sort()
    return onsets_frames

def get_onsets_by_cqt_rms(filename,onset_code,gap=0.1):

    y, sr = librosa.load(filename)
    #type_index = get_onsets_index_by_filename(filename)
    total_frames_number = get_total_frames_number(filename)
    # base_frames = onsets_base_frames_rhythm(type_index,total_frames_number)
    base_frames = onsets_base_frames(onset_code, total_frames_number)
    # 标准节拍个数
    topN = len(base_frames)
    gap = 0.1
    run_total = 0
    onset_frames_cqt = []
    #threshold = 0.35
    best_onset_frames_cqt = []
    best_total = 0
    threshold = get_best_threshod(y)
    #print("best threshold is {}".format(threshold))
    onset_frames_cqt = get_real_onsets_frames_rhythm(y, modify_by_energy=True, gap=gap)
    #waterline, best_starts_waterline = find_best_waterline(rms, 4, topN)
    while True:
        # 从CQT特征上获取节拍
        #onset_frames_cqt = get_real_onsets_frames_rhythm(y, modify_by_energy=True,gap=gap)
        if onset_frames_cqt:
            min_width = 5
            #print("min_width is {}".format(min_width))
            onset_frames_cqt = del_overcrowding(onset_frames_cqt, min_width)
            #print("0. onset_frames_cqt is {}".format(onset_frames_cqt))

            # CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref=np.max)
            # onset_frames_cqt = get_miss_onsets_by_cqt(CQT, onset_frames_cqt)

            # if len(onset_frames_cqt)<1:
            #     onset_frames_cqt = get_real_onsets_frames_rhythm(y, modify_by_energy=True)
            if len(onset_frames_cqt) > 0:
                min_width = 5
                #print("min_width is {}".format(min_width))
                onset_frames_cqt = del_overcrowding(onset_frames_cqt, min_width)
                #print("1. onset_frames_cqt is {}".format(onset_frames_cqt))

            # 根据rms阀值线找漏的
            if len(onset_frames_cqt) > 0:
                onset_frames_cqt = get_missing_by_best_threshod(y, onset_frames_cqt, threshold)

            # 去伪
            if len(onset_frames_cqt) > 0:
                onset_frames_cqt = find_false_onsets_rms(y, onset_frames_cqt, threshold)
                #print("2. onset_frames_cqt is {}".format(onset_frames_cqt))


            # 找漏的
            if np.abs(len(onset_frames_cqt) - topN) <=3 and len(onset_frames_cqt) > 0:
                onset_frames_cqt = get_miss_onsets_rms(y, onset_frames_cqt,threshold)
                onset_frames_cqt = get_miss_onsets_by_cqt(y, onset_frames_cqt)

                #print("3. onset_frames_cqt is {}".format(onset_frames_cqt))
            if len(onset_frames_cqt) >= best_total:
                best_total = len(onset_frames_cqt)
                best_onset_frames_cqt = onset_frames_cqt
                #print("4. onset_frames_cqt is {}".format(best_onset_frames_cqt))
        if len(onset_frames_cqt) - topN >= 0 and len(onset_frames_cqt) - topN <= 3 or run_total >0:
            #print("best_onset_frames_cqt,len, run_total is {},{},{}".format(best_onset_frames_cqt,len(onset_frames_cqt),run_total))
            break
        else:
            threshold *= 0.9
            run_total += 1

    return best_onset_frames_cqt,threshold

def get_onsets_by_cqt_rms_optimised(filename,onset_code):
    #type_index = get_onsets_index_by_filename(filename)
    total_frames_number = get_total_frames_number(filename)
    # base_frames = onsets_base_frames_rhythm(type_index,total_frames_number)
    base_frames = onsets_base_frames(onset_code, total_frames_number)
    # 标准节拍个数
    topN = len(base_frames)
    y, sr = librosa.load(filename)
    onsets_frames = get_real_onsets_frames_rhythm(y, modify_by_energy=True, gap=0.1)
    if onsets_frames:
        min_width = 5
        # print("min_width is {}".format(min_width))
        onsets_frames = del_overcrowding(onsets_frames, min_width)
        #print("0. onset_frames_cqt is {}".format(onsets_frames))
    # 如果已经匹配很好，就直接返回
    if len(onsets_frames)>0 and len(onsets_frames) == topN:
        base_frames = onsets_base_frames(onset_code, total_frames_number - onsets_frames[0])
        base_frames = [x + (onsets_frames[0] - base_frames[0]) for x in base_frames]
        min_d, best_y, modify_onsets_frames = get_dtw_min(onsets_frames.copy(), base_frames, 65)
        #print("min_d is {}".format(min_d))
        if min_d < 4:
            rms = librosa.feature.rmse(y=y)[0]
            rms = [x / np.std(rms) for x in rms]
            rms_on_onsets = [rms[x] for x in onsets_frames]
            mean_rms = np.mean(rms_on_onsets)
            threshold = mean_rms / np.max(rms)
            #threshold = np.min(rms_on_onsets)
            return onsets_frames,threshold

    best_onset_frames_cqt = []
    best_total = 0
    best_threshold = 0
    onset_frames_cqt,threshold = get_onsets_by_cqt_rms(filename,onset_code)
    if len(onset_frames_cqt) >= best_total:
        best_total = len(onset_frames_cqt)
        best_onset_frames_cqt = onset_frames_cqt
        best_threshold = threshold

    if len(onset_frames_cqt) < topN and onsets_frames != get_real_onsets_frames_rhythm(y, modify_by_energy=True, gap=0.12):
        onset_frames_cqt,threshold = get_onsets_by_cqt_rms(filename,onset_code, 0.12)
        if len(onset_frames_cqt) >= best_total:
            best_total = len(onset_frames_cqt)
            best_onset_frames_cqt = onset_frames_cqt
            best_threshold = threshold


    if len(onset_frames_cqt) < topN and onsets_frames != get_real_onsets_frames_rhythm(y, modify_by_energy=True, gap=0.09):
        onset_frames_cqt,threshold = get_onsets_by_cqt_rms(filename,onset_code, 0.09)
        if len(onset_frames_cqt) >= best_total:
            best_total = len(onset_frames_cqt)
            best_onset_frames_cqt = onset_frames_cqt
            best_threshold = threshold
    if len(best_onset_frames_cqt) <1:
        best_onset_frames_cqt = onsets_frames
    return best_onset_frames_cqt,best_threshold

def get_onsets_by_cqt_rms_optimised_v2(filename):
    type_index = get_onsets_index_by_filename(filename)
    total_frames_number = get_total_frames_number(filename)
    # base_frames = onsets_base_frames_rhythm(type_index,total_frames_number)
    base_frames = onsets_base_frames(codes[type_index], total_frames_number)
    # 标准节拍个数
    topN = len(base_frames)
    best_onset_frames_cqt = []
    best_total = 0
    best_threshold = 0
    onset_frames_cqt,threshold = get_onsets_by_cqt_rms(filename)
    if len(onset_frames_cqt) >= best_total:
        best_onset_frames_cqt = onset_frames_cqt
        best_threshold = threshold

    return best_onset_frames_cqt,best_threshold

def get_detail_cqt_rms(filename):
    y, sr = librosa.load(filename)
    CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref=np.max)
    onset_frames_cqt, best_threshold = get_onsets_by_cqt_rms_optimised(filename)
    #print("5. onset_frames_cqt,best_threshold is {},{}".format(onset_frames_cqt, best_threshold))
    # if len(onset_frames_cqt)<topN:
    onset_frames_cqt = get_miss_onsets_by_cqt(y, onset_frames_cqt)
    #print("6. onset_frames_cqt,best_threshold is {},{}".format(onset_frames_cqt, best_threshold))
    #onset_frames_cqt = check_onset_by_cqt_v2(y, onset_frames_cqt)
    #print("7. onset_frames_cqt,best_threshold is {},{}".format(onset_frames_cqt, best_threshold))
    onset_frames_cqt_time = librosa.frames_to_time(onset_frames_cqt, sr=sr)
    #print("onset_frames_cqt_time is {}".format(onset_frames_cqt_time))

    type_index = get_onsets_index_by_filename(filename)
    total_frames_number = get_total_frames_number(filename)
    best_y = []
    # 标准节拍时间点
    if len(onset_frames_cqt)> 0:
        base_frames = onsets_base_frames(codes[type_index], total_frames_number - onset_frames_cqt[0])
        base_frames = [x + (onset_frames_cqt[0] - base_frames[0]) for x in base_frames]
        min_d, best_y, onsets_frames = get_dtw_min(onset_frames_cqt, base_frames, 65)
    else:
        base_frames = onsets_base_frames(codes[type_index], total_frames_number)
    base_onsets = librosa.frames_to_time(base_frames, sr=sr)

    # librosa.display.specshow(CQT)
    plt.figure(figsize=(10, 6))
    plt.subplot(4, 1, 1)  # 要生成两行两列，这是第一个图plt.subplot('行','列','编号')
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Constant-Q power spectrogram (note)')
    librosa.display.specshow(CQT, y_axis='cqt_note', x_axis='time')
    plt.vlines(onset_frames_cqt_time, 0, sr, color='y', linestyle='solid')
    #plt.vlines(base_onsets, 0, sr, color='r', linestyle='solid')

    # print(plt.figure)

    plt.subplot(4, 1, 2)  # 要生成两行两列，这是第一个图plt.subplot('行','列','编号')
    librosa.display.waveplot(y, sr=sr)
    plt.vlines(onset_frames_cqt_time, -1 * np.max(y), np.max(y), color='y', linestyle='solid')

    plt.subplot(4, 1, 3)
    rms = librosa.feature.rmse(y=y)[0]
    rms = [x / np.std(rms) for x in rms]
    max_rms = np.max(rms)
    # rms = np.diff(rms)
    times = librosa.frames_to_time(np.arange(len(rms)))
    # rms_on_onset_frames_cqt = [rms[x] for x in onset_frames_cqt]
    # min_rms_on_onset_frames_cqt = np.min(rms_on_onset_frames_cqt)
    # rms = [1 if x >=min_rms_on_onset_frames_cqt else 0 for x in rms]
    plt.plot(times, rms)
    # plt.axhline(min_rms_on_onset_frames_cqt)
    plt.axhline(max_rms * best_threshold)
    # plt.vlines(onsets_frames_rms_best_time, 0,np.max(rms), color='y', linestyle='solid')
    plt.vlines(onset_frames_cqt_time, 0, np.max(rms), color='y', linestyle='solid')
    #plt.vlines(base_onsets, 0, np.max(rms), color='r', linestyle='solid')
    plt.xlim(0, np.max(times))

    plt.subplot(4, 1, 4)
    plt.vlines(base_onsets, 0, np.max(rms), color='r', linestyle='dashed')
    plt.xlim(0, np.max(times))
    plt.axhline(max_rms * best_threshold)
    return onset_frames_cqt,best_y,best_threshold,plt

def get_detail_cqt_rms_secondary_optimised(filename):

    onset_frames_cqt, best_y, best_threshold, _ = get_detail_cqt_rms(filename)

    y, sr = librosa.load(filename)

    loss_frames = []
    for i in range(len(onset_frames_cqt)-1):
        start = onset_frames_cqt[i]
        end = onset_frames_cqt[i+1]

        if end - start > 30:
            start_end_time = librosa.frames_to_time([start,end], sr=sr)
            #print("start_end_time is {}".format(start_end_time))
            y1,sr1 = librosa.load(filename,offset=start_end_time[0],duration=start_end_time[1] - start_end_time[0])
            # 根据rms阀值线找漏的
            if len(onset_frames_cqt) > 0:
                threshold = 0.6
                tmp = get_missing_by_best_threshod(y1, [start,end], threshold)
                if len(tmp)>=3:
                    for j in range(1,len(tmp)-1):
                        loss_frames.append(tmp[j])
                        #print("add is {}".format(tmp[1:-1]))
            # rms = librosa.feature.rmse(y=y1)[0]
            # rms_on_onset_frames_cqt = [rms[x] for x in [start,end]]
            # min_rms_on_onset_frames_cqt = np.min(rms_on_onset_frames_cqt)
            # rms = [1 if x >=min_rms_on_onset_frames_cqt else 0 for x in rms]
            #
            # loss = [i for i in range(len(rms)-6) if rms[i] == 0 and rms[i+1] == 1 and np.min(rms[i+1:i+6]) == 1 and i < end and i > start ]
            # for x in loss:
            #     loss_frames.append(x)

    if len(loss_frames)>0:
        for x in loss_frames:
            onset_frames_cqt.append(x)
        onset_frames_cqt.sort()

    CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref=np.max)
    #onset_frames_cqt, best_threshold = get_onsets_by_cqt_rms_optimised(filename)
    #print("5. onset_frames_cqt,best_threshold is {},{}".format(onset_frames_cqt, best_threshold))
    # if len(onset_frames_cqt)<topN:
    onset_frames_cqt = get_miss_onsets_by_cqt(y, onset_frames_cqt)
    onset_frames_cqt = find_false_onsets_rms_secondary_optimised(y, onset_frames_cqt, 0.1, 0.1)
    if onset_frames_cqt:
        min_width = 5
        # print("min_width is {}".format(min_width))
        onset_frames_cqt = del_overcrowding(onset_frames_cqt, min_width)
    #print("6. onset_frames_cqt,best_threshold is {},{}".format(onset_frames_cqt, best_threshold))
    #onset_frames_cqt = check_onset_by_cqt_v2(y, onset_frames_cqt)
    #print("7. onset_frames_cqt,best_threshold is {},{}".format(onset_frames_cqt, best_threshold))
    onset_frames_cqt_time = librosa.frames_to_time(onset_frames_cqt, sr=sr)



    type_index = get_onsets_index_by_filename(filename)
    total_frames_number = get_total_frames_number(filename)
    best_y = []
    # 标准节拍时间点
    if len(onset_frames_cqt)> 0:
        base_frames = onsets_base_frames_for_note(filename)
        base_frames = [x + onset_frames_cqt[0] - base_frames[0] for x in base_frames]
        min_d, best_y, onsets_frames = get_dtw_min(onset_frames_cqt, base_frames, 65)
    else:
        base_frames = onsets_base_frames_for_note(filename)
    base_onsets = librosa.frames_to_time(base_frames, sr=sr)
    plt.close() # 关闭第一次的图片句柄

    # librosa.display.specshow(CQT)
    plt.figure(figsize=(10, 6))
    plt.subplot(4, 1, 1)  # 要生成两行两列，这是第一个图plt.subplot('行','列','编号')
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Constant-Q power spectrogram (note)')
    # for x in onset_frames_cqt:
    #     sub_cqt = CQT.copy()[:,x]
    #     sub_cqt[0:20] = np.min(CQT)
    #     max_index = np.where(sub_cqt==np.max(sub_cqt))[0][0]
    #     print("max_index is {}".format(max_index))
    #     #plt.axhline(max_index,color="r")
    #     CQT[max_index,:] = np.min(CQT)

    librosa.display.specshow(CQT, y_axis='cqt_note', x_axis='time')
    plt.vlines(onset_frames_cqt_time, 0, sr, color='y', linestyle='solid')
    #plt.vlines(base_onsets, 0, sr, color='r', linestyle='solid')

    # print(plt.figure)

    plt.subplot(4, 1, 2)  # 要生成两行两列，这是第一个图plt.subplot('行','列','编号')
    librosa.display.waveplot(y, sr=sr)
    plt.vlines(onset_frames_cqt_time, -1 * np.max(y), np.max(y), color='y', linestyle='solid')

    plt.subplot(4, 1, 3)
    rms = librosa.feature.rmse(y=y)[0]
    rms = [x / np.std(rms) for x in rms]
    max_rms = np.max(rms)
    # rms = np.diff(rms)
    times = librosa.frames_to_time(np.arange(len(rms)))
    rms_on_onset_frames_cqt = [rms[x] for x in onset_frames_cqt]
    min_rms_on_onset_frames_cqt = np.min(rms_on_onset_frames_cqt)
    rms = [1 if x >=min_rms_on_onset_frames_cqt else 0 for x in rms]
    plt.plot(times, rms)
    # plt.axhline(min_rms_on_onset_frames_cqt)
    plt.axhline(max_rms * best_threshold)
    # plt.vlines(onsets_frames_rms_best_time, 0,np.max(rms), color='y', linestyle='solid')
    plt.vlines(onset_frames_cqt_time, 0, np.max(rms), color='y', linestyle='solid')
    #plt.vlines(base_onsets, 0, np.max(rms), color='r', linestyle='solid')
    plt.xlim(0, np.max(times))

    plt.subplot(4, 1, 4)
    plt.vlines(base_onsets, 0, np.max(rms), color='r', linestyle='dashed')
    plt.xlim(0, np.max(times))
    plt.axhline(max_rms * best_threshold)
    return onset_frames_cqt,best_y,best_threshold,plt

if __name__ == '__main__':
    #y, sr = load_and_trim('F:/项目/花城音乐项目/样式数据/ALL/旋律/1.31MP3/旋律1.100分.wav')
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/节奏/节7录音2(20).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/节奏/节8王（60）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/节奏/节6录音3(100).wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋律八（2）（60）.wav'
    #filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律1_40211（90）.wav'
    #filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律3_40302（95）.wav'
    #filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律6.4(90).wav'
    #filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律一（13）（98）.wav'
    #filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋8录音4(93).wav'



    savepath = './single_notes/data/test/'
    #onset_frames_cqt, best_y,best_threshold, plt = get_detail_cqt_rms(filename)
    onset_frames_cqt, best_y, best_threshold, plt = get_detail_cqt_rms_secondary_optimised(filename)

    print("onset_frames_cqt is {}".format(onset_frames_cqt))
    plt.show()

    dir_list = ['F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/']
    dir_list = ['e:/test_image/m1/A/']
    #dir_list = []
    total_accuracy = 0
    total_num = 0
    result_path = 'e:/test_image/n/'
    # clear_dir(result_path)
    # 要测试的数量
    test_num = 100
    score = 0
    for dir in dir_list:
        file_list = os.listdir(dir)
        # shuffle(file_list)  # 将语音文件随机排列
        # file_list = ['视唱1-01（95）.wav']
        for filename in file_list:
            # clear_dir(image_dir)
            # wavname = re.findall(pattern,filename)[0]
            print(dir + filename)
            # plt = draw_start_end_time(dir + filename)
            #plt = draw_baseline_and_note_on_cqt(dir + filename, False)
            onset_frames_cqt, best_y, best_threshold, plt = get_detail_cqt_rms_secondary_optimised(dir + filename)
            # tmp = os.listdir(result_path)

            if filename.find("tune") > 0 or filename.find("add") > 0 or filename.find("shift") > 0:
                score = re.sub("\D", "", filename.split("-")[0])  # 筛选数字
            else:
                score = re.sub("\D", "", filename)  # 筛选数字

            if str(score).find("100") > 0:
                score = 100
            else:
                score = int(score) % 100

            if int(score) >= 90:
                grade = 'A'
            elif int(score) >= 75:
                grade = 'B'
            elif int(score) >= 60:
                grade = 'C'
            elif int(score) >= 1:
                grade = 'D'
            else:
                grade = 'E'
            # result_path = result_path + grade + "/"
            # plt.savefig(result_path + filename + '.jpg', bbox_inches='tight', pad_inches=0)
            grade = 'A'
            plt.savefig(result_path + grade + "/" + filename + '.jpg', bbox_inches='tight', pad_inches=0)
            plt.clf()
