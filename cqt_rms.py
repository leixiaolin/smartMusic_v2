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
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏7-02（30）.wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏五（6）（100）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/视唱1-02（90）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律2（四）(96).wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律1.1(95).wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律2.1(80).wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律2.3(55).wav'
#filename = 'F:/项目/花城音乐项目/样式数据/3.19MP3/节奏/节奏六1(10).wav'

#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律二（10）（75）.wav'
# filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律二（8）（100）.wav'
# filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律7_40218（20）.wav'
# filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律一（9）（100）.wav'
# filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/旋律/旋律一（14）（95）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/节奏/节1罗（90）.wav'
filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏2.2(95).wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏1-04（95）.wav'
#filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏1.1(100).wav'


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
                onset_frames_cqt.append(i+1)
        else:
            if rms[i] == 1 and rms[i-1] == 0 and min_gap > 5 and raw_rms[i] > np.max(raw_rms) * threshold:
                onset_frames_cqt.append(i+1)
        onset_frames_cqt.sort()
    return onset_frames_cqt

def find_false_onsets_rms(y,onset_frames_cqt,threshold):
    rms = librosa.feature.rmse(y=y)[0]
    rms = [x / np.std(rms) for x in rms]
    result = [onset_frames_cqt[0]]
    for i in range(1,len(onset_frames_cqt)):
        start = onset_frames_cqt[i-1]
        end = onset_frames_cqt[i]
        sub_rms = rms[start:end]
        if end < len(rms) -2 and np.min(sub_rms) < threshold * np.max(rms) and (rms[end + 1]>threshold * np.max(rms) or rms[end + 2]>threshold * np.max(rms))  :
            result.append(end)
    return result

def get_onsets_by_cqt_rms(filename,threshold=0.35):

    y, sr = librosa.load(filename)
    type_index = get_onsets_index_by_filename(filename)
    total_frames_number = get_total_frames_number(filename)
    # base_frames = onsets_base_frames_rhythm(type_index,total_frames_number)
    base_frames = onsets_base_frames(codes[type_index], total_frames_number)
    # 标准节拍个数
    topN = len(base_frames)
    gap = 0.5
    run_total = 0
    onset_frames_cqt = []
    #threshold = 0.35
    best_onset_frames_cqt = []
    best_total = 0
    while True:
        # 从CQT特征上获取节拍
        onset_frames_cqt = get_real_onsets_frames_rhythm(y, modify_by_energy=True,gap=gap)
        if onset_frames_cqt:
            min_width = 3
            #print("min_width is {}".format(min_width))
            onset_frames_cqt = del_overcrowding(onset_frames_cqt, min_width)
            #print("0. onset_frames_cqt is {}".format(onset_frames_cqt))

            # if len(onset_frames_cqt)<1:
            #     onset_frames_cqt = get_real_onsets_frames_rhythm(y, modify_by_energy=True)
            if len(onset_frames_cqt) > 0:
                min_width = 3
                #print("min_width is {}".format(min_width))
                onset_frames_cqt = del_overcrowding(onset_frames_cqt, min_width)
                #print("1. onset_frames_cqt is {}".format(onset_frames_cqt))

            # 去伪
            if len(onset_frames_cqt) > 0:
                onset_frames_cqt = find_false_onsets_rms(y, onset_frames_cqt, threshold)
                #print("2. onset_frames_cqt is {}".format(onset_frames_cqt))

            # 找漏的
            if np.abs(len(onset_frames_cqt) - topN) <=3 and len(onset_frames_cqt) > 0:
                onset_frames_cqt = get_miss_onsets_rms(y, onset_frames_cqt,threshold)
                #print("3. onset_frames_cqt is {}".format(onset_frames_cqt))
            if len(onset_frames_cqt) >= best_total:
                best_total = len(onset_frames_cqt)
                best_onset_frames_cqt = onset_frames_cqt
                #print("4. onset_frames_cqt is {}".format(best_onset_frames_cqt))
        if len(onset_frames_cqt) - topN > 0 and len(onset_frames_cqt) - topN <= 3 or run_total >20:
            #print("len, run_total is {},{}".format(len(onset_frames_cqt),run_total))
            break
        else:
            gap *= 0.9
            run_total += 1

    return best_onset_frames_cqt

def get_onsets_by_cqt_rms_optimised(filename):
    type_index = get_onsets_index_by_filename(filename)
    total_frames_number = get_total_frames_number(filename)
    # base_frames = onsets_base_frames_rhythm(type_index,total_frames_number)
    base_frames = onsets_base_frames(codes[type_index], total_frames_number)
    # 标准节拍个数
    topN = len(base_frames)
    best_onset_frames_cqt = []
    best_total = 0
    best_threshold = 0
    onset_frames_cqt = get_onsets_by_cqt_rms(filename)
    if len(onset_frames_cqt) >= best_total:
        best_total = len(onset_frames_cqt)
        best_onset_frames_cqt = onset_frames_cqt
        best_threshold = 0.35

    if len(onset_frames_cqt) < topN:
        onset_frames_cqt = get_onsets_by_cqt_rms(filename, 0.3)
        if len(onset_frames_cqt) >= best_total:
            best_total = len(onset_frames_cqt)
            best_onset_frames_cqt = onset_frames_cqt
            best_threshold = 0.3

    if len(onset_frames_cqt) < topN:
        onset_frames_cqt = get_onsets_by_cqt_rms(filename, 0.2)
        if len(onset_frames_cqt) >= best_total:
            best_total = len(onset_frames_cqt)
            best_onset_frames_cqt = onset_frames_cqt
            best_threshold = 0.2

    if len(onset_frames_cqt) < topN:
        onset_frames_cqt = get_onsets_by_cqt_rms(filename, 0.1)
        if len(onset_frames_cqt) >= best_total:
            best_total = len(onset_frames_cqt)
            best_onset_frames_cqt = onset_frames_cqt
            best_threshold = 0.1

    return best_onset_frames_cqt,best_threshold
if __name__ == '__main__':
    #y, sr = load_and_trim('F:/项目/花城音乐项目/样式数据/ALL/旋律/1.31MP3/旋律1.100分.wav')
    savepath = './single_notes/data/test/'
    y, sr = librosa.load(filename)
    CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=16000), ref = np.max)
    onset_frames_cqt,best_threshold = get_onsets_by_cqt_rms_optimised(filename)
    print("5. onset_frames_cqt,best_threshold is {},{}".format(onset_frames_cqt,best_threshold))
    onset_frames_cqt_time = librosa.frames_to_time(onset_frames_cqt, sr=sr)

    type_index = get_onsets_index_by_filename(filename)
    total_frames_number = get_total_frames_number(filename)
    # 标准节拍时间点
    base_frames = onsets_base_frames(codes[type_index], total_frames_number-onset_frames_cqt[0])
    base_frames_time = librosa.frames_to_time(base_frames, sr=sr)
    min_d, best_y, onsets_frames = get_dtw_min(onset_frames_cqt, base_frames, 65)
    base_onsets = librosa.frames_to_time(best_y, sr=sr)

    #librosa.display.specshow(CQT)
    plt.figure(figsize=(10, 6))
    plt.subplot(3,1,1) #要生成两行两列，这是第一个图plt.subplot('行','列','编号')
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Constant-Q power spectrogram (note)')
    librosa.display.specshow(CQT, y_axis='cqt_note',x_axis='time')
    plt.vlines(onset_frames_cqt_time, 0,sr, color='y', linestyle='solid')
    plt.vlines(base_onsets, 0,sr, color='r', linestyle='solid')


    #print(plt.figure)

    plt.subplot(3,1,2) #要生成两行两列，这是第一个图plt.subplot('行','列','编号')
    librosa.display.waveplot(y, sr=sr)
    plt.vlines(onset_frames_cqt_time, -1*np.max(y),np.max(y), color='y', linestyle='solid')

    plt.subplot(3,1,3)

    rms = librosa.feature.rmse(y=y)[0]
    rms = [x / np.std(rms) for x in rms]
    max_rms = np.max(rms)
    #rms = np.diff(rms)
    times = librosa.frames_to_time(np.arange(len(rms)))
    rms_on_onset_frames_cqt = [rms[x] for x in onset_frames_cqt]
    min_rms_on_onset_frames_cqt = np.min(rms_on_onset_frames_cqt)
    #rms = [1 if x >=min_rms_on_onset_frames_cqt else 0 for x in rms]
    plt.plot(times, rms)
    #plt.axhline(min_rms_on_onset_frames_cqt)
    plt.axhline(max_rms*best_threshold)
    # plt.vlines(onsets_frames_rms_best_time, 0,np.max(rms), color='y', linestyle='solid')
    plt.vlines(onset_frames_cqt_time, 0,np.max(rms), color='y', linestyle='solid')
    plt.vlines(base_onsets, 0, np.max(rms), color='r', linestyle='solid')
    plt.xlim(0,np.max(times))
    plt.show()
