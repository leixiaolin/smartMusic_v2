import  numpy as np
import librosa
import matplotlib.pyplot as plt
import re

def get_basetime(s):
    if s is None or len(s) < 1:
        print("input is empty")

    s = s.replace('[','').replace(']','')
    tmp = s.split(';')
    print(tmp)
    result = []
    for c in tmp:
        if c.find(","):  # 包括","的情况，即有多个数值
            cc = c.split(",")
            for ccc in cc:
                if ccc.find("(") > 0: # 空音的情况
                    score = re.sub("\D", "", ccc)  # 筛选数字
                    score = "-" + score
                    result.append(score)
                else:
                    result.append(ccc)
        else: # 不包括","的情况，即只有一个数值
            if c.find("(") > 0:  # 空音的情况
                score = re.sub("\D", "", c)  # 筛选数字
                score = -1 * score
                result.append(score)
            else:
                result.append(c)
    return result

def onsets_base(code,time,start_point):
    result = get_basetime(code)
    print(result)
    total = 0
    for r in result:
        if int(r) > 0:  # 不是空音
            total += int(r)
        else:
            total -= int(r)

    off = 0  # 累积时长，用于计算后面每个节拍点的位置
    ds = []
    for i, r in enumerate(result):
        if int(r) > 0:  # 不是空音
            ds.append(start_point + time * off / total)
            off += int(r)
        else:
            off -= int(r)

    ds.append(time)
    return ds
def get_min_max_total(s):
    if s is None or len(s) < 1:
        print("input is empty")

    s = s.replace('[','').replace(']','')
    tmp = s.split(';')
    print(tmp)
    result = []
    for c in tmp:
        if c.find(","):  # 包括","的情况，即有多个数值
            cc = c.split(",")
            for ccc in cc:
                if ccc.find("(") > 0: # 空音的情况
                    score = re.sub("\D", "", ccc)  # 筛选数字
                    score = "-" + score
                    result.append(score)
                else:
                    result.append(ccc)
        else: # 不包括","的情况，即只有一个数值
            if c.find("(") > 0:  # 空音的情况
                score = re.sub("\D", "", c)  # 筛选数字
                score = -1 * score
                result.append(score)
            else:
                result.append(c)
    result = [int(x) for x in result]
    min = np.min(result)
    max = np.max(result)
    total = np.sum(result)
    last = result[-1]
    return min,max,last,total

def get_real_onsets_frames(y):
    y_max = max(y)
    # y = np.array([x if x > y_max*0.01 else y_max*0.01 for x in y])
    # 获取每个帧的能量
    energy = librosa.feature.rmse(y)
    print(np.mean(energy))
    energy_diff = np.diff(energy)
    #print(energy_diff)
    onsets_frames = librosa.onset.onset_detect(y)

    print(onsets_frames)
    print(np.diff(onsets_frames))

    some_y = [energy[0][x] for x in onsets_frames]
    print("some_y is {}".format(some_y)) # 节拍点对应帧的能量
    energy_mean = (np.sum(some_y) - np.max(some_y))/(len(some_y)-1)  # 获取能量均值
    print("energy_mean for some_y is {}".format(energy_mean))
    energy_gap = energy_mean * 0.3
    some_energy_diff = [energy_diff[0][x] if x < len(energy_diff) else energy_diff[0][x-1]  for x in onsets_frames]
    energy_diff_mean = np.mean(some_energy_diff)
    print("some_energy_diff is {}".format(some_energy_diff))
    print("energy_diff_meanis {}".format(energy_diff_mean))
    onsets_frames = [x for x in onsets_frames if energy[0][x] > energy_gap]  # 筛选能量过低的伪节拍点

    r,c = energy_diff.shape
    if onsets_frames[-1] >= c:
        first = onsets_frames[0]
        last = onsets_frames[-1]
        onsets_frames = [x for x in onsets_frames[1:-1] if energy[0][x] > energy[0][x - 1] and energy[0][x + 1] > energy[0][x]]  # 只选择上升沿的节拍点
        onsets_frames.append(last)
        onsets_frames.insert(0,first)
    else:
        first = onsets_frames[0]
        onsets_frames = [x for x in onsets_frames[1:] if energy[0][x] > energy[0][x -1] and energy[0][x + 1] > energy[0][x] ]  # 只选择上升沿的节拍点
        onsets_frames.insert(0,first)


    # 筛选过密的节拍点
    onsets_frames_new = []
    for i in range(0, len(onsets_frames)):
        if i == 0:
            onsets_frames_new.append(onsets_frames[i])
            continue
        if onsets_frames[i] - onsets_frames[i - 1] <= 3:
            middle = int((onsets_frames[i] + onsets_frames[i - 1]) / 2)
            # middle = onsets_frames[i]
            onsets_frames_new.pop()
            onsets_frames_new.append(middle)
        else:
            onsets_frames_new.append(onsets_frames[i])
    onsets_frames = onsets_frames_new
    return onsets_frames

def get_real_onsets_frames_rhythm(y):
    y_max = max(y)
    # y = np.array([x if x > y_max*0.01 else y_max*0.01 for x in y])
    # 获取每个帧的能量
    energy = librosa.feature.rmse(y)
    print(np.mean(energy))
    energy_diff = np.diff(energy)
    #print(energy_diff)
    onsets_frames = librosa.onset.onset_detect(y)

    print(onsets_frames)
    print(np.diff(onsets_frames))

    some_y = [energy[0][x] for x in onsets_frames]
    print("some_y is {}".format(some_y)) # 节拍点对应帧的能量
    energy_mean = (np.sum(some_y) - np.max(some_y))/(len(some_y)-1)  # 获取能量均值
    print("energy_mean for some_y is {}".format(energy_mean))
    energy_gap = energy_mean * 0.3
    some_energy_diff = [energy_diff[0][x] if x < len(energy_diff) else energy_diff[0][x-1]  for x in onsets_frames]
    energy_diff_mean = np.mean(some_energy_diff)
    print("some_energy_diff is {}".format(some_energy_diff))
    print("energy_diff_meanis {}".format(energy_diff_mean))
    onsets_frames = [x for x in onsets_frames if energy[0][x] > energy_gap]  # 筛选能量过低的伪节拍点

    # 筛选过密的节拍点
    onsets_frames_new = []
    for i in range(0, len(onsets_frames)):
        if i == 0:
            onsets_frames_new.append(onsets_frames[i])
            continue
        if onsets_frames[i] - onsets_frames[i - 1] <= 7:
            middle = int((onsets_frames[i] + onsets_frames[i - 1]) / 2)
            # middle = onsets_frames[i]
            onsets_frames_new.pop()
            onsets_frames_new.append(middle)
        else:
            onsets_frames_new.append(onsets_frames[i])
    onsets_frames = onsets_frames_new
    return onsets_frames

if __name__ == '__main__':
    start_point = 0.2
    time = 6.45
    #code = '[0500,0500;1000;1500;0500;0250,0250,0250,0250;1000]'
    code = '[1000,1000;2000;1000,500,500;2000]'
    #code = '[2000;1000,1000;500,500,1000;2000]'
    #code = '[1000,1000;500,500,1000;1000,1000;2000]'
    #code = '[1000,--(1000);1000,--(1000);500,250,250,1000;--(1000),1000]'
    #code = '[500;1000,500,1000,500;500,500,500,250,250,500,500;250,250,500,500,1000]'
    #code = '[1000,--(1000);1000,--(1000);1000,-(500),500;1000,1000]'
    #code = '[750,250,500,500,500,-(500);500,1000,500,500,-(500);750,250,500,500,500,-(500)]'
    ds = onsets_base(code,time,start_point)


    plt.vlines(ds[:-1], 0, 2400, color='black', linestyle='dashed')
    plt.vlines(ds[-1], 0, 2400, color='white', linestyle='dashed')
    #plt.vlines(time, 0, 2400, color='white', linestyle='dashed')
    plt.show()