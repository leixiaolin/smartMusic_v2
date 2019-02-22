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