import numpy as np
from create_base import *
from myDtw import *
from grade import *

# 找出多唱或漏唱的线的帧
def get_mismatch_line(standard_y,recognize_y):
    # standard_y标准线的帧列表 recognize_y识别线的帧列表
    ls = len(standard_y)
    lr = len(recognize_y)

    # 若标准线和识别线数量相同
    if ls == lr:
        return [],[]
    # 若漏唱，即标准线大于识别线数量
    elif ls > lr:
        return [ls-lr],[]
    # 多唱的情况
    elif ls!=0:
        min = 10000
        min_i = 0
        min_j = 0
        for i in standard_y:
            for j in recognize_y:
                if abs(i-j) < min:
                    min = abs(i-j)
                    min_i = i
                    min_j = j
        standard_y.remove(min_i)
        recognize_y.remove(min_j)
        get_mismatch_line(standard_y,recognize_y)
    return standard_y,recognize_y


def get_wrong(standard_y,recognize_y):
    if len(standard_y) > 0:
        lost_num = standard_y[0]
    else:
        lost_num = 0
    ex_frames = []
    for i in recognize_y:
        ex_frames.append(i)
    return lost_num,ex_frames

'''
计算多唱扣分，漏唱扣分
'''
def get_scores(standard_y,recognize_y,onsets_total,onsets_strength):
    standard_y, recognize_y = get_mismatch_line(standard_y, recognize_y)
    lost_num, ex_frames = get_wrong(standard_y, recognize_y)
    # print(standard_y,recognize_y)
    lost_score = 0
    ex_score =0
    if lost_num:
        print('漏唱了' + str(lost_num) + '句')
        lost_score = 100 /onsets_total * lost_num
    elif len(ex_frames) > 1:
        for x in ex_frames:
            strength = onsets_strength[int(x)]
            ex_score += int(100 /onsets_total * strength)
    else:
        print('节拍数一致')
    return lost_score,ex_score

def get_deviation(standard_y,recognize_y,codes):
    each_onset_score = 100/len(standard_y)
    score = 0
    total = 0
    for i in range(len(standard_y)-1):
        offset =np.abs((recognize_y[i+1]-recognize_y[i]) /(standard_y[i+1] - standard_y[i]) -1)
        standard_offset = get_code_offset(codes[i])
        if offset <= standard_offset:
            score = 0
        elif offset >= 1:
            score = each_onset_score
        else:
            score = each_onset_score * offset
        total +=score
    return total
def get_code_offset(code):
    offset = 0
    code = re.sub("\D", "", code)  # 筛选数字
    code = int(code)
    if code >= 4000:
        offset = 1/32
    elif code >= 2000:
        offset = 1/16
    elif code >= 1000:
        offset = 1/8
    elif code >= 500:
        offset = 1/4
    elif code >= 250:
        offset = 1/2
    return offset

def get_score(filename):

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

    type_index = get_onsets_index_by_filename(filename)
    y, sr = load_and_trim(filename)
    total_frames_number = get_total_frames_number(filename)

    onsets_frames, onsets_frames_strength = get_onsets_by_all(y, sr)

    # 在此处赋值防止后面实线被移动找不到强度
    recognize_y = onsets_frames

    # 标准节拍时间点
    base_frames = onsets_base_frames(codes[type_index], total_frames_number)
    print("base_frames is {}".format(base_frames))

    min_d, best_y, onsets_frames = get_dtw_min(onsets_frames, base_frames, 65,move=False)
    base_onsets = librosa.frames_to_time(best_y, sr=sr)
    print("base_onsets is {}".format(base_onsets))

    # 节拍时间点
    onstm = librosa.frames_to_time(onsets_frames, sr=sr)
    print("onstm is {}".format(onstm))

    plt.vlines(onstm, -1 * np.max(y), np.max(y), color='b', linestyle='solid')
    plt.vlines(base_onsets, -1 * np.max(y), np.max(y), color='r', linestyle='dashed')

    standard_y = best_y

    codes = get_code(type_index, 1)
    min_d = get_deviation(standard_y, recognize_y, codes)
    score = get_score1(standard_y, recognize_y, len(base_frames), onsets_frames_strength, min_d)

    # # 计算成绩测试
    # print('偏移分值为：{}'.format(min_d))
    # score = get_score1(standard_y, recognize_y, len(base_frames), onsets_frames_strength, min_d)
    # print('最终得分为：{}'.format(score))
    # standard_y, recognize_y = get_mismatch_line(standard_y, recognize_y)
    # lost_num, ex_frames = get_wrong(standard_y, recognize_y)
    #
    # if lost_num:
    #     print('漏唱了' + str(lost_num) + '句')
    # elif len(ex_frames) > 1:
    #     print('多唱的帧 is {}'.format(ex_frames))
    #     ex_frames_time = librosa.frames_to_time(ex_frames[1:], sr=sr)
    #     plt.vlines(ex_frames_time, -1 * np.max(y), np.max(y), color='black', linestyle='solid')
    # else:
    #     print('节拍数一致')
    #

    '''
    调试需要查看图片则取消这部分注释
    '''
    # lost_score, ex_score = get_scores(standard_y, recognize_y, len(base_frames), onsets_frames_strength)
    # print("lost_score, ex_score is : {},{}".format(lost_score, ex_score))
    # plt.show()

    return score

'''
调试则调用此函数
'''
def debug_get_score(filename):
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

    type_index = get_onsets_index_by_filename(filename)
    y, sr = load_and_trim(filename)
    total_frames_number = get_total_frames_number(filename)

    onsets_frames, onsets_frames_strength = get_onsets_by_all(y, sr)
    onsets_frames = get_onsets_frames_for_jz(filename)
    print("onsets_frames len is {}".format(len(onsets_frames)))
    onsets_frames_strength = librosa.onset.onset_strength(y=y, sr=sr)
    onsets_frames_strength = [x/np.max(onsets_frames_strength) for x in onsets_frames_strength]
    # 在此处赋值防止后面实线被移动找不到强度
    recognize_y = onsets_frames

    # 标准节拍时间点
    base_frames = onsets_base_frames(codes[type_index], total_frames_number)
    print("base_frames is {}".format(base_frames))
    print("base_frames len is {}".format(len(base_frames)))

    min_d, best_y, onsets_frames = get_dtw_min(onsets_frames, base_frames, 65)
    base_onsets = librosa.frames_to_time(best_y, sr=sr)
    print("base_onsets is {}".format(base_onsets))

    # 节拍时间点
    onstm = librosa.frames_to_time(onsets_frames, sr=sr)
    print("onstm is {}".format(onstm))

    plt.vlines(onstm, -1 * np.max(y), np.max(y), color='b', linestyle='solid')
    plt.vlines(base_onsets, -1 * np.max(y), np.max(y), color='r', linestyle='dashed')

    standard_y = best_y

    code = get_code(type_index,1)
    modify_recognize_y = recognize_y
    #多唱的情况
    if len(standard_y) < len(recognize_y):
        _, ex_recognize_y = get_mismatch_line(standard_y.copy(), recognize_y.copy())
        modify_recognize_y = [x for x in recognize_y if x not in ex_recognize_y]
        min_d = get_deviation(standard_y,modify_recognize_y,code)
    #漏唱的情况
    if len(standard_y) > len(recognize_y):
        _, lost_standard_y = get_mismatch_line(recognize_y.copy(),standard_y.copy())
        modify_standard_y = [x for x in standard_y if x not in lost_standard_y]
        min_d = get_deviation(modify_standard_y, recognize_y, code)
    #score = get_score1(standard_y, recognize_y, len(base_frames), onsets_frames_strength, min_d)

    # # 计算成绩测试
    print('偏移分值为：{}'.format(min_d))
    score,lost_score,ex_score,min_d = get_score1(standard_y, recognize_y, len(base_frames), onsets_frames_strength, min_d)
    print('最终得分为：{}'.format(score))

    print("lost_score, ex_score,min_d is : {},{},{}".format(lost_score, ex_score,min_d))
    #plt.text(0.2, 0.2, '偏移分值为:'+ str(round(min_d,2)))
    plt.show()

    return score

if __name__ == '__main__':

    # filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏1周(95).wav'

    # filename = './mp3/节奏/节奏1_40227（100）.wav'
    filename = './mp3/节奏/节奏4-01（88）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏1_40441（96）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏8_40213（30）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏四（11）（55）.wav'
    # filename = './mp3/节奏/节奏四（4）（60）.wav'
    # filename = './mp3/节奏/节奏2-02（20）.wav'

    debug_get_score(filename)

