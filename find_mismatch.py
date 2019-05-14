import numpy as np
from create_base import *
from myDtw import *
from grade import *
from cqt_rms import *
from note_lines_helper_test import *

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

def get_deviation(standard_y,recognize_y,codes,each_onset_score):
    #each_onset_score = 100/len(standard_y)
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

def get_deviation_for_note(standard_y,recognize_y,codes,each_onset_score):
    #each_onset_score = 100/len(standard_y)
    score = 0
    total = 0
    length = len(standard_y) if len(standard_y) < len(recognize_y) else len(recognize_y)
    for i in range(length-1):
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

def get_score(filename,code):



    type_index = get_onsets_index_by_filename(filename)
    y, sr = load_and_trim(filename)
    total_frames_number = get_total_frames_number(filename)

    onsets_frames, onsets_frames_strength = get_onsets_by_all(y, sr)

    # 在此处赋值防止后面实线被移动找不到强度
    recognize_y = onsets_frames

    # 标准节拍时间点
    base_frames = onsets_base_frames(code, total_frames_number)
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

    type_index = get_onsets_index_by_filename(filename)
    #y, sr = load_and_trim(filename)
    y, sr = librosa.load(filename)
    total_frames_number = get_total_frames_number(filename)

    #onsets_frames, onsets_frames_strength = get_onsets_by_all(y, sr)
    #onsets_frames = get_onsets_frames_for_jz(filename)
    onsets_frames, best_threshold = get_onsets_by_cqt_rms_optimised(filename)
    # if len(onset_frames_cqt)<topN:
    onsets_frames = get_miss_onsets_by_cqt(y, onsets_frames)
    print("onsets_frames len is {}".format(len(onsets_frames)))
    onsets_frames_strength = librosa.onset.onset_strength(y=y, sr=sr)
    onsets_frames_strength = [x/np.max(onsets_frames_strength) for x in onsets_frames_strength]
    # 在此处赋值防止后面实线被移动找不到强度
    recognize_y = onsets_frames

    # 标准节拍时间点
    base_frames = onsets_base_frames(codes[type_index], total_frames_number - onsets_frames[0])
    base_frames = [x + (onsets_frames[0] - base_frames[0] - 1) for x in base_frames]
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

    standard_y = best_y.copy()

    code = get_code(type_index,1)
    modify_recognize_y = recognize_y
    each_onset_score = 100 / len(standard_y)
    ex_recognize_y = []
    #多唱的情况
    if len(standard_y) < len(recognize_y):
        _, ex_recognize_y = get_mismatch_line(standard_y.copy(), recognize_y.copy())
        modify_recognize_y = [x for x in recognize_y if x not in ex_recognize_y]
        min_d = get_deviation(standard_y,modify_recognize_y,code,each_onset_score)
    #漏唱的情况
    if len(standard_y) > len(recognize_y):
        _, lost_standard_y = get_mismatch_line(recognize_y.copy(),standard_y.copy())
        modify_standard_y = [x for x in standard_y if x not in lost_standard_y]
        min_d = get_deviation(modify_standard_y, recognize_y, code,each_onset_score)
    if len(standard_y) == len(recognize_y):
        min_d = get_deviation(standard_y, recognize_y, code, each_onset_score)
    #score = get_score1(standard_y, recognize_y, len(base_frames), onsets_frames_strength, min_d)

    # # 计算成绩测试
    print('偏移分值为：{}'.format(min_d))
    score,lost_score,ex_score,min_d = get_score1(standard_y, recognize_y, len(base_frames), onsets_frames_strength, min_d)
    print('最终得分为：{}'.format(score))

    print("lost_score, ex_score,min_d is : {},{},{}".format(lost_score, ex_score,min_d))

    # 打印多唱的节拍
    if len(ex_recognize_y) > 0:
        ex_recognize_y_time = librosa.frames_to_time(ex_recognize_y)
        plt.vlines(ex_recognize_y_time, -1 * np.max(y), np.max(y), color='black', linestyle='solid')
    #plt.text(0.2, 0.2, '偏移分值为:'+ str(round(min_d,2)))
    plt.show()

    return score

'''
计算节奏型音频的分数
'''
def get_score_jz(filename,onset_code):
    # onset_code = onset_code.replace(";", ',')
    # onset_code = onset_code.replace("[", '')
    # onset_code = onset_code.replace("]", '')
    # onset_code = [x for x in onset_code.split(',')]

    score,lost_score,ex_score,min_d = get_score_jz_by_cqt_rms_optimised(filename,onset_code)
    #print('最终得分为：{}'.format(score))

    if int(score) < 90:
        score2, lost_score2, ex_score2, min_d2 = get_score_jz_by_onsets_frames_rhythm(filename,onset_code)
        if score2 > score:
            return int(score2), int(lost_score2), int(ex_score2), int(min_d2)

    return int(score),int(lost_score),int(ex_score),int(min_d)

'''
计算节奏型音频的分数
'''
def get_score_jz_by_cqt_rms_optimised(filename,onset_code):

    #type_index = get_onsets_index_by_filename(filename)
    #y, sr = load_and_trim(filename)
    y, sr = librosa.load(filename)
    total_frames_number = get_total_frames_number(filename)

    #onsets_frames, onsets_frames_strength = get_onsets_by_all(y, sr)
    #onsets_frames = get_onsets_frames_for_jz(filename)
    onsets_frames, best_threshold = get_onsets_by_cqt_rms_optimised(filename,onset_code)
    # if len(onset_frames_cqt)<topN:
    onsets_frames = get_miss_onsets_by_cqt(y, onsets_frames)
    #print("onsets_frames len is {}".format(len(onsets_frames)))
    onsets_frames_strength = librosa.onset.onset_strength(y=y, sr=sr)
    onsets_frames_strength = [x/np.max(onsets_frames_strength) for x in onsets_frames_strength]
    # 在此处赋值防止后面实线被移动找不到强度
    onsets_frames = list(set(onsets_frames))
    onsets_frames.sort()
    recognize_y = onsets_frames

    # 标准节拍时间点
    if len(onsets_frames) > 0:
        base_frames = onsets_base_frames(onset_code, total_frames_number - onsets_frames[0])
        base_frames = [x + (onsets_frames[0] - base_frames[0]) for x in base_frames]
        min_d, best_y, onsets_frames = get_dtw_min(onsets_frames, base_frames, 65)
    else:
        base_frames = onsets_base_frames(onset_code, total_frames_number)

    #print("base_frames is {}".format(base_frames))
    #print("base_frames len is {}".format(len(base_frames)))


    min_d, best_y, onsets_frames = get_dtw_min(onsets_frames, base_frames, 65)


    standard_y = best_y.copy()

    code = onset_code
    index = 0
    code = code.replace(";", ',')
    code = code.replace("[", '')
    code = code.replace("]", '')
    if code.find("(") >= 0:
        tmp = [x for x in code.split(',')]
        for i in range(len(tmp)):
            if tmp[i].find("(") >= 0:
                index = i
                break
        code = code.replace("(", '')
        code = code.replace(")", '')
        code = code.replace("-", '')
        code = code.replace("--", '')
    code = [x for x in code.split(',')]
    #code = [int(x) for x in code]
    if index > 0:
        code[index - 1] += code[index]
        del code[index]
    each_onset_score = 100 / len(standard_y)
    ex_recognize_y = []
    #多唱的情况
    if len(standard_y) < len(recognize_y):
        _, ex_recognize_y = get_mismatch_line(standard_y.copy(), recognize_y.copy())
        # 剥离多唱节拍，便于计算整体偏差分
        modify_recognize_y = [x for x in recognize_y if x not in ex_recognize_y]
        min_d = get_deviation(standard_y,modify_recognize_y,code,each_onset_score)
    #漏唱的情况
    if len(standard_y) > len(recognize_y):
        _, lost_standard_y = get_mismatch_line(recognize_y.copy(),standard_y.copy())
        # 剥离漏唱节拍，便于计算整体偏差分
        modify_standard_y = [x for x in standard_y if x not in lost_standard_y]
        min_d = get_deviation(modify_standard_y, recognize_y, code,each_onset_score)
    if len(standard_y) == len(recognize_y):
        min_d = get_deviation(standard_y, recognize_y, code, each_onset_score)
    #score = get_score1(standard_y, recognize_y, len(base_frames), onsets_frames_strength, min_d)

    # # 计算成绩测试
    #print('偏移分值为：{}'.format(min_d))
    score,lost_score,ex_score,min_d = get_score1(standard_y, recognize_y, len(base_frames), onsets_frames_strength, min_d)
    #print('最终得分为：{}'.format(score))

    return int(score),int(lost_score),int(ex_score),int(min_d)


'''
计算节奏型音频的分数
'''
def get_score_for_note(onsets_frames,base_frames,code):


    recognize_y = onsets_frames

    #min_d, best_y, onsets_frames = get_dtw_min(onsets_frames, base_frames, 65)

    base_frames = [x - (base_frames[0] - onsets_frames[0]) for x in base_frames]
    standard_y = base_frames
    print("standard_y is {}".format(standard_y))

    #code = get_code(type_index,2)
    each_onset_score = 100 / len(standard_y)
    print("each_onset_score is {}".format(each_onset_score))
    ex_recognize_y = []

    ex_recognize_y = []
    # 多唱的情况
    if len(standard_y) < len(recognize_y):
        _, ex_recognize_y = get_mismatch_line(standard_y.copy(), recognize_y.copy())
        # 剥离多唱节拍，便于计算整体偏差分
        modify_recognize_y = [x for x in recognize_y if x not in ex_recognize_y]
        min_d = get_deviation_for_note(standard_y, modify_recognize_y, code, each_onset_score)
    # 漏唱的情况
    if len(standard_y) > len(recognize_y):
        _, lost_standard_y = get_mismatch_line(recognize_y.copy(), standard_y.copy())
        # 加上漏唱节拍，便于计算整体偏差分
        modify_recognize_y = recognize_y.copy()
        for x in lost_standard_y:
            modify_recognize_y.append(x)
            modify_recognize_y.sort()
        min_d = get_deviation_for_note(standard_y, modify_recognize_y, code, each_onset_score)
    if len(standard_y) == len(recognize_y):
        min_d = get_deviation_for_note(standard_y, recognize_y, code, each_onset_score)
    #score = get_score1(standard_y, recognize_y, len(base_frames), onsets_frames_strength, min_d)

    # # 计算成绩测试
    #print('偏移分值为：{}'.format(min_d))
    onsets_frames_strength = np.ones(len(recognize_y))
    onsets_frames_strength = [x *0.5 for x in onsets_frames_strength]
    score,lost_score,ex_score,min_d = get_score_detail_for_note(standard_y, recognize_y, len(base_frames), onsets_frames_strength, min_d)
    #print('最终得分为：{}'.format(score))

    return int(score),int(lost_score),int(ex_score),int(min_d)

'''
计算节奏型音频的分数
'''
def get_score_for_note_v2(onsets_frames,base_frames,rhythm_code):
    rhythm_code = rhythm_code.replace(";", ',')
    rhythm_code = rhythm_code.replace("[", '')
    rhythm_code = rhythm_code.replace("]", '')
    rhythm_code = [x for x in rhythm_code.split(',')]
    recognize_y = onsets_frames
    standard_y = base_frames
    #rhythm_code = get_code(type_index, 2)
    each_onset_score = 100 / len(standard_y)

    xc,yc = get_matched_onset_frames_by_path_v3(standard_y, recognize_y)

    if len(xc)<1 or len(yc) <1:
        return 0,0,0,0
    std_number = len(standard_y) - len(xc) + len(recognize_y) - len(yc)
    #print("std_number is {}".format(std_number))

    min_d = get_deviation_for_note(xc,yc, rhythm_code, each_onset_score)

    if (len(standard_y) - len(xc))/len(standard_y) > 0.45:
        score = 30-min_d if 30-min_d > 0 else 10
        return score, 0, 0, 0
    # # 计算成绩测试
    #print('偏移分值为：{}'.format(min_d))
    onsets_frames_strength = np.ones(len(recognize_y))
    onsets_frames_strength = [x *0.5 for x in onsets_frames_strength]
    score,lost_score,ex_score,min_d = get_score_detail_for_note(standard_y, recognize_y, len(base_frames), onsets_frames_strength, min_d)
    #print('最终得分为：{}'.format(score))
    if std_number >= 4:
        #print(len(base_frames))
        score = int(score - each_onset_score*std_number*0.5)

    return int(score),int(lost_score),int(ex_score),int(min_d)

'''
计算节奏型音频的分数
'''
def get_score_jz_by_onsets_frames_rhythm(filename,onset_code):

    #type_index = get_onsets_index_by_filename(filename)
    #y, sr = load_and_trim(filename)
    y, sr = librosa.load(filename)
    total_frames_number = get_total_frames_number(filename)

    #onsets_frames, onsets_frames_strength = get_onsets_by_all(y, sr)
    #onsets_frames = get_onsets_frames_for_jz(filename)
    onsets_frames = get_real_onsets_frames_rhythm(y,modify_by_energy=True,gap=0.1)
    if onsets_frames:
        min_width = 5
        # print("min_width is {}".format(min_width))
        onsets_frames = del_overcrowding(onsets_frames, min_width)
        #print("0. onset_frames_cqt is {}".format(onsets_frames))
    #print("onsets_frames len is {}".format(len(onsets_frames)))
    onsets_frames_strength = librosa.onset.onset_strength(y=y, sr=sr)
    onsets_frames_strength = [x/np.max(onsets_frames_strength) for x in onsets_frames_strength]
    # 在此处赋值防止后面实线被移动找不到强度
    onsets_frames = list(set(onsets_frames))
    onsets_frames.sort()
    recognize_y = onsets_frames

    # 标准节拍时间点
    if len(onsets_frames) > 0:
        base_frames = onsets_base_frames(onset_code, total_frames_number - onsets_frames[0])
        base_frames = [x + (onsets_frames[0] - base_frames[0]) for x in base_frames]
        min_d, best_y, onsets_frames = get_dtw_min(onsets_frames, base_frames, 65)
    else:
        base_frames = onsets_base_frames(onset_code, total_frames_number)

    #print("base_frames is {}".format(base_frames))
    #print("base_frames len is {}".format(len(base_frames)))

    min_d, best_y, onsets_frames = get_dtw_min(onsets_frames, base_frames, 65)


    standard_y = best_y.copy()

    code = onset_code
    index = 0
    code = code.replace(";", ',')
    code = code.replace("[", '')
    code = code.replace("]", '')
    if code.find("(") >= 0:
        tmp = [x for x in code.split(',')]
        for i in range(len(tmp)):
            if tmp[i].find("(") >= 0:
                index = i
                break
        code = code.replace("(", '')
        code = code.replace(")", '')
        code = code.replace("-", '')
        code = code.replace("--", '')
    code = [x for x in code.split(',')]
    # code = [int(x) for x in code]
    if index > 0:
        code[index - 1] += code[index]
        del code[index]
    each_onset_score = 100 / len(standard_y)
    ex_recognize_y = []
    #多唱的情况
    if len(standard_y) < len(recognize_y):
        _, ex_recognize_y = get_mismatch_line(standard_y.copy(), recognize_y.copy())
        # 剥离多唱节拍，便于计算整体偏差分
        modify_recognize_y = [x for x in recognize_y if x not in ex_recognize_y]
        min_d = get_deviation(standard_y,modify_recognize_y,code,each_onset_score)
    #漏唱的情况
    if len(standard_y) > len(recognize_y):
        _, lost_standard_y = get_mismatch_line(recognize_y.copy(),standard_y.copy())
        # 剥离漏唱节拍，便于计算整体偏差分
        modify_standard_y = [x for x in standard_y if x not in lost_standard_y]
        min_d = get_deviation(modify_standard_y, recognize_y, code,each_onset_score)
    if len(standard_y) == len(recognize_y):
        min_d = get_deviation(standard_y, recognize_y, code, each_onset_score)
    #score = get_score1(standard_y, recognize_y, len(base_frames), onsets_frames_strength, min_d)

    # # 计算成绩测试
    #print('偏移分值为：{}'.format(min_d))
    score,lost_score,ex_score,min_d = get_score1(standard_y, recognize_y, len(base_frames), onsets_frames_strength, min_d)
    #print('最终得分为：{}'.format(score))

    return int(score),int(lost_score),int(ex_score),int(min_d)

def check_gap(b,c):
    print("b is {}".format(b))
    print("c is {}".format(c))
    diff1 = np.diff(b)
    print(diff1)
    #b = [15, 72, 90, 109, 128, 221, 240, 277, 296, 315, 334, 352, 446]
    diff2 = np.diff(c)
    print(diff2)
    c = [np.abs(diff1[i] - diff2[i]) for i in range(len(diff1))]
    print(np.sum(c))
    return np.sum(c)

def find_loss(s1,s2):
    best_gap = 100000
    best_x = 0
    for x in s2[1:]:
        tmp = s1.copy()
        tmp.append(x)
        tmp.sort()
        gap = check_gap(tmp, s2)
        if gap < best_gap:
            best_x = x
            best_gap = gap
    return best_x,best_gap



if __name__ == '__main__':

    # filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏1周(95).wav'

    # filename = './mp3/节奏/节奏1_40227（100）.wav'
    filename = './mp3/节奏/节奏4-01（88）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏1_40441（96）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏8_40213（30）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏四（9）（70）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏十（5）（100）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/节奏/节8王（60）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏十（7）（100）.wav'
    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/节奏/节奏五（4）（100）.wav'
    #filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/节奏/节5熙(35).wav'
    # filename = './mp3/节奏/节奏四（4）（60）.wav'
    # filename = './mp3/节奏/节奏2-02（20）.wav'

    #score, lost_score, ex_score, min_d = get_score_jz(filename)
    #print("score, lost_score, ex_score, min_d is {},{},{},{}".format(score, lost_score, ex_score, min_d))

    filename = 'F:/项目/花城音乐项目/样式数据/3.06MP3/旋律/旋4谭（95）.wav'
    onsets_frames = [75, 96, 133, 155, 163, 173, 183, 194, 232, 251, 268, 286, 308]
    base_frames = [0, 17, 51, 68, 76, 85, 93, 102, 119, 135, 152, 169, 186, 203]
    recognize_times = [12, 31, 16, 6, 9, 6, 11, 6, 10, 14, 15, 14, 38]
    type_index = get_onsets_index_by_filename_rhythm(filename)
    code = get_code(type_index,2)
    score, lost_score, ex_score, min_d = get_score_for_note(onsets_frames, base_frames, code)
    print("score, lost_score, ex_score, min_d is {},{},{},{}".format(score, lost_score, ex_score, min_d))

    s1 = [42, 49, 65, 124, 169, 213, 237, 258, 294, 307]
    s2 = [0, 19, 38, 75, 93, 112, 149, 168, 186, 214, 223]
    best_x, best_gap = find_loss(s1, s2)
    print("best_x,best_gap is {}===={}".format(best_x,best_gap))
