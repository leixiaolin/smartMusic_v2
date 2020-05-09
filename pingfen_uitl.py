# -*- coding:utf-8 -*-
import numpy as np
from LscHelper import find_lcseque,my_find_lcseque
from pinyin_util import modify_tyz_by_position,check_tyz

'''
音高节奏评分
pitch_time = [0,1, 1.5, 2, 3, 3.5, 4, 5, 6, 8, 9, 9.5, 10, 10.5, 11, 11.5, 12, 16, 17, 17.5, 18, 19, 19.5, 20, 21,
               21.5, 22, 23, 24, 25, 26, 26.5, 27,27.5, 28,32]
'''
def notation_rhythm_score(pitch_time,):
    pass

'''
根据标准歌词、歌词时间点、标准音符和音符时间点获取每个歌词对应的音符
standard_kc = '喜爱春天的人儿是心地纯洁的人像紫罗兰花儿一样是我知心朋友加'
standard_kc_time = [0,1, 2, 3, 3.5, 4, 5, 6, 8, 9, 10, 11, 11.5, 12, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 26.5, 27, 28,30,32]
standard_notations = '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
standard_notation_time = [0,1,1.5,2,3,3.5,4,5,6,8,9,9.5,10,10.5,11,11.5,12,16,17,17.5,18,19,19.5,20,21,21.5,22,23,24,25,26,26.5,27,27.5,28,32]
'''
def get_notations_on_kc(standard_kc,standard_kc_time,standard_notations,standard_notation_time):
    standard_kc_list = list(standard_kc)
    standard_notations_list = standard_notations.split(",")
    kc_with_notations = {}
    for i,kc in enumerate(standard_kc_list):
        start_time = standard_kc_time[i]
        end_time = standard_kc_time[i+1]
        notation_positions = [i for i,t in enumerate(standard_notation_time) if t>= start_time and t<end_time ] #音符开始时间落在歌词时间区间上的都算
        select_notations = [standard_notations_list[i] for i in notation_positions]
        kc_with_notations[i] = (kc,notation_positions,select_notations)
        print("{},{}".format(i,kc_with_notations[i]))
    return kc_with_notations


'''
歌词节奏评分
standard_kc = '喜爱春天的人儿是心地纯洁的人像紫罗兰花儿一样是我知心朋友加'
standard_kc_time = [0,1, 2, 3, 3.5, 4, 5, 6, 8, 9, 10, 11, 11.5, 12, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 26.5, 27, 28,30,32]
kc_detail = {20: ('惜', 0), 144: ('爱', 1), 236: ('春天', 2), 392: ('的', 4), 440: ('人', 5), 552: ('儿', 6), 640: ('时', 7), 824: ('心地', 8), 1040: ('纯洁', 10), 1188: ('的', 12), 1624: ('相思', 13), 1832: ('罗', 15), 1936: ('兰花', 16), 2100: ('花儿', 18), 2240: ('一样', 20), 2425: ('是', 22), 2545: ('我', 23), 2645: ('知心', 24), 2745: ('朋友', 26), 0: ('', 28)}
all_test_kc = '喜爱春天的人儿是心地纯洁的像紫罗兰花花儿一样是我知心朋友'
'''
def kc_rhythm_score(standard_kc,standard_kc_time,kc_detail,test_kc,real_loss_positions,end_time,score_seted):
    #待测歌词的时间点
    detail_time = [round((value) / 100, 2) for value in kc_detail.keys() if value > 0]
    # detail_time.append(standard_kc_time[-1]) #从标准时间序列中添加结束点
    detail_time.append(end_time)
    detail_time_diff = np.diff(detail_time)
    detail_time_diff = list(detail_time_diff)
    # print("detail_time_diff is {}, size is {}".format(detail_time_diff, len(detail_time_diff)))
    standard_kc_time_diff = np.diff(standard_kc_time)
    standard_kc_time_diff = list(standard_kc_time_diff)
    # print("standard_kc_time_diff is {}, size is {}".format(standard_kc_time_diff, len(standard_kc_time_diff)))
    each_score = round(score_seted/len(standard_kc),2)
    total_score = 0

    modify_test_kc = modify_tyz_by_position(standard_kc, test_kc)
    # 获取最大公共序列及其相关位置信息
    lcseque, standard_positions, test_positions = get_lcseque_and_position(standard_kc, modify_test_kc)
    if len(lcseque) < len(standard_kc) * 0.3 and len(lcseque) < 5:
        total_score, score_detail = 0,'歌词节奏评分项总分{}，每个歌词的分值为{}，歌词匹配结果较差，该项评分计为0分，请检查演唱内容是否与题目一致！'.format(score_seted,each_score)
        return total_score, score_detail

    score_detail = '歌词节奏评分项总分{}，每个歌词的分值为{}，下列歌词可能存在失分情况：'.format(score_seted,each_score) + '\n'
    keys = list(kc_detail.keys())
    offset_standard = 0.45
    for n,tup in enumerate(kc_detail.values()):
        try:
            start_time = round(keys[n]/100,2)
            ks = list(tup[0])
            p = tup[1]
            ps = [p]  # 保存歌词的位置
            if len(ks) > 1:     # 多个歌词在一起的情况
                for k in range(1,len(ks)):
                    ps.append(p+k)  # 获取每个歌词在待测歌词序列中的位置
            old_ps = ps.copy()

            test_duration = detail_time_diff[n] #计算该歌词的时长
            test_duration = round(test_duration,2)

            ps = [p for p in ps if p in test_positions] #筛选出在最大公共序列出现过的歌词的位置，如果是多唱的歌词，ps会为空
            offset = [o-ps[0] for o in old_ps]
            if len(ps) == 0:  # 歌词未在最大公共序列中出现过，属于多唱歌词
                str_detail = "{}: 开始于{}秒，持续时长为{}秒,该歌词未在标准歌词中出现，属于多唱歌词".format(tup[0],start_time,test_duration)
                # print(str_detail)
                score_detail += str_detail +'\n'
                continue
            tmp = [i for i,p in enumerate(test_positions) if p in ps] # 获取歌词位置的下标
            standard_ps = [standard_positions[i] for i in tmp]  # 获取歌词位置对应的标准歌词的下标
            standard_ps = [standard_ps[0] + o for o in offset]
            standard_duration = np.sum([standard_kc_time_diff[i] for i in standard_ps]) #计算该歌词对应的标准时长
            standard_duration = round(standard_duration, 2)

            offset_duration = round(np.abs(test_duration - standard_duration),2)
            offset_duration = round(offset_duration/standard_duration,2)
            if offset_duration <= offset_standard: #如果偏差小于25%，即可得分
                score = round(each_score * len(ps),2)
                total_score += score
                # str_detail = "{}: 开始于{}秒， 持续时长为{}秒, 标准时长为{}秒, 偏差值为{},得分为{}".format(tup[0],start_time,test_duration,standard_duration, offset_duration,score)
            else:
                after_this_standard_positions = [sp for sp in standard_positions if sp > standard_ps[-1]] # 该歌词之后的匹配上的标准歌词位置
                if len(after_this_standard_positions) > 0 and after_this_standard_positions[0] - standard_ps[-1] > 1: #如果该歌词有漏歌词的情况，即与后面的第一个位置的间隔会大于1
                    add_positions = [standard_ps[-1] + i for i in range(5) if standard_ps[-1] + i < after_this_standard_positions[0]]
                    add_durations = np.sum([standard_kc_time_diff[i] for i in add_positions])
                    add_durations = round(add_durations, 2)
                    standard_duration_added = standard_duration + add_durations
                    offset_duration = round(np.abs(test_duration - standard_duration_added), 2)
                    offset_duration = round(offset_duration / standard_duration_added, 2)
                    if offset_duration <= offset_standard:  # 如果偏差小于25%，即可得分
                        score = round(each_score * len(ps), 2)
                        total_score += score
                        # str_detail = "{}: 开始于{}秒， 持续时长为{}秒, 标准时长为{}秒, 偏差值为{},得分为{}".format(tup[0],start_time,test_duration,standard_duration, offset_duration,score)
                    else:
                        str_detail ="{}:  开始于{}秒，持续时长为{}秒, 标准时长为{}秒，与标准值偏差率为{},得分为{}".format(tup[0],start_time,test_duration,standard_duration, offset_duration, 0)
                        score_detail += str_detail + '\n'
                else:
                    str_detail ="{}:  开始于{}秒，持续时长为{}秒, 标准时长为{}秒，与标准值偏差率为{},得分为{}".format(tup[0],start_time,test_duration,standard_duration, offset_duration, 0)
                    score_detail += str_detail + '\n'
            # print(str_detail)
        except Exception:
            # print(tup[0] + "is error")
            pass
    # 找出未识别出来的歌词并进行分析
    loss_positions, loss_notations_in_standard = get_lossed_standard_notations(standard_kc, modify_test_kc)
    # no_score_in_loss_positions = [lp for lp in loss_positions if lp not in real_loss_positions]
    no_score_in_loss_notations = [loss_notations_in_standard[i] for i,lp in enumerate(loss_positions) if lp in real_loss_positions]
    if len(no_score_in_loss_notations) >0:
        str_detail ="{}: 未能识别出这些歌词，这些歌词得分为0".format(''.join(no_score_in_loss_notations))
        score_detail += str_detail + '\n'
    total_score += (len(loss_positions) - len(no_score_in_loss_notations)) * each_score
    total_score = round(total_score,2)
    total_score = total_score if total_score < score_seted else score_seted

    if total_score == score_seted:
        score_detail = "歌词节奏评分项总分{}，每个歌词的分值为{}，未存在失分的情况".format(score_seted,each_score)
    return total_score,score_detail

'''
音高评分算法
standard_notations = '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,6,7-,3,2,1,7-,1,6-'
numbered_notations = [None, '3', '2', '2', '1', '1', '7', '6', '6', '6', None, '4', '4', '4', '3', '2', '1', '2', '4', '3', None, '3', '4', '4', '3', '2', '4', '3', '1', '6', '7', '7', '3', '2', '1', '1', '6']
'''
def pitch_score(standard_notations,numbered_notations,standard_notation_times,test_times,score_seted):
    each_score = round(score_seted / len(standard_notations.split(',')), 2)
    total_score = score_seted
    score_detail = '音高评分项总分为{}，每个音高的分值为{}，下列音高可能存在失分情况：'.format(score_seted,each_score) + '\n'

    standard_notations = standard_notations.split(',')
    standard_notations = [s[0] for s in standard_notations]
    standard_notations = ''.join(standard_notations)

    numbered_notations = [n for n in numbered_notations if n is not None]
    numbered_notations = ''.join(numbered_notations)
    # print("standard_notations is {},size is {}".format(standard_notations, len(standard_notations)))
    # print("numbered_notations is {},size is {}".format(numbered_notations, len(numbered_notations)))

    #找出未匹配的音高，并对未匹配的每个音高进行分析
    lcseque, standard_positions, test_positions = get_lcseque_and_position(standard_notations, numbered_notations)
    loss_positions, loss_notations_in_standard = get_lossed_standard_notations(standard_notations, numbered_notations)
    real_loss_positions = []
    for j,lp in enumerate(loss_positions):
        loss_notation = 0
        try:
            # 未匹配的音高名称
            loss_notation = int(loss_notations_in_standard[j][0])

            #判断该标准时间点前后1秒是否存在疑似匹配对象
            gap = test_times[0] - standard_notation_times[0]
            start_time = standard_notation_times[lp] - 1 if standard_notation_times[lp] - 1 > 0 else 0
            end_time = standard_notation_times[lp] + 1
            start_time,end_time = start_time + gap, end_time + gap
            flag = check_by_nearly_notaions(loss_notation,start_time,end_time,numbered_notations,test_times)

            # 该未匹配音高的前一个准点
            anchor_before = int(lp) - 1
            anchor_before_in_test = [test_positions[i] for i, a in enumerate(standard_positions) if a <= anchor_before][-1]
            # 该未匹配音高的后一个准点
            anchor_after = int(lp) + 1
            anchor_after_in_test = [test_positions[i] for i,a in enumerate(standard_positions) if a >= anchor_after][0]
            if anchor_before_in_test + 1 == anchor_after_in_test: # 如果前后准点相临
                # 根据当前音高是否等于前一个准点的音高，如果等于即是存在连音的情况，可计分
                if loss_notation == int(numbered_notations[anchor_before_in_test]):
                    # total_score += each_score
                    pass
                elif flag:
                    total_score -= round(each_score * 0.5, 2)
                    str_detail = "第{}个音高:{} 未有精准匹配项但存在疑似匹配结果，扣{}分".format(lp+1, loss_notation,round(each_score * 0.5, 2))
                    score_detail += str_detail + '\n'
                else:
                    str_detail = "第{}个音高:{} 未匹配，扣{}分".format(lp+1, loss_notation,each_score)
                    total_score -= each_score
                    score_detail += str_detail + '\n'
                    real_loss_positions.append(lp)
            else: # 如果前后准点不相临
                # 获取前后准点之间的音高
                tmp = [int(a) for i,a in enumerate(numbered_notations) if i > anchor_before_in_test and i < anchor_after_in_test]
                offset = [np.abs(loss_notation - t) for t in tmp] # 音高差值
                if np.min(offset) <= 1: #如果音高差值不超过2，可计半分
                    total_score -= round(each_score * 0.5, 2)
                    str_detail = "第{}个音高:{} 识别结果与标准差值小于等于1，扣{}分".format(lp+1, loss_notation,round(each_score * 0.5, 2))
                elif flag:
                    total_score -= round(each_score * 0.5, 2)
                    str_detail = "第{}个音高:{} 未有精准匹配项但存在疑似匹配结果，扣{}分".format(lp+1, loss_notation,round(each_score * 0.5, 2))
                    score_detail += str_detail + '\n'
                else:
                    str_detail = "第{}个音高:{} 识别结果与标准差值大于1，扣{}分".format(lp+1, loss_notation,each_score)
                    total_score -= each_score
                    real_loss_positions.append(lp)
                    score_detail += str_detail + '\n'
        except Exception:
            real_loss_positions.append(lp)
            str_detail = "第{}个音高:{}  is error，扣{}分".format(lp+1,loss_notation,each_score)
            total_score -= each_score
            score_detail += str_detail + '\n'
        # print(str_detail)
    if total_score == score_seted:
        score_detail = "未存在失分的情况"
    return round(total_score,2), score_detail,real_loss_positions

def check_by_nearly_notaions(standard_notation,start_time,end_time,numbered_notations,test_times):
    standard_notation = int(standard_notation)
    flag = False
    for i,t in enumerate(test_times):
        if t >= start_time and t < end_time:
            test_notation = int(numbered_notations[i])
            if standard_notation == 7 and test_notation == 1:
                flag = True
            elif standard_notation == 1 and test_notation == 7:
                flag = True
            elif np.abs(test_notation - standard_notation) <= 1:
                flag = True
            if flag:
                return flag
    return flag
'''
音符节奏评分算法
standard_notations = '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,6,7-,3,2,1,7-,1,6-'
standard_notation_time = [0,1,1.5,2,3,3.5,4,5,6,8,9,9.5,10,10.5,11,11.5,12,16,17,17.5,18,19,19.5,20,21,21.5,22,23,24,25,26,26.5,27,27.5,28,32]
numbered_notations = [None, '3', '2', '2', '1', '1', '7', '6', '6', '6', None, '4', '4', '4', '3', '2', '1', '2', '4', '3', None, '3', '4', '4', '3', '2', '4', '3', '1', '6', '7', '7', '3', '2', '1', '1', '6']
test_times is [0.29, 0.75, 2.02, 2.41, 3.45, 3.93, 4.37, 5.53, 6.38, 6.59, 7.8, 8.37, 8.61, 9.5, 9.86, 10.13, 10.45, 10.99, 11.51, 11.97, 12.25, 12.67, 13.49, 16.25, 16.49, 17.41, 17.85, 18.33, 19.15, 19.41, 19.81, 20.01, 20.29, 20.56, 21.35, 21.87, 22.37, 23.85, 24.27, 24.54, 24.82, 25.47, 25.84, 26.43, 27.11, 27.43, 28.0, 28.48, 30.38,32]
'''
def notation_duration_score(standard_notations,standard_notation_time,numbered_notations,test_times,end_time,real_loss_positions,score_seted):
    # end_time = test_times[-1]
    each_score = round(score_seted / len(standard_notations.split(',')), 2)
    total_score = 0
    score_detail = '音符节奏评分项总分为{}，每个音高的分值为{}，下列音高可能存在失分情况：'.format(score_seted,each_score) + '\n'

    standard_notations = standard_notations.split(',')
    standard_len = len(standard_notations)
    standard_notations = [s[0] for s in standard_notations]
    standard_notations = ''.join(standard_notations)

    test_times = [test_times[i] for i,n in enumerate(numbered_notations) if n is not None]
    test_times.append(end_time)
    test_times_diff = np.diff(test_times)
    # print("test_times_diff is {},size is {}".format(test_times_diff, len(test_times_diff)))
    numbered_notations = [n for n in numbered_notations if n is not None]
    numbered_notations = ''.join(numbered_notations)
    numbered_notations_list = list(numbered_notations)
    # print("standard_notations is {},size is {}".format(standard_notations, len(standard_notations)))
    # print("numbered_notations is {},size is {}".format(numbered_notations, len(numbered_notations)))

    standard_notation_time_diff = np.diff(standard_notation_time)
    # print("standard_notation_time_diff is {},size is {}".format(standard_notation_time_diff, len(standard_notation_time_diff)))

    # 找出未匹配的音高，并对未匹配的每个音高进行分析
    loss_positions, loss_notations_in_standard = get_lossed_standard_notations(standard_notations, numbered_notations)
    false_loss_positions = []
    # print("loss_positions is {},size is {}".format(loss_positions, len(loss_positions)))
    # 找出最大公共子序列，并对每个匹配上的音符时长进行判断计分处理
    lcseque, standard_positions, test_positions = get_lcseque_and_position(standard_notations, numbered_notations)
    offset_standard = 0.45
    for l,s in enumerate(lcseque):
        if l < len(lcseque):
            standard_position = standard_positions[l]
            standard_duration = standard_notation_time_diff[standard_position]
            standard_duration = round(standard_duration,2)
            test_position = test_positions[l]
            test_duration = test_times_diff[test_position]
            test_duration = round(test_duration,2)
            offset = np.abs(test_duration - standard_duration)
            offset = offset / standard_duration
            offset = round(offset,2)

            if standard_position == standard_len -1: #最后一个要求最低，只有匹配上就算对
                total_score += each_score
            else:
                if offset <= offset_standard: # 时长偏差小于0.25，该音符可得满分
                    total_score += each_score
                else:
                    if int(standard_position + 1) in loss_positions: #如果后一个音符在未匹配序列中，后一个音符在有可能是漏识别的音符，所以要两个的时长一起来判断
                        next_standard_duration = standard_notation_time_diff[standard_position+1]
                        if np.abs(test_duration / (next_standard_duration + standard_duration) -1) < 0.5:
                            total_score += each_score
                            false_loss_positions.append(int(standard_position + 1) )
                            continue
                        else:
                            str_detail = "第{}个音高:{} 开始于{}秒，结束于{}秒，时长为{}秒，标准时长为{}秒，与标准值偏差率为{}，大于规定范围，扣{}分".format(standard_position+1, s,test_times[test_position],test_times[test_position+1],test_duration,standard_duration, offset,each_score)
                            score_detail += str_detail + '\n'
                    else:
                        # 根据附近的音符时长来判断，即附近差值不超过1的音符时长都可能是该音符比较对象
                        nearly_positions = [p for p in range(test_position-1,test_position+2) if p in range(len(numbered_notations)) and p in range(len(test_times_diff)) and np.abs(int(numbered_notations[p]) - int(s)) <=1 ]
                        nearly_durations = [test_times_diff[n] for n in nearly_positions]
                        nearly_durations_added = [nearly_durations[i-1] + nearly_durations[i] for i in range(1,len(nearly_durations))]
                        nearly_sum = [np.sum(nearly_durations)]
                        all_nearly_durations = nearly_durations + nearly_durations_added + nearly_sum
                        offset_on_nearly = [np.abs((nd - standard_duration) / standard_duration) for nd in all_nearly_durations]
                        offset = np.min(offset_on_nearly)
                        offset = round(offset,2 )
                        if offset <= offset_standard:  # 时长偏差小于0.25，该音符可得满分
                            total_score += each_score
                        else:
                            str_detail = "第{}个音高:{} 开始于{}秒，结束于{}秒，时长为{}秒，标准时长为{}秒，与标准值偏差率为{}，大于规定范围，扣{}分".format(standard_position+1, s,test_times[test_position],test_times[test_position+1],test_duration,standard_duration, offset,each_score)
                            score_detail += str_detail + '\n'
                    #     print("in loss")
    no_score_in_loss_notations = [loss_notations_in_standard[i] for i,lp in enumerate(loss_positions) if lp not in false_loss_positions]
    no_score_in_loss_positions = [lp for i, lp in enumerate(loss_positions) if lp not in false_loss_positions]
    if len(no_score_in_loss_notations) > 0:
        for i,nn in enumerate(no_score_in_loss_notations):
            if no_score_in_loss_positions[i] in real_loss_positions:
                str_detail = "第{}个音高:{} 未能识别，得分为0".format(no_score_in_loss_positions[i]+1,nn)
                score_detail += str_detail + '\n'
            else:
                total_score += each_score *0.5
                str_detail = "第{}个音高:{} 未有精准匹配项但存在疑似匹配结果，得分为{}".format(no_score_in_loss_positions[i] + 1, nn,round(each_score *0.5,2))
                score_detail += str_detail + '\n'
    total_score += (len(loss_positions) - len(no_score_in_loss_notations)) * each_score
    total_score = round(total_score, 2)
    total_score = total_score if total_score < score_seted else score_seted
    if total_score == score_seted:
        score_detail = "未存在失分的情况"
    return round(total_score,2), score_detail

'''
获取最大公共序列及其相关位置信息
'''
def get_lcseque_and_position(standard_kc,test_kc):
    lcseque = find_lcseque(standard_kc,test_kc)
    test_start = 0
    standard_start = 0
    standard_kc_list, test_kc_list = list(standard_kc),list(test_kc)
    test_positions = []
    standard_positions = []
    for i,s in enumerate(lcseque):
        # 获取最大公共序列在标准序列中的位置
        tmp = standard_kc_list[standard_start:]
        standard_index = tmp.index(s)
        standard_positions.append(standard_start + standard_index)
        old_standard_start = standard_start + standard_index
        standard_start = standard_start + standard_index + 1

        # 获取最大公共序列在待测序列中的位置
        # tmp = test_kc_list[test_start:]
        # if s in tmp:
        #     test_index = tmp.index(s)
        #     test_positions.append(test_start + test_index)
        #     test_start = test_start + test_index + 1

        tmp = test_kc_list[test_start:]
        if s in tmp:
            test_index = tmp.index(s)
            if test_index + 2 < len(tmp) and i+2 < len(lcseque) and tmp[test_index + 1] == s:
                if tmp[test_index + 1] == lcseque[i+1]:
                    test_positions.append(test_start + test_index)
                    test_start = test_start + test_index + 1
                elif tmp[test_index + 2] == lcseque[i+2]:
                    test_positions.append(test_start + test_index + 1)
                    test_start = test_start + test_index + 2
                else:
                    test_positions.append(test_start + test_index)
                    test_start = test_start + test_index + 1
            else:
                test_positions.append(test_start + test_index)
                test_start = test_start + test_index + 1
    return lcseque,standard_positions,test_positions

'''
当识别歌词存在漏词的情况，根据音高匹配结果修正识别结果，即修正test_kc和kc_detail
'''
def modify_iat_result(standard_kc,standard_notations,standard_notations_times,test_kc,kc_detail,numbered_notations,merge_times):
    try:
        lcseque, standard_positions, test_positions = get_lcseque_and_position(standard_kc, test_kc)
        if len(lcseque) < len(standard_kc): #如果识别歌词存在漏词的情况，找到漏掉的歌词的位置
            all_positions = [i for i in range(len(standard_kc))]
            loss_positions = [i for i in all_positions if i not in standard_positions]
            for lp in loss_positions:
                anchor_tmp = [i for i,p in enumerate(standard_positions) if p < lp]
                anchor_before = anchor_tmp[-1]
                anchor_before_position = test_positions[anchor_before] #漏掉的歌词前面能匹配的位置点
                anchor_before_time = get_anchor_before_time_from_kc_detail(kc_detail,anchor_before_position)
                anchor_tmp = [i for i,p in enumerate(standard_positions) if p > lp]
                anchor_after = anchor_tmp[0]
                anchor_after_position = test_positions[anchor_after] #漏掉的歌词后面能匹配的位置点
                anchor_after_time = get_anchor_after_time_from_kc_detail(kc_detail, anchor_after_position)
                select_numbered_notations = [n for i,n in enumerate(numbered_notations) if merge_times[i] > anchor_before_time and merge_times[i] < anchor_after_time]
                for s in select_numbered_notations:
                    if s == standard_notations[lp]:
                        pass # 待完善

        else:
            test_kc_modifed, kc_detail_modifed = test_kc,kc_detail
    except Exception:
        # test_kc_modifed, kc_detail_modifed = test_kc, kc_detail
        pass
    return test_kc_modifed, kc_detail_modifed

def get_anchor_before_time_from_kc_detail(kc_detail,anchor_position):
    tmp = []
    for (k, v) in kc_detail.items():
        time = round(k/100,2)
        position = v[1]
        if position <= anchor_position:
            tmp.append(time)
    return tmp[-1]

def get_anchor_after_time_from_kc_detail(kc_detail,anchor_position):
    tmp = []
    for (k, v) in kc_detail.items():
        time = round(k/100,2)
        position = v[1]
        if position >= anchor_position:
            tmp.append(time)
    return tmp[0]

def get_lossed_standard_notations(standard_notations, numbered_notations):
    lcseque, standard_positions, test_positions = get_lcseque_and_position(standard_notations, numbered_notations)
    all_positions = [i for i in range(len(standard_notations))]
    loss_positions = [i for i in all_positions if i not in standard_positions]
    loss_notations_in_standard = [standard_notations[i] for i in loss_positions]
    # print("lcseque is {}, size is {}".format(lcseque, len(lcseque)))
    # print("standard_positions is {}, size is {}".format(standard_positions, len(standard_positions)))
    # print("test_positions is {}, size is {}".format(test_positions, len(test_positions)))
    # print("loss_positions is {}, size is {}".format(loss_positions, len(loss_positions)))
    # print("loss_notations_in_standard is {}, size is {}".format(loss_notations_in_standard, len(loss_notations_in_standard)))
    return loss_positions,loss_notations_in_standard
'''
整合三个评分项（歌词节奏、音符节奏、音高）
'''
def get_all_scores(standard_kc,standard_kc_time,test_kc,standard_notations, numbered_notations,standard_notation_time,test_times,kc_detail,end_time):
    score_seted = 35
    # standard_notations = '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,6,7-,3,2,1,7-,1,6-'
    # numbered_notations = [None, '3', '2', '2', '1', '1', '7', '6', '6', '6', None, '4', '4', '4', '3', '2', '1', '2',
    #                       '4', '3', None, '3', '4', '4', '3', '2', '4', '3', '1', '6', '7', '7', '3', '2', '1', '1',
    #                       '6']
    pitch_total_score, pitch_score_detail, real_loss_positions = pitch_score(standard_notations, numbered_notations,standard_notation_time,test_times,score_seted)
    # print("pitch_total_score is {}".format(pitch_total_score))
    # print("pitch_score_detail is {}".format(pitch_score_detail))
    # print("real_loss_positions is {}".format(real_loss_positions))

    # print("====================================================================================================")
    # standard_notations = '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
    # numbered_notations = [None, '3', '3', '2', '1', '1', '7', '6', '6', '5', '6', None, '4', '4', '3', '2', '1', '2',
    #                       '4', '3', None, '4', '4', '3', '2', '2', '4', '3', '3', '7', '5', '6', None, '7', '1', '3',
    #                       '2', '1', '7', '1', '6', '6']
    # standard_notation_time = [0, 1, 1.5, 2, 3, 3.5, 4, 5, 6, 8, 9, 9.5, 10, 10.5, 11, 11.5, 12, 16, 17, 17.5, 18, 19,
    #                           19.5, 20, 21, 21.5, 22, 23, 24, 25, 26, 26.5, 27, 27.5, 28, 32]
    # test_times = [0.2, 0.47, 1.44, 1.97, 2.36, 3.48, 3.92, 4.4, 5.52, 6.4, 6.61, 8.24, 8.56, 9.55, 10.1, 10.4, 11.05,
    #               11.52, 11.88, 12.66, 16.24, 16.45, 17.49, 17.96, 18.32, 19.36, 19.96, 20.41, 21.0, 22.04, 22.4, 22.8,
    #               24.25, 24.57, 25.45, 25.66, 26.45, 26.97, 27.45, 27.93, 28.47, 29.84]
    # end_time = 32
    score_seted = 35
    notation_duration_total_score, notation_duration_score_detail = notation_duration_score(standard_notations,
                                                                                            standard_notation_time,
                                                                                            numbered_notations,
                                                                                            test_times, end_time,
                                                                                            real_loss_positions,
                                                                                            score_seted)
    # print("notation_duration_total_score is {}".format(notation_duration_total_score))
    # print("notation_duration_score_detail is {}".format(notation_duration_score_detail))

    # standard_kc = '喜爱春天的人儿是心地纯洁的人像紫罗兰花儿一样是我知心朋友'
    # standard_kc_time = [0, 1, 2, 3, 3.5, 4, 5, 6, 8, 9, 10, 11, 11.5, 12, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,26.5, 27, 28, 32]
    # standard_notations = '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
    # standard_notation_time = [0, 1, 1.5, 2, 3, 3.5, 4, 5, 6, 8, 9, 9.5, 10, 10.5, 11, 11.5, 12, 16, 17, 17.5, 18, 19, 19.5, 20, 21, 21.5, 22, 23, 24, 25, 26, 26.5, 27, 27.5, 28, 32]
    # kc_with_notations = get_notations_on_kc(standard_kc, standard_kc_time, standard_notations, standard_notation_time)
    # print("kc_with_notations is {}".format(kc_with_notations))

    # standard_kc = '喜爱春天的人儿是心地纯洁的人像紫罗兰花儿一样是我知心朋友'
    # test_kc = '惜爱春天的人儿时心地纯洁的相思罗兰花花儿一样是我知心朋友'
    # test_kc = '喜爱春天的人儿时心地纯洁的相思罗兰花花儿一样是我知心朋友'

    # print("====================================================================================================")

    # standard_kc_time = [0, 1, 2, 3, 3.5, 4, 5, 6, 8, 9, 10, 11, 11.5, 12, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
    #                     26.5, 27, 28, 32]
    # kc_detail = {20: ('惜', 0), 144: ('爱', 1), 236: ('春天', 2), 392: ('的', 4), 440: ('人', 5), 552: ('儿', 6),
    #              640: ('时', 7), 824: ('心地', 8), 1040: ('纯洁', 10), 1188: ('的', 12), 1624: ('相思', 13), 1832: ('罗', 15),
    #              1936: ('兰花', 16), 2100: ('花儿', 18), 2240: ('一样', 20), 2425: ('是', 22), 2545: ('我', 23),
    #              2645: ('知心', 24), 2745: ('朋友', 26)}

    score_seted = 30
    kc_duration_total_score, kc_rhythm_sscore_detail = kc_rhythm_score(standard_kc, standard_kc_time, kc_detail, test_kc, real_loss_positions,end_time,score_seted)
    # print("kc_rhythm_score is {}".format(kc_rhythm_score))
    # print("kc_rhythm_sscore_detail is {}".format(kc_rhythm_sscore_detail))
    total_score = pitch_total_score + notation_duration_total_score + kc_duration_total_score
    total_score = round(total_score,2)
    return total_score,pitch_total_score,notation_duration_total_score,kc_duration_total_score,pitch_score_detail,notation_duration_score_detail,kc_rhythm_sscore_detail

def get_all_scores_by_st(standard_kc,standard_kc_time,standard_notations, numbered_notations,standard_notation_time,test_times,kc_detail,end_time):
    score_seted = 35
    notation_duration_total_score, notation_duration_score_detail, pitch_total_score, pitch_score_detail = notation_duration_and_pitch_score_by_st(
        standard_notations, standard_notation_time, numbered_notations, test_times, end_time, score_seted)
    # print("notation_duration_total_score is {}".format(notation_duration_total_score))
    # print("notation_duration_score_detail is {}".format(notation_duration_score_detail))
    # print("pitch_total_score is {}".format(pitch_total_score))
    # print("pitch_score_detail is {}".format(pitch_score_detail))

    score_seted = 30
    # print("======standard_kc is {}".format(standard_kc))
    # print("======standard_kc_time is {}".format(standard_kc_time))
    # print("======kc_detail is {}".format(kc_detail))
    # print("======end_time is {}".format(end_time))
    kc_duration_total_score, kc_duration_score_detail = kc_duration_score_by_st(standard_kc, standard_kc_time,
                                                                                kc_detail, end_time, score_seted)
    # print("kc_duration_total_score is {}".format(kc_duration_total_score))
    # print("kc_duration_score_detail is {}".format(kc_duration_score_detail))
    total_score = notation_duration_total_score + pitch_total_score + kc_duration_total_score
    total_score = round(total_score, 2)
    return total_score,pitch_total_score,notation_duration_total_score,kc_duration_total_score,pitch_score_detail,notation_duration_score_detail,kc_duration_score_detail
'''
以标准音符时间点来计算音符节奏得分和音高得分
1、若该时间段内正确音符连续时长占比大于50%，则该音符节奏可得分；
2、若该时间段内存在正确音高，则该音高可得分。
standard_notations = '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,6,7-,3,2,1,7-,1,6-'
standard_notation_time = [0,1,1.5,2,3,3.5,4,5,6,8,9,9.5,10,10.5,11,11.5,12,16,17,17.5,18,19,19.5,20,21,21.5,22,23,24,25,26,26.5,27,27.5,28,32]
numbered_notations = [None, '3', '2', '2', '1', '1', '7', '6', '6', '6', None, '4', '4', '4', '3', '2', '1', '2', '4', '3', None, '3', '4', '4', '3', '2', '4', '3', '1', '6', '7', '7', '3', '2', '1', '1', '6']
test_times is [0.29, 0.75, 2.02, 2.41, 3.45, 3.93, 4.37, 5.53, 6.38, 6.59, 7.8, 8.37, 8.61, 9.5, 9.86, 10.13, 10.45, 10.99, 11.51, 11.97, 12.25, 12.67, 13.49, 16.25, 16.49, 17.41, 17.85, 18.33, 19.15, 19.41, 19.81, 20.01, 20.29, 20.56, 21.35, 21.87, 22.37, 23.85, 24.27, 24.54, 24.82, 25.47, 25.84, 26.43, 27.11, 27.43, 28.0, 28.48, 30.38,32]
'''
def notation_duration_and_pitch_score_by_st(standard_notations,standard_notation_time,numbered_notations,test_times,end_time,score_seted):
    # end_time = test_times[-1]
    each_score = round(score_seted / len(standard_notations.split(',')), 2)

    notation_duration_total_score = 0
    notation_duration_score_detail = '音符节奏评分项总分为{}，每个音高的分值为{}，下列音高可能存在失分情况：'.format(score_seted, each_score) + '\n'

    pitch_total_score = 0
    pitch_score_detail = '音高评分项总分为{}，每个音高的分值为{}，下列音符可能存在失分情况：'.format(score_seted,each_score) + '\n'

    standard_notations = standard_notations.split(',')
    standard_notations = [s[0] for s in standard_notations]

    for i,sn in enumerate(standard_notations):
        range_start = standard_notation_time[i]
        range_end = standard_notation_time[i+1]
        range_duration = range_end - range_start
        notations_in_range = get_pitch_info_in_time_range(range_start,range_end,numbered_notations,test_times,end_time)
        if len(notations_in_range) == 0:
            str_detail = "第{}个音符:{}，开始于{}，结束于{}，连续时长为{}，未有匹配的音符，扣{}分".format(i+1, sn,range_start,range_end,range_duration,each_score)
            notation_duration_score_detail += str_detail + '\n'

            str_detail = "第{}个音高:{}，开始于{}，结束于{}，连续时长为{}，未有匹配的音高，扣{}分".format(i + 1, sn,range_start,range_end,range_duration,each_score)
            pitch_score_detail += str_detail + '\n'
        else:
            total_duration = 0
            for nir in notations_in_range:
                notation = nir[0]
                duration = nir[1]
                if np.abs(int(notation) - int(sn)) <=1:
                    total_duration += duration
            if total_duration > 0:
                pitch_total_score += each_score
                offset = np.abs((total_duration - range_duration)/range_duration)
                if offset < 0.5: #能匹配上的音符累积时长偏差小于0.5
                    notation_duration_total_score += each_score
                else:
                    str_detail = "第{}个音符:{}，开始于{}，结束于{}，连续时长为{}，实际时长为{}，偏差率为{}，扣{}分".format(i + 1, sn,range_start,range_end,range_duration,total_duration,offset, each_score)
                    notation_duration_score_detail += str_detail + '\n'
            else:
                str_detail = "第{}个音符:{}，开始于{}，结束于{}，连续时长为{}，该时间段未有匹配的音符，扣{}分".format(i + 1, sn,range_start,range_end,range_duration,each_score)
                notation_duration_score_detail += str_detail + '\n'

                str_detail = "第{}个音高:{}，开始于{}，结束于{}，连续时长为{}，该时间段未有匹配的音高，扣{}分".format(i + 1, sn,range_start,range_end,range_duration,each_score)
                pitch_score_detail += str_detail + '\n'

    notation_duration_total_score = round(notation_duration_total_score, 2)
    pitch_total_score = round(pitch_total_score, 2)
    if notation_duration_total_score == score_seted:
        notation_duration_score_detail = "音符节奏评分项总分为{}，每个音高的分值为{},未存在失分的情况".format(score_seted, each_score)
    if pitch_total_score == score_seted:
        pitch_score_detail = "音高评分项总分为{}，每个音高的分值为{},未存在失分的情况".format(score_seted, each_score)
    return notation_duration_total_score,notation_duration_score_detail,pitch_total_score,pitch_score_detail

'''
以标准音符时间点来计算歌词节奏得分和音高得分
1、若该时间段内正确歌词连续时长占比大于50%，则该歌词节奏可得分；
standard_kc = '喜爱春天的人儿是心地纯洁的人像紫罗兰花儿一样是我知心朋友'
standard_kc_time = [0, 1, 2, 3, 3.5, 4, 5, 6, 8, 9, 10, 11, 11.5, 12, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 26.5, 27, 28, 32]
kc_detail = {20: ('惜', 0), 144: ('爱', 1), 236: ('春天', 2), 392: ('的', 4), 440: ('人', 5), 552: ('儿', 6),
             640: ('时', 7), 824: ('心地', 8), 1040: ('纯洁', 10), 1188: ('的', 12), 1624: ('相思', 13), 1832: ('罗', 15),
             1936: ('兰花', 16), 2100: ('花儿', 18), 2240: ('一样', 20), 2425: ('是', 22), 2545: ('我', 23),
             2645: ('知心', 24), 2745: ('朋友', 26)}
'''
def kc_duration_score_by_st(standard_kc,standard_kc_time,kc_detail,end_time,score_seted):
    # end_time = test_times[-1]
    each_score = round(score_seted / len(standard_kc), 2)

    kc_duration_total_score = 0
    kc_duration_score_detail = '歌词节奏评分项总分为{}，每个歌词的分值为{}，下列歌词可能存在失分情况：'.format(score_seted, each_score) + '\n'

    for i,sn in enumerate(standard_kc):
        try:
            range_start = standard_kc_time[i]
            range_end = standard_kc_time[i+1]
            range_duration = range_end - range_start

            #获取某个时段的歌词信息
            notations_in_range = get_kc_info_in_time_range(range_start,range_end,kc_detail,end_time)
            if len(notations_in_range) == 0:
                str_detail = "第{}个歌词:{}，开始于{}，结束于{}，连续时长为{}，未有匹配的歌词，扣{}分".format(i+1, sn,range_start,range_end,range_duration,each_score)
                kc_duration_score_detail += str_detail + '\n'

            else:
                total_duration = 0
                for nir in notations_in_range:
                    test_str = nir[0]
                    duration = nir[1]
                    if check_tyz(sn,test_str):
                        total_duration += duration
                if total_duration > 0:
                    offset = np.abs((total_duration - range_duration)/range_duration)
                    if offset < 0.5: #能匹配上的音符累积时长偏差小于0.5
                        kc_duration_total_score += each_score
                    else:
                        str_detail = "第{}个歌词:{}，开始于{}，结束于{}，连续时长为{}，未有匹配的歌词，扣{}分".format(i+1, sn,range_start,range_end,range_duration,each_score)
                        kc_duration_score_detail += str_detail + '\n'
                else:
                    str_detail = "第{}个歌词:{}，开始于{}，结束于{}，连续时长为{}，未有匹配的歌词，扣{}分".format(i+1, sn,range_start,range_end,range_duration,each_score)
                    kc_duration_score_detail += str_detail + '\n'
        except Exception:
            print("{},{}".format(i,sn))
            pass
    kc_duration_total_score = round(kc_duration_total_score, 2)
    if kc_duration_total_score == score_seted:
        kc_duration_score_detail = "未存在失分的情况"

    return kc_duration_total_score,kc_duration_score_detail



'''
获取某个时段的音高信息
'''
def get_pitch_info_in_time_range(range_start,range_end,numbered_notations,test_times,end_time):
    test_times_with_end_time = test_times.copy()
    test_times_with_end_time.append(end_time)
    result = []
    for i,nn in enumerate(numbered_notations):
        if nn is not None:
            duration = 0
            nn_start_time = test_times_with_end_time[i]
            nn_end_time = test_times_with_end_time[i+1]
            if nn_start_time >= range_start and nn_start_time < range_end:  # 如果该音高的头部落在区间内
                duration = nn_end_time - nn_start_time if nn_end_time <= range_end else range_end - nn_start_time
            elif nn_end_time > range_start and nn_end_time <= range_end:    # 如果该音高的尾巴落在区间内
                duration = nn_end_time - range_start if nn_start_time <= range_start else nn_end_time - nn_start_time
            elif nn_start_time <= range_start and nn_end_time >= range_end:     #覆盖整个区间的
                duration = range_end - range_start
            if duration > 0:
                duration = round(duration,2)
                tmp = (nn,duration)
                result.append(tmp)
    return result

'''
获取某个时段的歌词信息
'''
def get_kc_info_in_time_range(range_start,range_end,kc_detail,end_time):
    test_times = [round((value) / 100, 2) for value in kc_detail.keys() if value > 0]
    test_times_with_end_time = test_times.copy()
    test_times_with_end_time.append(end_time)
    result = []
    keys = list(kc_detail.keys())
    keys = [round(k/100,2) for k in keys]
    keys.append(end_time)
    index = 0
    for k,v in kc_detail.items():
        if v[1] != "":
            duration = 0
            kc_start_time = keys[index]
            kc_end_time = keys[index+1]
            if kc_start_time >= range_start and kc_start_time < range_end:  # 如果该音高的头部落在区间内
                duration = kc_end_time - kc_start_time if kc_end_time <= range_end else range_end - kc_start_time
            elif kc_end_time > range_start and kc_end_time <= range_end:    # 如果该音高的尾巴落在区间内
                duration = kc_end_time - range_start if kc_start_time <= range_start else kc_end_time - kc_start_time

            if duration > 0:
                duration = round(duration,2)
                tmp = (v[0],duration)
                result.append(tmp)
        index += 1
    return result

if __name__ == "__main__":
    standard_kc = '喜爱春天的人儿是心地纯洁的人像紫罗兰花儿一样是我知心朋友'
    test_kc = '惜爱春天的人儿时心地纯洁的相思罗兰花花儿一样是我知心朋友'
    test_kc = '喜爱春天的人儿是心的一纯像紫罗兰花儿一样是我知心朋友'
    lcseque, standard_positions,test_positions = get_lcseque_and_position(standard_kc,test_kc)
    print("lcseque is {}, size is {}".format(lcseque,len(lcseque)))
    print("standard_positions is {}, size is {}".format(standard_positions, len(standard_positions)))
    print("test_positions is {}, size is {}".format(test_positions, len(test_positions)))

    standard_kc_time = [0, 1, 2, 3, 3.5, 4, 5, 6, 8, 9, 10, 11, 11.5, 12, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                        26.5, 27, 28,32]
    kc_detail = {20: ('惜', 0), 144: ('爱', 1), 236: ('春天', 2), 392: ('的', 4), 440: ('人', 5), 552: ('儿', 6),
                 640: ('时', 7), 824: ('心地', 8), 1040: ('纯洁', 10), 1188: ('的', 12), 1624: ('相思', 13), 1832: ('罗', 15),
                 1936: ('兰花', 16), 2100: ('花儿', 18), 2240: ('一样', 20), 2425: ('是', 22), 2545: ('我', 23),
                 2645: ('知心', 24), 2745: ('朋友', 26)}


    numbered_notations = [None, '3', '2', '2', '1', '1', '7', '6', '6', '6', None, '4', '4', '4', '3', '2', '1', '2',
                           '4', '3', None, '3', '4', '4', '3', '2', '4', '3', '1', '6', '7', '7', '3', '2', '1', '1',
                           '6']
    numbered_notations = [n for n in numbered_notations if n is not None]
    numbered_notations = ''.join(numbered_notations)
    print("numbered_notations is {},size is {}".format(numbered_notations,len(numbered_notations)))
    standard_notations = '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,6,7-,3,2,1,7-,1,6-'
    standard_notations = standard_notations.split(',')
    standard_notations = [s[0] for s in standard_notations]
    standard_notations= ''.join(standard_notations)
    print("standard_notations is {},size is {}".format(standard_notations,len(standard_notations)))
    loss_positions, loss_notations_in_standard = get_lossed_standard_notations(standard_notations, numbered_notations)
    print("lcseque is {}, size is {}".format(lcseque, len(lcseque)))
    print("standard_positions is {}, size is {}".format(standard_positions, len(standard_positions)))
    print("test_positions is {}, size is {}".format(test_positions, len(test_positions)))
    print("loss_positions is {}, size is {}".format(loss_positions, len(loss_positions)))
    print("loss_notations_in_standard is {}, size is {}".format(loss_notations_in_standard, len(loss_notations_in_standard)))

    score_seted = 35
    standard_notations = '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,6,7-,3,2,1,7-,1,6-'
    numbered_notations = [None, '3', '2', '2', '1', '1', '7', '6', '6', '6', None, '4', '4', '4', '3', '2', '1', '2', '4', '3', None, '3', '4', '4', '3', '2', '4', '3', '1', '6', '7', '7', '3', '2', '1', '1', '6']
    pitch_total_score, pitch_score_detail,real_loss_positions = pitch_score(standard_notations, numbered_notations, score_seted)
    print("pitch_total_score is {}".format(pitch_total_score))
    print("pitch_score_detail is {}".format(pitch_score_detail))
    print("real_loss_positions is {}".format(real_loss_positions))

    print("====================================================================================================")
    standard_notations = '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
    numbered_notations = [None, '3', '3', '2', '1', '1', '7', '6', '6', '5', '6', None, '4', '4', '3', '2', '1', '2', '4', '3', None, '4', '4', '3', '2', '2', '4', '3', '3', '7', '5', '6', None, '7', '1', '3', '2', '1', '7', '1', '6', '6']
    standard_notation_time = [0,1,1.5,2,3,3.5,4,5,6,8,9,9.5,10,10.5,11,11.5,12,16,17,17.5,18,19,19.5,20,21,21.5,22,23,24,25,26,26.5,27,27.5,28,32]
    test_times = [0.2, 0.47, 1.44, 1.97, 2.36, 3.48, 3.92, 4.4, 5.52, 6.4, 6.61, 8.24, 8.56, 9.55, 10.1, 10.4, 11.05, 11.52, 11.88, 12.66, 16.24, 16.45, 17.49, 17.96, 18.32, 19.36, 19.96, 20.41, 21.0, 22.04, 22.4, 22.8, 24.25, 24.57, 25.45, 25.66, 26.45, 26.97, 27.45, 27.93, 28.47, 29.84]
    end_time = 32
    score_seted = 35
    notation_duration_total_score, notation_duration_score_detail = notation_duration_score(standard_notations, standard_notation_time, numbered_notations, test_times,end_time, score_seted)
    print("notation_duration_total_score is {}".format(notation_duration_total_score))
    print("notation_duration_score_detail is {}".format(notation_duration_score_detail))

    # standard_kc = '喜爱春天的人儿是心地纯洁的人像紫罗兰花儿一样是我知心朋友'
    # standard_kc_time = [0, 1, 2, 3, 3.5, 4, 5, 6, 8, 9, 10, 11, 11.5, 12, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,26.5, 27, 28, 32]
    # standard_notations = '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
    # standard_notation_time = [0, 1, 1.5, 2, 3, 3.5, 4, 5, 6, 8, 9, 9.5, 10, 10.5, 11, 11.5, 12, 16, 17, 17.5, 18, 19, 19.5, 20, 21, 21.5, 22, 23, 24, 25, 26, 26.5, 27, 27.5, 28, 32]
    # kc_with_notations = get_notations_on_kc(standard_kc, standard_kc_time, standard_notations, standard_notation_time)
    # print("kc_with_notations is {}".format(kc_with_notations))

    standard_kc = '喜爱春天的人儿是心地纯洁的人像紫罗兰花儿一样是我知心朋友'
    test_kc = '惜爱春天的人儿时心地纯洁的相思罗兰花花儿一样是我知心朋友'
    # test_kc = '喜爱春天的人儿时心地纯洁的相思罗兰花花儿一样是我知心朋友'


    print("====================================================================================================")

    standard_kc_time = [0, 1, 2, 3, 3.5, 4, 5, 6, 8, 9, 10, 11, 11.5, 12, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                        26.5, 27, 28, 32]
    kc_detail = {20: ('惜', 0), 144: ('爱', 1), 236: ('春天', 2), 392: ('的', 4), 440: ('人', 5), 552: ('儿', 6),
                 640: ('时', 7), 824: ('心地', 8), 1040: ('纯洁', 10), 1188: ('的', 12), 1624: ('相思', 13), 1832: ('罗', 15),
                 1936: ('兰花', 16), 2100: ('花儿', 18), 2240: ('一样', 20), 2425: ('是', 22), 2545: ('我', 23),
                 2645: ('知心', 24), 2745: ('朋友', 26)}

    score_seted = 30
    total_score, score_detail = kc_rhythm_score(standard_kc, standard_kc_time, kc_detail, test_kc, real_loss_positions,score_seted)
    print("total_score is {}".format(total_score))
    print("score_detail is {}".format(score_detail))

    print("=================================000000000000000000000000000000000000000000============================")
    standard_kc = '喜爱春天的人儿是心地纯洁的人像紫罗兰花儿一样是我知心朋友'
    test_kc = '惜爱春天的人儿时心地纯洁的相思罗兰花花儿一样是我知心朋友'
    standard_kc_time = [0, 1, 2, 3, 3.5, 4, 5, 6, 8, 9, 10, 11, 11.5, 12, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 26.5, 27, 28, 32]
    kc_detail = {20: ('惜', 0), 144: ('爱', 1), 236: ('春天', 2), 392: ('的', 4), 440: ('人', 5), 552: ('儿', 6),
                 640: ('时', 7), 824: ('心地', 8), 1040: ('纯洁', 10), 1188: ('的', 12), 1624: ('相思', 13), 1832: ('罗', 15),
                 1936: ('兰花', 16), 2100: ('花儿', 18), 2240: ('一样', 20), 2425: ('是', 22), 2545: ('我', 23),
                 2645: ('知心', 24), 2745: ('朋友', 26)}
    standard_notations = '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,7-,3,2,1,7-,1,6-'
    numbered_notations = [None, '3', '3', '2', '1', '1', '7', '6', '6', '5', '6', None, '4', '4', '3', '2', '1', '2', '4', '3', None, '4', '4', '3', '2', '2', '4', '3', '3', '7', '5', '6', None, '7', '1', '3', '2', '1', '7', '1', '6', '6']
    standard_notation_time = [0,1,1.5,2,3,3.5,4,5,6,8,9,9.5,10,10.5,11,11.5,12,16,17,17.5,18,19,19.5,20,21,21.5,22,23,24,25,26,26.5,27,27.5,28,32]
    test_times = [0.2, 0.47, 1.44, 1.97, 2.36, 3.48, 3.92, 4.4, 5.52, 6.4, 6.61, 8.24, 8.56, 9.55, 10.1, 10.4, 11.05, 11.52, 11.88, 12.66, 16.24, 16.45, 17.49, 17.96, 18.32, 19.36, 19.96, 20.41, 21.0, 22.04, 22.4, 22.8, 24.25, 24.57, 25.45, 25.66, 26.45, 26.97, 27.45, 27.93, 28.47, 29.84]
    end_time = 32
    get_all_scores(standard_kc,standard_kc_time, test_kc, standard_notations, numbered_notations, standard_notation_time, test_times, kc_detail, end_time)

    print("=================================000000000000000000000000000000000000000000============================")
    notation_duration_total_score, notation_duration_score_detail, pitch_total_score, pitch_score_detail = notation_duration_and_pitch_score_by_st(standard_notations,standard_notation_time,numbered_notations,test_times,end_time,score_seted)
    print("notation_duration_total_score is {}".format(notation_duration_total_score))
    print("notation_duration_score_detail is {}".format(notation_duration_score_detail))
    print("pitch_total_score is {}".format(pitch_total_score))
    print("pitch_score_detail is {}".format(pitch_score_detail))

    print("=================================000000000000000000000000000000000000000000============================")
    kc_duration_total_score, kc_duration_score_detail = kc_duration_score_by_st(standard_kc,standard_kc_time,kc_detail,end_time,score_seted)
    print("kc_duration_total_score is {}".format(kc_duration_total_score))
    print("kc_duration_score_detail is {}".format(kc_duration_score_detail))

    print("=================================00000000000000000=====================00000000000============================")
    get_all_scores_by_st(standard_kc, standard_kc_time, standard_notations, numbered_notations, standard_notation_time,
                         test_times, kc_detail, end_time)