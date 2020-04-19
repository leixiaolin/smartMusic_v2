# -*- coding:utf-8 -*-
import numpy as np
from LscHelper import find_lcseque,my_find_lcseque
from pinyin_util import modify_tyz_by_position

'''
音高节奏评分
pitch_time = [0,1, 1.5, 2, 3, 3.5, 4, 5, 6, 8, 9, 9.5, 10, 10.5, 11, 11.5, 12, 16, 17, 17.5, 18, 19, 19.5, 20, 21,
               21.5, 22, 23, 24, 25, 26, 26.5, 27,27.5, 28,32]
'''
def notation_rhythm_score(pitch_time,):
    pass


'''
歌词节奏评分
standard_kc = '喜爱春天的人儿是心地纯洁的人像紫罗兰花儿一样是我知心朋友加'
standard_kc_time = [0,1, 2, 3, 3.5, 4, 5, 6, 8, 9, 10, 11, 11.5, 12, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 26.5, 27, 28,30,32]
kc_detail = {20: ('惜', 0), 144: ('爱', 1), 236: ('春天', 2), 392: ('的', 4), 440: ('人', 5), 552: ('儿', 6), 640: ('时', 7), 824: ('心地', 8), 1040: ('纯洁', 10), 1188: ('的', 12), 1624: ('相思', 13), 1832: ('罗', 15), 1936: ('兰花', 16), 2100: ('花儿', 18), 2240: ('一样', 20), 2425: ('是', 22), 2545: ('我', 23), 2645: ('知心', 24), 2745: ('朋友', 26), 0: ('', 28)}
all_test_kc = '喜爱春天的人儿是心地纯洁的像紫罗兰花花儿一样是我知心朋友'
'''
def kc_rhythm_score(standard_kc,standard_kc_time,kc_detail,test_kc,score_seted):
    #待测歌词的时间点
    detail_time = [round((value) / 100, 2) for value in kc_detail.keys() if value > 0]
    detail_time.append(standard_kc_time[-1]) #从标准时间序列中添加结束点
    detail_time_diff = np.diff(detail_time)
    detail_time_diff = list(detail_time_diff)
    print("detail_time_diff is {}, size is {}".format(detail_time_diff, len(detail_time_diff)))
    standard_kc_time_diff = np.diff(standard_kc_time)
    standard_kc_time_diff = list(standard_kc_time_diff)
    print("standard_kc_time_diff is {}, size is {}".format(standard_kc_time_diff, len(standard_kc_time_diff)))
    each_score = round(score_seted/len(standard_kc),2)
    total_score = 0

    modify_test_kc = modify_tyz_by_position(standard_kc, test_kc)
    # 获取最大公共序列及其相关位置信息
    lcseque, standard_positions, test_positions = get_lcseque_and_position(standard_kc, modify_test_kc)

    score_detail = '评测细节：' + '\n'
    for i,tup in enumerate(kc_detail.values()):
        try:
            ks = list(tup[0])
            p = tup[1]
            ps = [p]
            if len(ks) > 1: #多个歌词在一起的情况
                for k in range(1,len(ks)):
                    ps.append(p+k) # 获取每个歌词在待测歌词序列中的位置

            test_duration = detail_time_diff[i] #计算该歌词的时长
            test_duration = round(test_duration,2)

            ps = [p for p in ps if p in test_positions] #筛选出在最大公共序列出现过的歌词的位置，如果是多唱的歌词，ps会为空
            if len(ps) == 0:
                str_detail = "{}: duration is {}, standard_duration is {} ,this is uncalled-for".format(tup[0],test_duration)
                # print(str_detail)
                score_detail += str_detail +'\n'
                continue
            tmp = [i for i,p in enumerate(test_positions) if p in ps] # 获取歌词位置的下标
            standard_ps = [standard_positions[i] for i in tmp]
            standard_duration = np.sum([standard_kc_time_diff[i] for i in standard_ps]) #计算该歌词对应的标准时长
            standard_duration = round(standard_duration, 2)

            offset_duration = round(np.abs(test_duration - standard_duration),2)
            if offset_duration <= standard_duration * 0.25: #如果偏差小于25%，即可得分
                score = round(each_score * len(ps),2)
                total_score += score
                str_detail = "{}: duration is {}, standard_duration is {} ,offset_duration is {},score is {}".format(tup[0],test_duration,standard_duration, offset_duration,score)
            else:
                str_detail ="{}: duration is {}, standard_duration is {} ,offset_duration is {},score is {}".format(tup[0],test_duration,standard_duration, offset_duration, 0)
            score_detail += str_detail + '\n'
            # print(str_detail)
        except Exception:
            print(tup[0] + "is error")

    return total_score,score_detail

'''
音高评分算法
standard_notations = '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,6,7-,3,2,1,7-,1,6-'
numbered_notations = [None, '3', '2', '2', '1', '1', '7', '6', '6', '6', None, '4', '4', '4', '3', '2', '1', '2', '4', '3', None, '3', '4', '4', '3', '2', '4', '3', '1', '6', '7', '7', '3', '2', '1', '1', '6']
'''
def pitch_score(standard_notations,numbered_notations,score_seted):
    each_score = round(score_seted / len(standard_notations.split(',')), 2)
    total_score = score_seted
    score_detail = '评测细节：下列音高存在失分情况' + '\n'

    standard_notations = standard_notations.split(',')
    standard_notations = [s[0] for s in standard_notations]
    standard_notations = ''.join(standard_notations)

    numbered_notations = [n for n in numbered_notations if n is not None]
    numbered_notations = ''.join(numbered_notations)
    print("standard_notations is {},size is {}".format(standard_notations, len(standard_notations)))
    print("numbered_notations is {},size is {}".format(numbered_notations, len(numbered_notations)))

    #找出未匹配的音高，并对未匹配的每个音高进行分析
    lcseque, standard_positions, test_positions = get_lcseque_and_position(standard_notations, numbered_notations)
    loss_positions, loss_notations_in_standard = get_lossed_standard_notations(standard_notations, numbered_notations)
    for j,lp in enumerate(loss_positions):
        loss_notation = 0
        try:
            loss_notation = int(loss_notations_in_standard[j][0])
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
                else:
                    str_detail = "第{}个音高:{} 未匹配，扣{}分".format(lp, loss_notation,each_score)
                    total_score -= each_score
                    score_detail += str_detail + '\n'
            else: # 如果前后准点不相临
                # 获取前后准点之间的音高
                tmp = [int(a) for i,a in enumerate(numbered_notations) if i > anchor_before_in_test and i < anchor_after_in_test]
                offset = [np.abs(loss_notation - t) for t in tmp] # 音高差值
                if np.min(offset) <= 1: #如果音高差值不超过1，可计半分
                    total_score -= round(each_score * 0.5, 2)
                    str_detail = "第{}个音高:{} 与标准差值较小，扣{}分".format(lp, loss_notation,round(each_score * 0.5, 2))
                else:
                    str_detail = "第{}个音高:{} 与标准差值较大，扣{}分".format(lp, loss_notation,each_score)
                    total_score -= each_score
                score_detail += str_detail + '\n'
        except Exception:
            str_detail = "第{}个音高:{}  is error，扣{}分".format(lp,loss_notation,each_score)
            total_score -= each_score
            score_detail += str_detail + '\n'
        # print(str_detail)
    if total_score == score_seted:
        score_detail = "未存在失分的情况"
    return round(total_score,2), score_detail

'''
音符节奏评分算法
standard_notations = '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,3,1,6-,6-,6,7-,3,2,1,7-,1,6-'
standard_notation_time = [0,1,1.5,2,3,3.5,4,5,6,8,9,9.5,10,10.5,11,11.5,12,16,17,17.5,18,19,19.5,20,21,21.5,22,23,24,25,26,26.5,27,27.5,28,32]
numbered_notations = [None, '3', '2', '2', '1', '1', '7', '6', '6', '6', None, '4', '4', '4', '3', '2', '1', '2', '4', '3', None, '3', '4', '4', '3', '2', '4', '3', '1', '6', '7', '7', '3', '2', '1', '1', '6']
test_times is [0.29, 0.75, 2.02, 2.41, 3.45, 3.93, 4.37, 5.53, 6.38, 6.59, 7.8, 8.37, 8.61, 9.5, 9.86, 10.13, 10.45, 10.99, 11.51, 11.97, 12.25, 12.67, 13.49, 16.25, 16.49, 17.41, 17.85, 18.33, 19.15, 19.41, 19.81, 20.01, 20.29, 20.56, 21.35, 21.87, 22.37, 23.85, 24.27, 24.54, 24.82, 25.47, 25.84, 26.43, 27.11, 27.43, 28.0, 28.48, 30.38,32]
'''
def notation_duration_score(standard_notations,standard_notation_time,numbered_notations,test_times,end_time,score_seted):
    # end_time = test_times[-1]
    each_score = round(score_seted / len(standard_notations.split(',')), 2)
    total_score = 0
    score_detail = '评测细节：下列音高存在失分情况' + '\n'

    standard_notations = standard_notations.split(',')
    standard_notations = [s[0] for s in standard_notations]
    standard_notations = ''.join(standard_notations)

    test_times = [test_times[i] for i,n in enumerate(numbered_notations) if n is not None]
    test_times.append(end_time)
    test_times_diff = np.diff(test_times)
    numbered_notations = [n for n in numbered_notations if n is not None]
    numbered_notations = ''.join(numbered_notations)
    numbered_notations_list = list(numbered_notations)
    print("standard_notations is {},size is {}".format(standard_notations, len(standard_notations)))
    print("numbered_notations is {},size is {}".format(numbered_notations, len(numbered_notations)))

    standard_notation_time_diff = np.diff(standard_notation_time)

    # 找出最大公共子序列，并对每个匹配上的音符时长进行判断计分处理
    lcseque, standard_positions, test_positions = get_lcseque_and_position(standard_notations, numbered_notations)
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

            if offset <= 0.25: # 时长偏差小于0.25，该音符可得满分
                total_score += each_score
            else:
                str_detail = "第{}个音高:{} 时长为{}，标准时长为{}，与标准差值为{}，大于规定范围，扣{}分".format(standard_position, s,test_duration,standard_duration, offset,each_score)
                score_detail += str_detail + '\n'

    #找出未匹配的音高，并对未匹配的每个音高进行分析
    # loss_positions, loss_notations_in_standard = get_lossed_standard_notations(standard_notations, numbered_notations)
    # for j,lp in enumerate(loss_positions):
    #     loss_notation = 0
    #     try:
    #         loss_notation = int(loss_notations_in_standard[j][0])
    #         # 该未匹配音高的前一个准点
    #         anchor_before = int(lp) - 1
    #         anchor_before_in_test = [test_positions[i] for i, a in enumerate(standard_positions) if a <= anchor_before][-1]
    #         # 该未匹配音高的后一个准点
    #         anchor_after = int(lp) + 1
    #         anchor_after_in_test = [test_positions[i] for i,a in enumerate(standard_positions) if a >= anchor_after][0]
    #         if anchor_before_in_test + 1 == anchor_after_in_test: # 如果前后准点相临
    #             # 根据当前音高是否等于前一个准点的音高，如果等于即是存在连音的情况，可计分
    #             if loss_notation == int(numbered_notations[anchor_before_in_test]):
    #                 # total_score += each_score
    #                 pass
    #             else:
    #                 str_detail = "第{}个音高:{} 未匹配，扣{}分".format(lp, loss_notation,each_score)
    #                 total_score -= each_score
    #                 score_detail += str_detail + '\n'
    #         else: # 如果前后准点不相临
    #             # 获取前后准点之间的音高
    #             tmp = [int(a) for i,a in enumerate(numbered_notations) if i > anchor_before_in_test and i < anchor_after_in_test]
    #             offset = [np.abs(loss_notation - t) for t in tmp] # 音高差值
    #             if np.min(offset) <= 1: #如果音高差值不超过1，可计半分
    #                 total_score -= round(each_score * 0.5, 2)
    #                 str_detail = "第{}个音高:{} 与标准差值较小，扣{}分".format(lp, loss_notation,round(each_score * 0.5, 2))
    #             else:
    #                 str_detail = "第{}个音高:{} 与标准差值较大，扣{}分".format(lp, loss_notation,each_score)
    #                 total_score -= each_score
    #             score_detail += str_detail + '\n'
    #     except Exception:
    #         str_detail = "第{}个音高:{}  is error，扣{}分".format(lp,loss_notation,each_score)
    #         total_score -= each_score
    #         score_detail += str_detail + '\n'
        # print(str_detail)
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
    print("lcseque is {}, size is {}".format(lcseque, len(lcseque)))
    print("standard_positions is {}, size is {}".format(standard_positions, len(standard_positions)))
    print("test_positions is {}, size is {}".format(test_positions, len(test_positions)))
    print("loss_positions is {}, size is {}".format(loss_positions, len(loss_positions)))
    print("loss_notations_in_standard is {}, size is {}".format(loss_notations_in_standard, len(loss_notations_in_standard)))
    return loss_positions,loss_notations_in_standard

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

    score_seted = 30
    total_score,score_detail = kc_rhythm_score(standard_kc, standard_kc_time, kc_detail, test_kc, score_seted)
    print("total_score is {}".format(total_score))
    print("score_detail is {}".format(score_detail))

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

    score_seted = 30
    standard_notations = '3,3,2,1,1,7-,6-,6-,6-,4,4,3,2,1,2,4,3,4,4,3,2,2,4,3,1,3,4,5,6,6,1+,7,6,1+'
    numbered_notations = [None, '3', '2', '2', '1', '1', '7', '6', '6', '6', None, '4', '4', '4', '3', '2', '1', '2', '4', '3', None, '3', '4', '4', '3', '2', '4', '3', '1', '6', '7', '7', '3', '2', '1', '1', '6']
    # pitch_total_score, pitch_score_detail = pitch_score(standard_notations, numbered_notations, score_seted)
    # print("pitch_total_score is {}".format(pitch_total_score))
    # print("pitch_score_detail is {}".format(pitch_score_detail))


    numbered_notations = [None, '3', '3', '2', '1', '1', '7', '6', '6', '5', '6', None, '4', '4', '3', '2', '1', '2', '4', '3', None, '4', '4', '3', '2', '2', '4', '3', '3', '7', '5', '6', None, '7', '1', '3', '2', '1', '7', '1', '6', '6']
    standard_notation_time = [0,1,1.5,2,3,3.5,4,5,6,8,9,9.5,10,10.5,11,11.5,12,16,17,17.5,18,19,19.5,20,21,21.5,22,23,24,25,26,26.5,27,27.5,28,32]
    test_times = [0.2, 0.47, 1.44, 1.97, 2.36, 3.48, 3.92, 4.4, 5.52, 6.4, 6.61, 8.24, 8.56, 9.55, 10.1, 10.4, 11.05, 11.52, 11.88, 12.66, 16.24, 16.45, 17.49, 17.96, 18.32, 19.36, 19.96, 20.41, 21.0, 22.04, 22.4, 22.8, 24.25, 24.57, 25.45, 25.66, 26.45, 26.97, 27.45, 27.93, 28.47, 29.84]
    end_time = 32
    notation_duration_total_score, notation_duration_score_detail = notation_duration_score(standard_notations, standard_notation_time, numbered_notations, test_times,end_time, score_seted)
    print("notation_duration_total_score is {}".format(notation_duration_total_score))
    print("notation_duration_score_detail is {}".format(notation_duration_score_detail))