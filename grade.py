
# 计算分数方法1
def get_score1(standard_y,recognize_y,onsets_total,onsets_strength,min_d):
    '''
    计算多唱扣分，漏唱扣分
    standard_y：标准帧
    recognize_y：识别帧
    onsets_total：总节拍数
    onsets_strength：节拍强度
    min_d：偏移分值
    '''
    from find_mismatch import get_mismatch_line,get_wrong
    score = 100
    standard_y, recognize_y = get_mismatch_line(standard_y, recognize_y)
    lost_num, ex_frames = get_wrong(standard_y, recognize_y)
    # print(standard_y,recognize_y)
    lost_score = 0
    ex_score = 0
    if lost_num:
        print('漏唱了' + str(lost_num) + '句')
        lost_score = 100 / onsets_total * lost_num
    elif len(ex_frames) >= 1:
        for x in ex_frames:
            strength = onsets_strength[int(x)]
            ex_score += int(100 / onsets_total * strength)
    else:
        print('节拍数一致')
    # 计算分数
    score = score-lost_score-ex_score-min_d
    if score <0:
        score = 0
    return score


    # # onsets_num:总节拍数量 lost_num:漏唱数量 ex_frames:多唱的帧 dict:识别节拍与所对应的强度 shift:偏移分值
    # score = 100
    # # 每节拍分值
    # score_per_onset = 100 / onsets_num
    # print('score_per_onset:{}'.format(score_per_onset))
    # score -= shift
    # ex_strength = []
    # # 漏唱：每句扣score_per_onset分
    # if lost_num:
    #     print('漏唱了'+str(lost_num)+'句')
    #     score -= score_per_onset*lost_num
    # # 多唱：每句扣 节拍强度*score_per_onset分
    #
    # elif len(ex_frames):
    #     for i in ex_frames:
    #         score -= score_per_onset * dict[i]
    #         ex_strength.append(dict[i])
    # print('多唱的帧强度：{}'.format(ex_strength))
    # return score



#def get_score2()

