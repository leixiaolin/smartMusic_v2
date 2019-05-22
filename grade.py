
'''
计算分数方法1
'''
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
        #print('漏唱了' + str(lost_num) + '句')
        if lost_num <= 3:
            lost_score = 100 / onsets_total * lost_num * 0.5
        else:
            lost_score = 100 / onsets_total * lost_num
    elif len(ex_frames) >= 1:
        for x in ex_frames:
            strength = onsets_strength[int(x)]
            ex_score += int(100 / onsets_total * strength)
    else:
        #print('节拍数一致')
        pass
    # 计算分数
    score = score-int(lost_score)-int(ex_score)-int(min_d)
    if score <0:
        score = 0
    return score,int(lost_score),int(ex_score),int(min_d)


'''
计算分数方法1
'''

def get_score_detail_for_note(standard_y, recognize_y, onsets_total, onsets_strength, min_d):
    '''
    计算多唱扣分，漏唱扣分
    standard_y：标准帧
    recognize_y：识别帧
    onsets_total：总节拍数
    onsets_strength：节拍强度
    min_d：偏移分值
    '''
    from find_mismatch import get_mismatch_line, get_wrong
    score = 100
    total_length = len(standard_y)
    standard_y, recognize_y = get_mismatch_line(standard_y, recognize_y)
    lost_num, ex_frames = get_wrong(standard_y, recognize_y)
    # print(standard_y,recognize_y)
    lost_score = 0
    ex_score = 0
    if lost_num:
        #print('漏唱了' + str(lost_num) + '句')
        if lost_num/total_length > 0.1:
            lost_score = 100 / onsets_total * lost_num * 0.5
    elif len(ex_frames) >= 0:
        for x in ex_frames:
            strength = 0.5
            ex_score += int(100 / onsets_total * strength)
    else:
        #print('节拍数一致')
        pass
    # 计算分数
    score = score - int(lost_score) - int(ex_score) - int(min_d)
    if score < 0:
        score = 0
    return score, int(lost_score), int(ex_score), int(min_d)


'''
计算分数方法2
'''
# def get_score2():


