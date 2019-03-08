# 通过帧获取该帧节拍强度
def get_strength(frame):
    strength = 0
    # 获取该帧节拍强度

    return strength

# 计算分数方法1
def calculate(onsets_num,lost_num,ex_frames,shift):
    score = 100
    # 每节拍分值
    score_per_onset = 100 / onsets_num
    # 漏唱：每句扣score_per_onset分
    if lost_num:
        print('漏唱了'+str(lost_num)+'句')
        score -= score_per_onset*lost_num
    # 多唱：每句扣 节拍强度*score_per_onset分
    elif len(ex_frames):
        for i in ex_frames:
            score -= score_per_onset * get_strength(i)
    else:
        score -= shift

