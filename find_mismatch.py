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
    else:
        while(len(standard_y)!=0 and len(recognize_y)>=2 ):
            if(abs(standard_y[0]-recognize_y[0]) <= abs(standard_y[0]-recognize_y[1])):
                recognize_y.remove(recognize_y[0])
            else:
                recognize_y.remove(recognize_y[1])

            standard_y.remove(standard_y[0])

    return standard_y,recognize_y

# 返回漏唱或多唱的情况
def get_wrong(standard_y,recognize_y):
    lost_num = len(standard_y)
    ex_frames = ['多唱的帧']
    for i in recognize_y:
        ex_frames.append(i)
    return lost_num,ex_frames



if __name__ == '__main__':
    # standard_y = ['24','52','89','123']
    # recognize_y = ['36','72','123']
    standard_y = [24,52,89,160]
    recognize_y = [50,72,123,465,789]
    standard_y,recognize_y = get_mismatch_line(standard_y,recognize_y)
    lost_num,ex_frames = get_wrong(standard_y,recognize_y)
    #print(standard_y,recognize_y)
    if lost_num:
        print('漏唱了'+str(lost_num)+'句')
    else:
        print(ex_frames)