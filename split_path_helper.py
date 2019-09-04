# -*- coding: UTF-8 -*-
import sys
savepath = '/home/lei/bot-rating/split_pic/'
savepath = 'E:/t/'  # 保存要测试的目录
def get_split_pic_save_path():
    path = sys.path[0]
    savepath = path + '/split_pic/'
    return savepath