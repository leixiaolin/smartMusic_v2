# -*- coding: UTF-8 -*-
import os
savepath = '/home/lei/bot-rating/split_pic/'
savepath = 'E:/t/'  # 保存要测试的目录
def get_split_pic_save_path():
    path = os.path.split(os.path.realpath(__file__))[0]
    savepath = path + '/split_pic/'
    return savepath