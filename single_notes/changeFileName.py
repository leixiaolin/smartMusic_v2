import numpy as np
import os
import shutil

'''
人工打乱样本顺序，防止样本正负扎堆
'''
tmp = ['no','yes']

level = 10000000
# dis_dir = 'F:/项目/花城音乐项目/参考代码/tensorflow_models_nets-master/onsets/train'
# scr_dir = 'F:/项目/花城音乐项目/参考代码/tensorflow_models_nets-master/onsets/val'

dis_dir = './data/train'

# dis_dir = 'F:/项目/花城音乐项目/参考代码/tensorflow_models_nets-master/rhythm/train'
# scr_dir = 'F:/项目/花城音乐项目/参考代码/tensorflow_models_nets-master/rhythm/val'
def clear_dir(sdir):
    for i in tmp:

        s_dir = sdir + '/' + i
        shutil.rmtree(s_dir)
        os.mkdir(s_dir)

for i in tmp:
    d_dir = dis_dir + '/' + i
    total = len(os.listdir(d_dir))
    if total < level:
        level = total

print(level)

# 创建临时文件
isExists = os.path.exists(dis_dir + "/tmp/")

# 判断结果
if not isExists:
    os.mkdir(dis_dir + "/tmp/")


#
for i in tmp:
    d_dir = dis_dir + '/' + i
    total = len(os.listdir(d_dir))
    files = os.listdir(d_dir)
    for j in range(total):
        shutil.move(d_dir + '/' + files[j], dis_dir + '/tmp/'+ str(j +1) + '.jpg')

    total = len(os.listdir(dis_dir + '/tmp/'))
    files = os.listdir(dis_dir + '/tmp/')
    for j in range(total):
        shutil.move(dis_dir + '/tmp/' + files[j], d_dir + '/' + files[j])