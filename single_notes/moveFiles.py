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
scr_dir = './data/val'

# dis_dir = 'F:/项目/花城音乐项目/参考代码/tensorflow_models_nets-master/rhythm/train'
# scr_dir = 'F:/项目/花城音乐项目/参考代码/tensorflow_models_nets-master/rhythm/val'
def clear_dir(sdir):
    for i in tmp:

        s_dir = sdir + '/' + i
        shutil.rmtree(s_dir)
        os.mkdir(s_dir)

# if not os.path.exists('./data/train/no'):
#     os.mkdir('./data/train/no')
# if not os.path.exists('./data/train/yes'):
#     os.mkdir('./data/train/yes')
# if not os.path.exists('./data/val/no'):
#     os.mkdir('./data/val/no')
# if not os.path.exists('./data/val/yes'):
#     os.mkdir('./data/val/yes')

for i in tmp:
    d_dir = dis_dir + '/' + i
    s_dir = scr_dir + '/' + i
    total = len(os.listdir(d_dir))
    # if total < level:
    #     level = total
    level = total

    print(level)
    level = int(level * 0.7)
    print(level)

    # 清空文件夹
    # clear_dir(scr_dir)

    #for i in tmp:
    d_dir = dis_dir + '/' + i
    total = len(os.listdir(d_dir))
    while total > level:
        files = os.listdir(d_dir)
        index = np.random.randint(total)
        shutil.move(d_dir + '/' + files[index], s_dir + '/' + files[index])
        total = len(os.listdir(d_dir))
# print(level)
# level = int(level*0.7)
# print(level)
#
#
# # 清空文件夹
# # clear_dir(scr_dir)
#
#
#
# for i in tmp:
#     d_dir = dis_dir + '/' + i
#     s_dir = scr_dir + '/' + i
#     total = len(os.listdir(d_dir))
#     while total > level:
#         files = os.listdir(d_dir)
#         index = np.random.randint(total)
#         shutil.move(d_dir + '/' + files[index], s_dir + '/' + files[index])
#         total = len(os.listdir(d_dir))