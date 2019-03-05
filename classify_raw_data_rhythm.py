
import shutil
import os
from create_base import *
from create_labels_files import write_txt

score = 0
path_index = np.array(['1.31MP3','2.2MP3','2.18MP3','2019-01-29'])

tmp = ['A','B','C','D','E']
dis_dir = 'F:/项目/花城音乐项目/参考代码/tensorflow_models_nets-master/raw_data/rhythm/'
def clear_dir(dis_dir):
    for i in tmp:
        d_dir = dis_dir + '/' + i
        shutil.rmtree(d_dir)
        os.mkdir(d_dir)

#清空文件夹
for x in tmp:
    clear_dir(dis_dir)

# 保存新文件名与原始文件的对应关系
files_list = []
new_old_txt = './onsets/new_and_old.txt'
for i in range(1,5):
    COOKED_DIR = 'F:/项目/花城音乐项目/样式数据/ALL/旋律/'+ path_index[i-1] + '/'
    #savepath = 'F:\\mfcc_pic\\'+ str(i) +'\\'
    for root, dirs, files in os.walk(COOKED_DIR):
        print("Root = ", root, "dirs = ", dirs, "files = ", files)

        base_onsets = []

        index = 0
        for filename in files:
            print(filename)
            if filename.find('wav') <= 0:
                continue
            elif filename.find('shift') > 0:
                continue
            else:
                index = index + 1


            if filename.find('标准') > 0:
                score = 100
            else:
                score = re.sub("\D", "", filename)  # 筛选数字

            if str(score).find("100") > 0:
                score = 100
            else:
                score = int(score) % 100

            if int(score) >=90:
                grade = 'A'
                savepath = dis_dir + 'A/'
            elif int(score) >= 75:
                grade = 'B'
                savepath = dis_dir + 'B/'
            elif int(score) >=60:
                grade = 'C'
                savepath = dis_dir + 'C/'
            elif int(score) >=1:
                grade = 'D'
                savepath = dis_dir + 'D/'
            else:
                grade = 'E'
                savepath = dis_dir + 'E/'

            shutil.copyfile(root + '/' + filename, savepath + '/' + filename)
