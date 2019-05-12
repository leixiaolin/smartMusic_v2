import matplotlib.pyplot as plt
import re
import shutil
import librosa.display
import matplotlib.pyplot as plt
from create_base import *
from create_labels_files import *
from find_mismatch import *
from grade import *

score = 0

date = '3.19'
src_path = 'F:/项目/花城音乐项目/样式数据/3.19MP3/节奏/'
new_old_txt = './onsets/'+date+'best_dtw.txt'


# 保存新文件名与原始文件的对应关系
files_list = []
files_list_a = []
files_list_b = []
files_list_c = []
files_list_d = []
# new_old_txt = './onsets/best_dtw.txt'
files = list_all_files(src_path)

print(files)
index = 0
# 用于分析人工打分与算法打分误差


# 测试单个文件
#files = ['F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏六（5）（80）.wav']
files = ['F:/项目/花城音乐项目/样式数据/2.27MP3/节奏/节奏8_40210（30）.wav']
for filename in files:
    print(filename)
    if filename.find('wav') <= 0:
        continue
    elif filename.find('shift') > 0:
        continue
    score, lost_score, ex_score, min_d = get_score_jz(filename)
    print("score, lost_score, ex_score, min_d is {},{},{},{}".format(score, lost_score, ex_score, min_d))

    if int(score) >=90:
        grade = 'A'
        files_list_a.append([filename + ' - ' + grade, score, lost_score, ex_score, min_d])
    elif int(score) >= 75:
        grade = 'B'
        files_list_b.append([filename + ' - ' + grade, score, lost_score, ex_score, min_d])
    elif int(score) >=60:
        grade = 'C'
        files_list_c.append([filename + ' - ' + grade, score, lost_score, ex_score, min_d])
    elif int(score) >=1:
        grade = 'D'
        files_list_d.append([filename + ' - ' + grade, score, lost_score, ex_score, min_d])
    else:
        grade = 'E'

t1 = np.append(files_list_a,files_list_b).reshape(len(files_list_a)+len(files_list_b),5)
t2 = np.append(files_list_c,files_list_d).reshape(len(files_list_c)+len(files_list_d),5)
files_list = np.append(t1,t2).reshape(len(t1)+len(t2),5)

write_txt(files_list, new_old_txt, mode='w')





