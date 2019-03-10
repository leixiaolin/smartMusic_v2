from find_mismatch import get_score
import os
'''
运行此脚本可以测试我们的方法的得分 并与人工标注进行对比
'''
test_dir = './mp3/节奏/'
file_list = os.listdir(test_dir)
score_list = []
# 要测试的数量
test_num = 3
if test_num>len(file_list):
    test_dir = len(file_list)

for i in range(0,test_num):
    print(file_list[i])
    score = get_score(test_dir+file_list[i])
    score_list.append(score)

print('-----------------------------------finish-----------------------------------')
for i in range(0,len(score_list)):
    print(file_list[i]+'的最终分数为：{}'.format(score_list[i]))
