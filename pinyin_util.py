# -*- coding:utf-8 -*-
# pip install PyHamcrest
# pip install pypinyin
from pypinyin import pinyin, lazy_pinyin, Style
from LscHelper import find_lcseque

'''
易错声母判断,包括：1、n——l；   2、f——h; 3、r——l； 4、z，c，s——zh，ch，sh，r; 5、j——q——x
'''
def check_initial(char1,char2):
    # 如果“儿”化音未识别出来,删除标准字符串的“儿”化音
    initail_1,initail_2,initail_3,initail_4,initail_5 = 'nl','fh','rl','zcszhchsh','jqx'

    if initail_1.find(char1) >= 0 and initail_1.find(char2) >= 0:
        return True
    elif initail_2.find(char1) >= 0 and initail_2.find(char2) >= 0:
        return True
    elif initail_3.find(char1) >= 0 and initail_3.find(char2) >= 0:
        return True
    elif initail_4.find(char1) >= 0 and initail_4.find(char2) >= 0:
        return True
    elif initail_5.find(char1) >= 0 and initail_5.find(char2) >= 0:
        return True
    else:
        return False

'''
获取两个中文拼音的编辑距离
'''
def get_edit_distance_from_chars(standard_char,tested_char):
    py_standard = lazy_pinyin(standard_char)
    py_standard = "".join(py_standard)
    py_tested = lazy_pinyin(tested_char)
    py_tested = "".join(py_tested)
    lcseque = find_lcseque(py_standard, py_tested)
    return len(py_standard) - len(lcseque)

'''
参照标准歌词，纠正同音字
'''
def modify_tyz(standard_str,tested_str):
    py_standard = lazy_pinyin(standard_str)
    py_tested = lazy_pinyin(tested_str)
    start = 0
    stand_len = len(py_standard)
    result = []
    for i,s in enumerate(py_tested):
        if i >= stand_len:
            result.append(tested_str[i])
        elif s == py_standard[i]:
            result.append(standard_str[i])
            start = i
        else:
            if start  == stand_len - 1:
                break
            end = start + 3 if start + 3 < stand_len else stand_len
            tmp = py_standard[start+1:end]
            for j,t in enumerate(tmp):
                if t == s: #识别结果中有漏字的情况会导致与后面的字相同
                    index = start + 1 + j
                    result.append(standard_str[index])
                    start = index
                lcseque = find_lcseque(s, t)
                if len(s) - len(lcseque) <=1 and len(t) - len(lcseque) <=1: # 编辑距离小于2的情况，则纠正
                    index = start + 1 + j
                    result.append(standard_str[index])
                    start = index
    return result

'''
参照标准歌词，纠正同音字
'''
def modify_tyz_by_position(standard_str,tested_str):
    py_standard = lazy_pinyin(standard_str)
    py_tested = lazy_pinyin(tested_str)
    stand_len = len(py_standard)
    result = []
    for i,s in enumerate(py_tested):
        flag = False
        start = i - 5 if i - 5 > 0 else 0
        end = i + 5 if i + 5 < stand_len else stand_len
        tmp = py_standard[start:end]
        tmp_len = len(tmp)
        for n in range(stand_len): # 遍历查找子列表在父列表中的位置
            if py_standard[n:n + tmp_len] == tmp:
                anchor_point = n
                break
        if s in tmp:
            index = tmp.index(s)
            result.append(standard_str[anchor_point + index])
        else:
            for j, t in enumerate(tmp):
                lcseque = find_lcseque(s, t)
                if len(s) - len(lcseque) <= 1 and len(t) - len(lcseque) <= 1:  # 编辑距离小于2的情况，则纠正
                    if t[0] != s[0] and check_initial(t[0],s[0]): # 声母不同的情况,则检查是不是常用的易错声母
                        result.append(standard_str[anchor_point + j])
                        flag = True
                    elif t[0] == s[0]: #声母相同的情况下，直接纠正
                        result.append(standard_str[anchor_point + j])
                        flag = True
            if flag is False:
                result.append(tested_str[i])
    return result

if __name__ == '__main__':
    lp = lazy_pinyin('相思儿')  # 不考虑多音字的情况
    print("".join(lp))
    lp = lazy_pinyin('像紫')  # 不考虑多音字的情况
    print("".join(lp))

    lp = lazy_pinyin('父亲')  # 不考虑多音字的情况
    print("".join(lp))
    lp = lazy_pinyin('服务器')  # 不考虑多音字的情况
    print("".join(lp))
    ss = '相思儿'
    ss = ss.replace("儿","")
    print(ss)

    standard_char, tested_char = '相','像'
    standard_char, tested_char = '思', '紫'
    edit_distance = get_edit_distance_from_chars(standard_char, tested_char)
    print(edit_distance)

    standard_str, tested_str = '喜爱春天的人儿是心地纯洁的人像紫罗兰花儿一样是我知心朋友','惜爱春天的人儿时新的一处镶紫罗兰花儿一样是我知心朋友'
    # modify_str = modify_tyz(standard_str,tested_str)
    modify_str = modify_tyz_by_position(standard_str,tested_str)
    modify_str = ''.join(modify_str)
    print(modify_str)
