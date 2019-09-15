          # -*- coding: UTF-8 -*-
import numpy as np

##### https://blog.csdn.net/miner_zhu/article/details/81159902
def my_find_lcseque(s1, s2): #s1 为标准字符串
    if len(s1) == len(s2):
        tmp = ''
        positions = []
        for i,x in enumerate(s1):
            if x == s2[i]:
                tmp = tmp + str(x)
                positions.append(i)
        return tmp,positions,positions
    #先找最长公共子串
    if s1[:3] == s2[:3]:
        lcsubstr = s1[:3]
    else:
        lcsubstr, mmax = find_lcsubstr(s1, s2)

    #然后以最长公共子串位置切分s1
    split_point = s1.index(lcsubstr)
    before_s1 = s1[:split_point]
    after_s1 = s1[split_point:]

    # 然后以最长公共子串位置切分s2
    split_points = find_all(lcsubstr,s2)
    offset = [np.abs(s - split_point) for s in split_points]
    min_index = offset.index(np.min(offset))
    split_point = split_points[min_index]
    split_point2 = split_point
    # split_point = s2.index(lcsubstr)
    before_s2 = s2[:split_point]
    after_s2 = s2[split_point:]
    before_lcseque = ''
    after_lcseque = ''
    if len(before_s1) > 0 and len(before_s2) > 0:
        before_lcseque = find_lcseque(before_s1, before_s2)

    if len(after_s1) > 0 and len(after_s2) > 0:
        after_lcseque = find_lcseque(after_s1, after_s2)

    lcseque = before_lcseque + after_lcseque

    positions = []
    raw_positions = []
    if len(before_lcseque) > 0:
        for i,b in enumerate(before_lcseque):
            indexs = [n for n in range(len(before_s1)) if before_s1[n] == b and n < i + 4]
            if len(indexs) > 0:
                for x in indexs:
                    if x not in positions:
                        positions.append(x)
                        break
            indexs = [n for n in range(len(before_s2)) if before_s2[n] == b  and n < i + 4]
            if len(indexs) > 0:
                if len(positions) > 0 and positions[-1] in indexs:
                    raw_positions.append(x)
                else:
                    for x in indexs:
                        if len(raw_positions) == 0:
                            raw_positions.append(x)
                            break
                        elif x not in raw_positions and x > raw_positions[-1]:
                            raw_positions.append(x)
                            break

    if len(after_lcseque) > 0:
        split_point = s1.index(lcsubstr)
        after_s1 = after_s1[len(lcsubstr):] #取公共子串后面的内容
        if len(lcsubstr) > 0:
            for i in range(len(lcsubstr)):
                index = split_point + i
                positions.append(index)
                index = split_point2 + i
                raw_positions.append(index)
            after_s2 = after_s2[len(lcsubstr):] #取公共子串后面的内容
            for i,b in enumerate(after_s2):
                indexs = [n for n in range(len(after_s1)) if after_s1[n] == b and n < i + 3]
                if len(indexs)>0:
                    for x in indexs:
                        if after_s1[x:] == after_s2[i:]: #如果后面的字符都能匹配上
                            if x+split_point+len(lcsubstr)  not in positions and x+split_point+len(lcsubstr) > positions[-1]:
                                positions.append(x+split_point+len(lcsubstr))
                                raw_positions.append(i + len(lcsubstr))
                                break
                    for x in indexs:
                        if x+split_point+len(lcsubstr)  not in positions and x+split_point+len(lcsubstr) > positions[-1]:
                            positions.append(x+split_point+len(lcsubstr))
                            raw_positions.append(i + len(lcsubstr))
                            break
    tmp = []
    for i in range(len(positions)-1):
        if positions[i]<np.min(positions[i+1:]):
                tmp.append(positions[i])
    tmp.append(positions[-1])
    positions = tmp
    return lcseque,positions,raw_positions

def find_lcseque(s1, s2):   
     # 生成字符串长度加1的0矩阵，m用来保存对应位置匹配的结果  
    m = [ [ 0 for x in range(len(s2)+1) ] for y in range(len(s1)+1) ]   
    # d用来记录转移方向  
    d = [ [ None for x in range(len(s2)+1) ] for y in range(len(s1)+1) ]   

    for p1 in range(len(s1)):   
        for p2 in range(len(s2)):   
            if s1[p1] == s2[p2]:            #字符匹配成功，则该位置的值为左上方的值加1  
                m[p1+1][p2+1] = m[p1][p2]+1  
                d[p1+1][p2+1] = 'ok'            
            elif m[p1+1][p2] > m[p1][p2+1]:  #左值大于上值，则该位置的值为左值，并标记回溯时的方向  
                m[p1+1][p2+1] = m[p1+1][p2]   
                d[p1+1][p2+1] = 'left'            
            else:                           #上值大于左值，则该位置的值为上值，并标记方向up  
                m[p1+1][p2+1] = m[p1][p2+1]     
                d[p1+1][p2+1] = 'up'           
    (p1, p2) = (len(s1), len(s2))   
    #print(numpy.array(d))
    s = []   
    while m[p1][p2]:    #不为None时  
        c = d[p1][p2]  
        if c == 'ok':   #匹配成功，插入该字符，并向左上角找下一个  
            s.append(s1[p1-1])  
            p1-=1  
            p2-=1   
        if c =='left':  #根据标记，向左找下一个  
            p2 -= 1  
        if c == 'up':   #根据标记，向上找下一个  
            p1 -= 1  
    s.reverse()   
    return ''.join(s)


def find_lcseque_for_note(s1, s2):
    # 生成字符串长度加1的0矩阵，m用来保存对应位置匹配的结果
    m = [[0 for x in range(len(s2) + 1)] for y in range(len(s1) + 1)]
    # d用来记录转移方向
    d = [[None for x in range(len(s2) + 1)] for y in range(len(s1) + 1)]

    for p1 in range(len(s1)):
        for p2 in range(len(s2)):
            # if s1[p1] == s2[p2] or np.abs(int(s1[p1]) - int(s2[p2])) <= 1:  # 字符匹配成功，则该位置的值为左上方的值加1
            if s1[p1] == s2[p2]:  # 字符匹配成功，则该位置的值为左上方的值加1
                m[p1 + 1][p2 + 1] = m[p1][p2] + 1
                d[p1 + 1][p2 + 1] = 'ok'
            elif m[p1 + 1][p2] > m[p1][p2 + 1]:  # 左值大于上值，则该位置的值为左值，并标记回溯时的方向
                m[p1 + 1][p2 + 1] = m[p1 + 1][p2]
                d[p1 + 1][p2 + 1] = 'left'
            else:  # 上值大于左值，则该位置的值为上值，并标记方向up
                m[p1 + 1][p2 + 1] = m[p1][p2 + 1]
                d[p1 + 1][p2 + 1] = 'up'
    (p1, p2) = (len(s1), len(s2))
    # print(numpy.array(d))
    s = []
    while m[p1][p2]:  # 不为None时
        c = d[p1][p2]
        if c == 'ok':  # 匹配成功，插入该字符，并向左上角找下一个
            s.append(s1[p1 - 1])
            p1 -= 1
            p2 -= 1
        if c == 'left':  # 根据标记，向左找下一个
            p2 -= 1
        if c == 'up':  # 根据标记，向上找下一个
            p1 -= 1
    s.reverse()
    return ''.join(s)

def find_lcsubstr(s1, s2):
    m=[[0 for i in range(len(s2)+1)]  for j in range(len(s1)+1)]  #生成0矩阵，为方便后续计算，比字符串长度多了一列
    mmax=0   #最长匹配的长度
    p=0  #最长匹配对应在s1中的最后一位
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i]==s2[j]:
                m[i+1][j+1]=m[i][j]+1
                if m[i+1][j+1]>mmax:
                    mmax=m[i+1][j+1]
                    p=i+1
    return s1[p-mmax:p],mmax   #返回最长子串及其长度

def find_all(sub, s):
    index_list = []
    index = s.find(sub)
    while index != -1:
        index_list.append(index)
        index = s.find(sub, index + 1)

    if len(index_list) > 0:
        return index_list
    else:
        return -1


if __name__ == '__main__':
    a = 'GEGCGGGIIC'
    #b = 'EIIGCEGC'
    b = 'GEGCEGIIC'
    # c = find_lcseque_for_note(a,b)
    # s1,mmax = find_lcsubstr(a, b)
    print(a)
    print(b)
    # print(c)
    # print(s1)
    # print(mmax)

    lcseque, positions,raw_positions = my_find_lcseque(a, b)
    print(lcseque)
    print(positions)
    print(raw_positions)