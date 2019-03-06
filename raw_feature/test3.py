import heapq

a = [43, 5, 65, 4, 5, 8, 87]

re1 = heapq.nlargest(3, a)  # 求最大的三个元素，并排序
re2 = map(a.index, heapq.nlargest(3, a))  # 求最大的三个索引    nsmallest与nlargest相反，求最小

print(re1)
print(list(re2))  # 因为re1由map()生成的不是list，直接print不出来，添加list()就行了