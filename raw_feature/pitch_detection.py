import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('f:/3.png')

#img = cv2.imread('F:/35-95-A.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

mser = cv2.MSER_create(_min_area=20,_max_area=2000, _max_variation=0.7)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))         #定义矩形结构元素

opened2 = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel,iterations=1)     #开运算2
regions, boxes = mser.detectRegions(gray)

# print(img.item(10,10,0))
# print(img.item(10,10,1))
# print(img.item(10,10,2))
#
# print(img.item(1,323,0))
# print(img.item(1,323,1))
# print(img.item(1,323,2))

c_max = np.argmax(opened2,axis=0)
print(c_max.shape[0])
print(c_max)

for x in range(c_max.shape[0]):
    #img.item(x, c_max[x], 0)
    img.itemset((c_max[x],x, 0), 0)
    img.itemset((c_max[x],x,1), 0)
    img.itemset((c_max[x],x, 2), 0)


# for box in boxes:
#     x, y, w, h = box
#     if y > 250 and w > h*5:
#         cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

plt.imshow(img, 'brg')
plt.show()