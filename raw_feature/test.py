import cv2
import numpy as np
original_img = cv2.imread('f:/2.png',0)
# gray_res = cv2.resize(original_img,None,fx=1,fy=1,
#                  interpolation = cv2.INTER_CUBIC)                #图形太大了缩小一点
# B, G, img = cv2.split(res)
# _,RedThresh = cv2.threshold(img,160,255,cv2.THRESH_BINARY)     #设定红色通道阈值160（阈值影响开闭运算效果）
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(20,20))         #定义矩形结构元素

closed1 = cv2.morphologyEx(original_img, cv2.MORPH_CLOSE, kernel,iterations=1)    #闭运算1
closed2 = cv2.morphologyEx(original_img, cv2.MORPH_CLOSE, kernel,iterations=3)    #闭运算2
opened1 = cv2.morphologyEx(original_img, cv2.MORPH_OPEN, kernel,iterations=1)     #开运算1
opened2 = cv2.morphologyEx(original_img, cv2.MORPH_OPEN, kernel,iterations=3)     #开运算2
gradient = cv2.morphologyEx(original_img, cv2.MORPH_GRADIENT, kernel)             #梯度

#显示如下腐蚀后的图像
#cv2.imshow("gray_res", gray_res)
cv2.imshow("Close1",closed1)
#cv2.imshow("Close2",closed2)
#cv2.imshow("Open1", opened1)
#cv2.imshow("Open2", opened2)
#cv2.imshow("gradient", gradient)

cv2.waitKey(0)
cv2.destroyAllWindows()