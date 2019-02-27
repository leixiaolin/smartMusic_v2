#
# # -*-encoding:utf-8-*-
# from PIL import Image
# from PIL import ImageFilter
# from PIL import ImageFont
# from PIL import ImageDraw
# import numpy as np
# from PIL import Image
#
# import cv2
#
#
# def main():
#     # img = cv2.imread("test001.jpg") #读取图片
#     # cv2.imshow("1",img) #显示图片
#     # cv2.waitKey(10000)# 延时10s
#
#     # image = Image.open("test002.png")
#     # img = image.convert('1')  # 转化为灰度图
#
#     # img.show()
#
#     img = cv2.imread("f:/2.png", 1)
#     kernel = np.ones((10, 10), np.uint8)*np.max(img)  # 全为1的过滤数组
#     print(kernel)
#
#     # 腐蚀(去白小点)
#     img2 = cv2.erode(img, kernel)  # 腐蚀
#     cv2.imshow('canny', img)
#     cv2.waitKey(2000)
#     # cv2.waitKey()
#
#     # 膨胀(去黑小点)
#     img3 = cv2.dilate(img, kernel)  # 膨胀
#     cv2.imshow('canny', img3)
#     cv2.waitKey(2000)
#     # cv2.waitKey()
#
#     # 先腐蚀后膨胀叫开运算（因为先腐蚀会分开物体，这样容易记住），其作用是：分离物体，消除小区域。
#     opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)  # 开运算
#     cv2.imshow('canny', opening)
#     cv2.waitKey(2000)
#
#     # 闭运算则相反：先膨胀后腐蚀（先膨胀会使白色的部分扩张，以至于消除/"闭合"物体里面的小黑洞，所以叫闭运算）
#     closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)  # 闭运算
#     cv2.imshow('canny', closing)
#     cv2.waitKey(2000)
#
#
# if __name__ == '__main__':
#     main()


import cv2


def closeopration(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    iClose = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return iClose


image = cv2.imread('f:/2.png')
print(image.shape)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
iClose = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
cv2.imshow('image', image)
cv2.imshow('iClose', iClose)

cv2.waitKey(0)
cv2.destroyAllWindows()
