# coding=utf-8
import cv2
import numpy as np

img = cv2.imread("f:/3.png", 0)

img = cv2.GaussianBlur(img, (3, 3), 0)
canny = cv2.Canny(img, 50, 150)

cv2.imshow('Canny', canny)
cv2.waitKey(0)
cv2.destroyAllWindows()