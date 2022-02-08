import numpy as np
import cv2
from pylab import *
import random
xys_matched = []
src_xys = np.loadtxt("./surf_xy_records/surf_xy_records4.txt")
dst_xys = np.load("./predict_xy_records/predict_xy4.npy")

dst_index = 0
for index in dst_xys[:,2]:
    xys_matched.append([src_xys[int(index),0],src_xys[int(index),1],dst_xys[dst_index,0],dst_xys[dst_index,1]])
    dst_index += 1

img1 = cv2.imread("D:\\Python3\\FeatureMatch\\RectifiedImages2\\ICur1_0005.bmp")
# img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.imread("D:\\Python3\\FeatureMatch\\RectifiedImages2\\ICur2_0005.bmp")
# img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
color_list = [0,255]
img = np.concatenate((img1,img2),axis=1)
row = 0
for [y1,x1,y2,x2] in xys_matched:
    color = (random.choice(color_list),random.choice(color_list),random.choice(color_list))
    cv2.circle(img,(int(y1),int(x1)), 3, color)
    cv2.circle(img, (int(y2)+360, int(x2)), 3, color)
    cv2.line(img,(int(y1),int(x1)),(int(y2+360), int(x2)),color)
    row += 1
cv2.imwrite("./imgs/4-"+str(len(dst_xys))+".bmp",img)
#cv2.imshow("PM_detected", img)
# cv2.waitKey()
# cv2.destroyAllWindows()