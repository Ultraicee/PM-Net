import cv2
import os
import numpy as np


imgnames = os.listdir("D:/Python3/FeatureMatch/RectifiedImages2")
# savepath = "D:\\Python3\\MyCNN\\data\\surf_descriptor_matched_img\\"
halfnum = int(0.5*len(imgnames))
src_xy = []
dst_xy = []
surf = cv2.xfeatures2d.SURF_create()
savepath = "D:\\Python3\\PM-Net\\surf_xy_records\\"
for i in range(10):
    imgname1 = 'D:/Python3/FeatureMatch/RectifiedImages2/'+imgnames[i]

    img1 = cv2.imread(imgname1)
    kp1, des1 = surf.detectAndCompute(img1, None)
    img = cv2.drawKeypoints(image=img1, outImage=img1, keypoints=kp1,
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                            color=(0, 0, 255))
    cv2.imshow('surf', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("surf.png",img)
    # pts_data = np.zeros((len(kp1), 3), dtype='int')
    # # 改变数组的表现形式，不改变数据内容，数据内容是每个关键点的坐标位置
    # pts_xy = cv2.KeyPoint_convert(kp1)
    # num = 0
    # for xy in pts_xy:
    #     pts_data[num,:]=xy[0],xy[1],i
    #     num += 1
    # np.savetxt(savepath+"surf_xy_records"+str(i)+".txt", pts_data, fmt='%i', delimiter=' ')
    # print("Finished!")

