import numpy as np
import cv2
import pandas as pd


XYZ=[]
for i in range(5):
    src_xys = np.loadtxt("./surf_xy_records/surf_xy_records"+str(i)+".txt")
    dst_xys = np.load("./predict_xy_records/predict_xy"+str(i)+".npy")

    dst_index = 0
    pts1, pts2 = [], []
    for index in dst_xys[:,2]:
        pts1.append((src_xys[int(index),0],src_xys[int(index),1]))
        pts2.append((dst_xys[dst_index, 0], dst_xys[dst_index, 1]))
        dst_index += 1

    # 基于三角测量还原
    P1 = np.array([[524.844130000000, 0., 217.173580000000, 0.],
                   [0., 577.110240000000, 150.763790000000, 0.],
                   [0.,                0.,                1, 0.]])
    P2 = np.array([[524.520233073651, 7.52736070286657, 160.059379742608, 2970.80088985420],
                   [-13.4461587284326, 582.512489166281, 156.250630493458, 39.9550566570000],
                   [-0.0426087861184525, 0.00970718276166426, 0.999044674650910, 0.398860000000000]])
    points4D = cv2.triangulatePoints(P1, P2, np.mat(pts1).T, np.mat(pts2).T, points4D=10)

    for point in points4D.T:
        if point[0]*point[1]*point[2] ==0 or point[2]>0:
            continue
        else:
            XYZ.append([point[0],point[1],point[2]])

print("Num of points4D:", len(XYZ))
df = pd.DataFrame(XYZ,columns=['X','Y','Z'])
df.to_excel("XYZ1.xlsx",index=False)