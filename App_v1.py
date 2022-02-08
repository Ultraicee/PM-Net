from tqdm import tqdm
import cv2
import numpy as np
import tensorflow as tf
# from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models
tf.keras.backend.set_floatx('float64')


def GetPatchImage(img_id,row_index, col_index,mode):
    # deal with dmp
    name =  "%04d" % (int(img_id)+1)
    if mode == 1:
        container_img = np.array(cv2.imread(filepath1 + name + '.bmp'), dtype=np.int)
    elif mode == 2:
        container_img = np.array(cv2.imread(filepath1 + name + '.bmp'), dtype=np.int)
    # extract the patch from the iamge
    patch_image = container_img[row_index-32 : row_index+32, col_index-32 : col_index+32]
    return patch_image


if __name__ == '__main__':
    new_model = models.load_model('model_liberty_50000.h5')
    # tf.keras.utils.plot_model(new_model, to_file='model.png', show_shapes=True, show_layer_names=True,rankdir='TB', dpi=900, expand_nested=True)
    filepath1 = "./img_data/ICur1_"
    filepath2 = "./img_data/ICur2_"

    right = 0
    false = 0
    data_num = 3030
    dst_xys = np.loadtxt("xy_records.txt")
    scores = []

    for n in tqdm(range(300)):
        # 读取第n帧左帧的所有img1
        y1,x1,y2,x2,img_index = dst_xys[n,:]
        if x1>=32 and x1<=288-32 and y1>=32 and y1<=360-32:
            img1 = GetPatchImage(img_index,int(x1),int(y1),1)
            img1 = img1 / 255.  # (num,64,64,3)
            img1 = img1.reshape(1,64,64,3)
        else:
            continue
        name2 = "%04d" % (int(img_index) + 1)
        init_img2 = np.array(cv2.imread(filepath2 + name2 + '.bmp'), dtype=np.int)
        init_img2 = init_img2 / 255.
        nearly_out = ((y2-32)/3)+1
        # nearly_out = y2
        # print("True location_y: ", int(nearly_out))
        row = 0
        predict_array = np.zeros((99,2))
        for m in range(32,328,3):
            # 滑动窗口截取图像
            img2 = init_img2[int(x1)-32:int(x1)+32,m-32:m+32]
            if np.size(img2) == 64*64*3:
                img2 = img2.reshape(1,64,64,3)
            else:
                continue
            y_preds = new_model.predict([img1, img2])
            predict_array[row,:] = y_preds
            row += 1

        predict_y = np.argmax(predict_array,axis=0)[1]
        # 预测分数大于3时启用
        if predict_array[predict_y,1]<3:
            #print("No good predict_y.")
            continue
        #print("Predict_y:",predict_y)
        scores.append(predict_array[predict_y,1])
        # predict_y = 3*(int(predict_y)-1)+32

        if abs(int(predict_y)-nearly_out) < 2:
            right += 1
            # print('Predict Right!')
        else:
            false += 1
            # print('Predict False!')
    print("Total right %d and false %d, acc = %f." %(right,false,right/(right+false))) #50000->0.936;100000->0.894