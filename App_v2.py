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
    filepath1 = "D:\\Python3\\FeatureMatch\\RectifiedImages2\\ICur1_"
    filepath2 = "D:\\Python3\\FeatureMatch\\RectifiedImages2\\ICur2_"

    # dst_xys = np.loadtxt("xy_records.txt")
    for order in range(2,10):
        dst_xys = np.loadtxt("./surf_xy_records/surf_xy_records"+str(order)+".txt")
        predict_xy = []

        for n in range(len(dst_xys)):
            #y1,x1,y2,x2,img_index = dst_xys[n,:]
            y1, x1, img_index = dst_xys[n, :]
            if x1>=32 and x1<=288-32 and y1>=32 and y1<=360-32:
                img1 = GetPatchImage(img_index,int(x1),int(y1),1)
                img1 = img1 / 255.  # (num,64,64,3)
                img1 = img1.reshape(1,64,64,3)
            else:
                print("Out of range, miss the point",n)
                continue
            name2 = "%04d" % (int(img_index) + 1)
            init_img2 = np.array(cv2.imread(filepath2 + name2 + '.bmp'), dtype=np.int)
            init_img2 = init_img2 / 255.

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
                print("Oops, the No.",n,"of Img ",int(img_index), " has not good predict_y.")
                continue
            else:
                predict_y = 3 * (int(predict_y) - 1) + 32
                print(n,"Predict_y:",predict_y)
                [y,x,index]= [predict_y,x1,n]
                predict_xy.append([y,x,index])

        predict_xy = np.array((predict_xy),dtype='int')
        np.save("./predict_xy_records/predict_xy"+str(order)+".npy", predict_xy)
        print("-----------",order,"Finished!-----------")
