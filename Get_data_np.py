import numpy as np
import cv2
from tqdm import tqdm


def GetPatchImage(patch_id):
    # deal with dmp
    PATCHES_PER_IMAGE = 16 * 16
    PATCHES_PER_ROW = 16
    PATCH_SIZE = 64
    container_idx, container_order = divmod(patch_id, PATCHES_PER_IMAGE)
    row_index, col_index = divmod(container_order, PATCHES_PER_ROW)

    # extract the patch from the iamge
    str_container_idx = "%04d" % container_idx
    container_img = cv2.imread(container_dir + str_container_idx + '.bmp')
    patch_image = container_img[PATCH_SIZE * row_index: PATCH_SIZE * (row_index + 1),
                  PATCH_SIZE * col_index:PATCH_SIZE * (col_index + 1)]
    return patch_image

if __name__ == '__main__':
    # get patch_id and label
    container_dir = "D:\\Python3\\MatchNet_dataset\\liberty\\patches"
    data_num = 50000
    txt_filename = "D:\\Python3\\MatchNet_dataset\\liberty\\m50_50000_50000_0.txt"
    img_width, img_height, channels = 64, 64, 3

    img1s = np.zeros((data_num, img_height, img_width, channels))
    img2s = np.zeros((data_num, img_height, img_width, channels))
    labels = np.zeros((data_num, 1), dtype='int32')

    with open(txt_filename, 'r') as file_to_read:
        for num in tqdm(range(data_num)):
            lines = file_to_read.readline()  # 整行读取数据
            patch_id1 = lines.split(' ')[0]
            img1s[num, :] = GetPatchImage(int(patch_id1)) / 255.
            patch_id2 = lines.split(' ')[3]
            img2s[num, :] = GetPatchImage(int(patch_id2)) / 255.
            patch_3d_id1 = lines.split(' ')[1]
            patch_3d_id2 = lines.split(' ')[4]
            if patch_3d_id1 == patch_3d_id2:
                label = 1
            else:
                label = 0
            labels[num, :] = label

    print(" Get data successfully.")
    np.save("img1s_50000.npy", img1s)
    np.save("img2s_50000.npy", img2s)
    np.save("labels_50000.npy", labels)
