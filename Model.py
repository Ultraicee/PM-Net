import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Model
from tensorflow.keras.backend import concatenate
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPool2D
tf.keras.backend.set_floatx('float64')

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def feature_net(input_data):
    conv0 = Conv2D(24, kernel_size=5, strides=(1, 1), padding='SAME', activation='relu')(input_data)
    pool0 = MaxPool2D((3, 3), strides=(2, 2), padding='SAME')(conv0)

    conv1 = Conv2D(64, kernel_size=5, strides=(1, 1), padding='SAME', activation='relu')(pool0)
    pool1 = MaxPool2D((3, 3), strides=(2, 2), padding='SAME')(conv1)

    conv2 = Conv2D(96, kernel_size=3, strides=(1, 1), padding='SAME', activation='relu')(pool1)
    pool2 = MaxPool2D((3, 3), strides=(2, 2), padding='SAME')(conv2)

    fc0 = Dense(256)(pool2)

    return Flatten()(fc0)


if __name__ == '__main__':

    container_dir = "D:\\Python3\\MatchNet_dataset\\liberty\\patches"
    data_num = 50000

    # input image size and channels
    img_width, img_height, channels = 64, 64, 3
    input_data1 = Input(shape=(img_width, img_height, channels))
    input_data2 = Input(shape=(img_width, img_height, channels))

    # Similarity network
    merge = concatenate([feature_net(input_data1),feature_net(input_data2)],axis=-1)
    fc1 = Dense(512, activation='relu', name='fc1')(merge)
    dropout1 = Dropout(0.1)(fc1)
    fc2 = Dense(512, activation='relu', name='fc2')(dropout1)
    dropout2 = Dropout(0.1)(fc2)

    output = Dense(2, name='fc3')(dropout2)

    matchnet = Model(inputs=[input_data1, input_data2], outputs=output)
    matchnet.compile(optimizer=tf.keras.optimizers.SGD(lr=0.01,momentum=0.9,decay=1e-6,nesterov=True),
                     loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                     metrics=['accuracy'])

    # tf.keras.utils.plot_model(matchnet, to_file='model.png', show_shapes=True, show_layer_names=True,rankdir='TB', dpi=900, expand_nested=True)
    checkpoint_path = "./model_save/training_liberty.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    # 创建一个保存模型权重的回调
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1,
                                                     period=1)

    # train
    img1s = np.load("img1s_50000.npy")
    img2s = np.load("img2s_50000.npy")
    labels = np.load("labels_50000.npy")
    labels = to_categorical(labels)

    history = matchnet.fit([img1s,img2s],labels, epochs=20,  verbose=2, callbacks=[cp_callback],validation_split=0.2)

    # 将整个模型保存为HDF5文件
    matchnet.save('./model_liberty.h5')


