import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models
tf.keras.backend.set_floatx('float64')

if __name__ == '__main__':
    new_model = models.load_model('model_liberty.h5')
    # tf.keras.utils.plot_model(new_model, to_file='new_model.png', show_shapes=True, show_layer_names=True, rankdir='TB',
                              #dpi=900, expand_nested=True)
    img1s_test = np.load("img1s_10000.npy")
    img2s_test = np.load("img2s_10000.npy")
    labels_test = np.load("labels_10000.npy")
    labels_test = to_categorical(labels_test)

    test_loss, test_acc = new_model.evaluate([img1s_test, img2s_test], labels_test, verbose=2)
    print("Test accuracy: {:5.2f}%".format(100 * test_acc))