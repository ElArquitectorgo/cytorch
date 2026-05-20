import keras
import numpy as np

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_flattened = np.reshape(x_train, (60000, 784))
x_flattened = x_flattened.astype(np.float32) / 255

y_train = y_train.astype(np.float32)

x_flattened.tofile('train_data_mnist.bin')
y_train.tofile('train_labels_mnist.bin')