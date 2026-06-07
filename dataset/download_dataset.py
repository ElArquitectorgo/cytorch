import keras
import numpy as np

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train_flattened = np.reshape(x_train, (60000, 784))
x_train_flattened = x_train_flattened.astype(np.float32) / 255

y_train = y_train.astype(np.float32)

x_train_flattened.tofile('train_data_mnist.bin')
y_train.tofile('train_labels_mnist.bin')

x_test_flattened = np.reshape(x_test, (10000, 784))
x_test_flattened = x_test_flattened.astype(np.float32) / 255

y_test = y_test.astype(np.float32)

x_test_flattened.tofile('test_data_mnist.bin')
y_test.tofile('test_labels_mnist.bin')