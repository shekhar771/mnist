import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def show_graph(images, labels):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        idx = np.random.randint(0, images.shape[0]-1)
        img = images[idx]
        label = labels[idx]
        plt.subplot(5, 5, i+1)
        plt.title(str(label))
        plt.tight_layout()
        plt.imshow(img, cmap='gray')
    plt.show()


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # print('x_train', x_train.shape)
    # print('x_test', x_test.shape)
    # print('y_train', y_train.shape)
    # print('y_test', y_test.shape)
    show_graph(x_train, y_train)
