import sys, os

import numpy as np
from matplotlib import pyplot as plt

from ch5.TwoLayerNet import TwoLayerNet
from common.utils import image_to_bytes

sys.path.append(os.pardir)
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

if __name__ == '__main__':
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []
    bd_three_test_list = []

    iters_num = 10000
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.1
    network = TwoLayerNet(input_size=784, hidden_size=30, output_size=10)

    img_bytes = image_to_bytes("../sample/three.png")

    iter_per_epoch = max(train_size / batch_size, 1)

    for i in range(iters_num):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        grad = network.gradient(x_batch, t_batch)

        for key in ('W1', 'b1', 'W2', 'b2'):
            network.params[key] -= learning_rate * grad[key]

        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)

        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            bdx = np.array(img_bytes)
            test_bd_three = network.accuracy(bdx, np.array([3]))
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            bd_three_test_list.append(test_bd_three)
            print(train_acc, test_acc)

    plt.plot(range(len(train_acc_list)), train_acc_list)
    plt.plot(range(len(train_acc_list)), test_acc_list)
    plt.show()
