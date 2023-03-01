import sys, os

from ch4.simplenet import SimpleNet
from ch4.two_layer_net import TwoLayerNet

sys.path.append(os.pardir)
import numpy as np
import matplotlib.pylab as plt
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=False)


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size

def cross_entropy_error_v_label(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    print(y)
    print(t)
    print(y[np.arange(batch_size), t])
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)

def function_1(x):
    return 0.01 * x**2 + 0.1 * x

def tangent_line(f, x):
    d = numerical_diff(f, x)
    y = f(x) - d*x
    return lambda t: d*t + y


def _numerical_gradient_no_batch(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 値を元に戻す

    return grad


def numerical_gradient(f, X):
    if X.ndim == 1:
        return _numerical_gradient_no_batch(f, X)
    else:
        grad = np.zeros_like(X)

        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_no_batch(f, x)

        return grad

def function_2(x):
    if x.ndim == 1:
        return np.sum(x**2)
    else:
        return np.sum(x**2, axis=1)

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    x_history = []

    for i in range(step_num):
        x_history.append( x.copy() )

        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x, np.array(x_history)



if __name__ == '__main__':
    train_loss_list = []

    iters_num = 10000
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.1
    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

    for i in range(iters_num):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        grad = network.numerical_gradient(x_batch, t_batch)

        for key in ('W1', 'b1', 'W2', 'b2'):
            network.params[key] -= learning_rate * grad[key]

        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)

    plt.plot(range(iters_num), train_loss_list)
    plt.show()




    # net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
    # print(net.params['W1'].shape)
    # print(net.params['b1'].shape)
    # print(net.params['W2'].shape)
    # print(net.params['b1'].shape)
    #
    # x = np.random.rand(100, 784)
    # t = np.random.rand(100, 10)
    # y = net.predict(x)
    # grads = net.numerical_gradient(x, t)
    # print(grads)


    # net = SimpleNet()
    # print(net.W)
    #
    # x = np.array([0.6, 0.9])
    # p = net.predict(x)
    # print(p)
    #
    # np.argmax(p)
    #
    # t = np.array([0, 0, 1])
    # loss = net.loss(x, t)
    # print(loss)
    #
    # def f(W):
    #     return net.loss(x, t)
    #
    # dW = numerical_gradient(f, net.W)
    # print(dW)



    # init_x = np.array([-3.0, 4.0])
    #
    # lr = 0.1
    # step_num = 20
    # x, x_history = gradient_descent(function_2, init_x, lr=lr, step_num=step_num)
    #
    # plt.plot([-5, 5], [0, 0], '--b')
    # plt.plot([0, 0], [-5, 5], '--b')
    # plt.plot(x_history[:, 0], x_history[:, 1], 'o')
    #
    # plt.xlim(-3.5, 3.5)
    # plt.ylim(-4.5, 4.5)
    # plt.xlabel("X0")
    # plt.ylabel("X1")
    # plt.show()


    # x = np.arange(0.0, 20.0, 0.1)
    # y = function_1(x)
    # plt.xlabel("x")
    # plt.ylabel("f(x)")
    # tf = tangent_line(function_1, 5)
    # y2 = tf(x)
    #
    # plt.plot(x, y)
    # plt.plot(x, y2)
    # plt.show()
    #
    # print(numerical_diff(function_1, 5))
    # print(numerical_diff(function_1, 10))


    # y = [[0,0,0,0,0,1,0,0,0,0], [0,1,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,1,0,0,0]]
    # a = cross_entropy_error_v_label(np.array(y), t_train[0:3])
    # print(a)
