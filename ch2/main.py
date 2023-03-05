import matplotlib.pyplot as plt
import numpy as np
def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y


if __name__ == '__main__':
    x1 = range(-2, 3)
    x2 = []

    for x in x1:
        x2.append(0.5 - x)

    plt.plot(x1, x2)
    plt.xlabel('x1')
    plt.ylabel('x2')

    plt.axhline(0, color='gray', linewidth=.5)
    plt.axvline(0, color='gray', linewidth=.5)
    plt.scatter([0, 1], [0, 1])
    plt.scatter([0, 1], [1, 0])
    plt.fill_between(x=x1, y1=x2, y2=-1.5, alpha=0.2)

    plt.show()
