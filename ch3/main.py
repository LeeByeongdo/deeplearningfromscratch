import numpy as np
from matplotlib import pyplot as plt

from common.functions import sigmoid, relu

if __name__ == '__main__':
    x1 = np.arange(-7, 7, 0.1)
    x2 = []

    for x in x1:
        x2.append(relu(x))

    plt.plot(x1, x2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axvline(0, color='gray', linewidth=.5)

    plt.show()
