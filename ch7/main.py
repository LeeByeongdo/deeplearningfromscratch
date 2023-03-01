import sys, os

import numpy as np

sys.path.append(os.pardir)
from common.util import im2col

if __name__ == '__main__':
    x1 = np.random.rand(1, 3, 7, 7)
    col1 = im2col(x1, 5, 5, stride=1, pad=0)
    print(col1.shape)

