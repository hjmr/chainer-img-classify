#!/usr/bin/env python3

import sys
import numpy as np
from chainer import optimizers, serializers

from MyNet import MyNet
from read_data import read_one_image


if __name__ == '__main__':
    model = MyNet()
    serializers.load_npz('models/mynet.model', model)
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    test_images = []
    for i in range(1, len(sys.argv)):
        test_images.append(read_one_image(sys.argv[i]))

    for i in range(len(test_images)):
        y, y_data = model.forward([test_images[i]], None, train=False)
        pred = np.argmax(y_data)
        print(pred)
