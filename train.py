#!/usr/bin/env python3

import numpy as np
from chainer import optimizers, serializers

from config import Config
from MyNet import MyNet
from read_data import read_data


if __name__ == '__main__':
    train_images, train_labels = read_data('train.txt')
    test_images, test_labels = read_data('test.txt')

    model = MyNet()
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    num_train = len(train_images)
    num_test = len(test_images)

    for epoch in range(100):
        accum_loss = None
        bs = Config.BATCH_SIZE
        sffindx = np.random.permutation(num_train)
        for i in range(0, num_train, bs):
            x = [train_images[j] for j in sffindx[i:(i + bs) if(i + bs < num_train) else num_train]]
            t = [train_labels[j] for j in sffindx[i:(i + bs) if(i + bs < num_train) else num_train]]

            model.cleargrads()
            loss, acc = model.forward(x, t)
            accum_loss = loss if accum_loss is None else accum_loss + loss

        accum_loss.backward()
        optimizer.update()

        if epoch % 10 == 0:
            print(epoch, accum_loss.data)

    outfile = "models/mynet.model"
    serializers.save_npz(outfile, model)
