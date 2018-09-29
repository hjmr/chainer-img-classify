import numpy as np
from chainer import optimizers, serializers

from config import Config
from MyNet import MyNet
from read_data import read_data


train_images, train_labels = read_data('train.txt')

model = MyNet()
optimizer = optimizers.Adam()
optimizer.setup(model)

num_train = len(train_images)

for epoch in range(100):
    accum_loss = None
    bs = Config.BATCH_SIZE
    perm = np.random.permutation(num_train)
    for i in range(0, num_train, bs):
        x_sample = train_images[perm[i:(i + bs) if(i + bs < num_train) else num_train]]
        y_sample = train_labels[perm[i:(i + bs) if(i + bs < num_train) else num_train]]

        model.zerograds()
        loss, acc = model.train(x_sample, y_sample)
        loss.backward()
        optimizer.update()

        accum_loss = loss if accum_loss is None else accum_loss + loss

    if epoch % 10 == 0:
        print(epoch, accum_loss.data)

outfile = "models/mynet.model"
serializers.save_npz(outfile, model)
