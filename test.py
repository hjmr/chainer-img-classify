import sys
import numpy as np
import chainer
from chainer import serializers

from MyNet import MyNet
from read_data import read_one_image


model = MyNet()
serializers.load_npz('models/mynet.model', model)

test_images = []
for i in range(1, len(sys.argv)):
    test_images.append(read_one_image(sys.argv[i]))

for i in range(len(test_images)):
    img = np.array([test_images[i]], dtype=np.float32)
    with chainer.using_config("train", False):
        y = model.forward(img)
        pred = np.argmax(y)
        print(pred)
