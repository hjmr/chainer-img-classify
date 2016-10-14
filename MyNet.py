import numpy as np
from chainer import Chain, Variable
import chainer.links as L
import chainer.functions as F

from config import Config


class MyNet(Chain):

    def __init__(self):
        img_channels = 1 if Config.IMAGE_MONO else 3
        super(MyNet, self).__init__(
            conv1=L.Convolution2D(img_channels, Config.CONV1_OUT_CHANNELS, Config.CONV_SIZE),
            conv2=L.Convolution2D(Config.CONV1_OUT_CHANNELS, Config.CONV2_OUT_CHANNELS, Config.CONV_SIZE),
            l3=L.Linear(Config.NUM_HIDDEN_NEURONS1, Config.NUM_HIDDEN_NEURONS2),
            l4=L.Linear(Config.NUM_HIDDEN_NEURONS2, Config.NUM_CLASSES)
        )

    def forward(self, x, t, train=True, dropout_ratio=0.5):
        x_data = Variable(np.array(x).astype(np.float32))
        if t is not None:
            t_data = Variable(np.array(t).astype(np.int32))
        else:
            train = False

        h_conv1 = F.relu(self.conv1(x_data))
        h_pool1 = F.max_pooling_2d(h_conv1, ksize=(4, 4))
        h_norm1 = F.local_response_normalization(h_pool1)

        h_conv2 = F.relu(self.conv2(h_norm1))
        h_norm2 = F.local_response_normalization(h_conv2)
        h_pool2 = F.max_pooling_2d(h_norm2, ksize=(4, 4))

        h3 = F.dropout(F.relu(self.l3(h_pool2)), train=train, ratio=dropout_ratio)
        y = self.l4(h3)
        if train:
            return F.softmax_cross_entropy(y, t_data), F.accuracy(y, t_data)
        else:
            return F.softmax(y), y.data
