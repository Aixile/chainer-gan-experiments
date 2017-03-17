import numpy as np
import math
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, optimizers, serializers, Variable
from chainer import function
from chainer.utils import type_check


class DIS_64(chainer.Chain):
    def __init__(self, channel_in=3):
        super(DIS_64, self).__init__(
            c1=L.Convolution2D(None, 128, 4, 2, 1, wscale=0.02*math.sqrt(channel_in*4*4)), #32*32
            c2=L.Convolution2D(128, 256, 4, 2, 1, wscale=0.02*math.sqrt(128*4*4)), #16*16
            c3=L.Convolution2D(256, 512, 4, 2, 1, wscale=0.02*math.sqrt(256*4*4)), #8*8
            c4=L.Convolution2D(512, 1024, 4, 2, 1, wscale=0.02*math.sqrt(512*4*4)), #4*4
            l8l=L.Linear(None, 1, wscale=0.02*math.sqrt(1024*4*4)),
            bnc2=L.BatchNormalization(256),
            bnc3=L.BatchNormalization(512),
            bnc4=L.BatchNormalization(1024)
        )

    def __call__(self, x, test=False):
        h = F.relu(self.c1(x))
        h = F.relu(self.bnc2(self.c2(h), test=test))
        h = F.relu(self.bnc3(self.c3(h), test=test))
        h = F.relu(self.bnc4(self.c4(h), test=test))
        return self.l8l(h)
