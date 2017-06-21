import numpy as np
import math
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, optimizers, serializers, Variable
from chainer import function
from chainer.utils import type_check
from .ops import *

class DCGANDiscriminator(chainer.Chain):
    def __init__(self, in_ch=3, base_size=128, down_layers=4, use_bn=True, noise_all_layers=False, conv_as_last=False, w_init=None):
        layers = {}

        self.down_layers = down_layers
        self.conv_as_last = conv_as_last

        if use_bn:
            norm = 'bn'
        else:
            norm = None

        act = F.leaky_relu
        if w_init is None:
            w_init = chainer.initializers.Normal(0.02)

        layers['c_first'] = NNBlock(in_ch, base_size, nn='down_conv', norm=None, activation=act, noise=noise_all_layers, w_init=w_init)
        base = base_size

        for i in range(down_layers-1):
            layers['c'+str(i)] = NNBlock(base, base*2, nn='down_conv', norm=norm, activation=act, noise=noise_all_layers, w_init=w_init)
            base*=2

        if conv_as_last:
            layers['c_last'] = NNBlock(base, 1, nn='conv', norm=None, activation=None, w_init=w_init)
        else:
            layers['c_last'] = NNBlock(None, 1, nn='linear', norm=None, activation=None, w_init=w_init)

        super(DCGANDiscriminator, self).__init__(**layers)

    def __call__(self, x, test=False):
        h = self.c_first(x, test=test)
        for i in range(self.down_layers-1):
            h = getattr(self, 'c'+str(i))(h, test=test)
        if not self.conv_as_last:
            _b, _ch, _w, _h = h.data.shape
            h = F.reshape(h, (_b, _ch*_w*_h))
        h = self.c_last(h, test=test)
        return h

    def clip(self):
        weight_clipping(self)
