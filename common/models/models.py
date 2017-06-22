import numpy as np
import math
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, optimizers, serializers, Variable
from chainer import function
from chainer.utils import type_check
from .ops import *

class ThreeLayersMLP(chainer.Chain):
    def __init__(self, hidden_size=768, output_size=1, use_bn=True):
        if use_bn:
            norm = 'bn'
        else:
            norm = None

        super(ThreeLayersMLP, self).__init__(
            l0 = NNBlock(None, hidden_size, norm=norm, nn='linear'),
            l1 = NNBlock(hidden_size, hidden_size, norm=norm, nn='linear'),
            l2 = NNBlock(hidden_size, output_size, norm=None, activation=None, nn='linear'),
        )

    def __call__(self, x, test=False):
        h = self.l0(x, test=test)
        h = self.l1(h, test=test)
        h = self.l2(h, test=test)
        return h

class DCGANEncoder(chainer.Chain):
    def __init__(self, in_ch=3, out_len=128, base_size=128, down_layers=4, use_bn=True, w_init=None):
        layers = {}

        self.down_layers = down_layers
        self.conv_as_last = conv_as_last

        if use_bn:
            norm = 'bn'
        else:
            norm = None

        act = F.relu
        #if w_init is None:
        #    w_init = chainer.initializers.Normal(0.02)

        layers['c_first'] = NNBlock(in_ch, base_size, nn='down_conv', norm=None, activation=act,  w_init=w_init)
        base = base_size

        for i in range(down_layers-1):
            layers['c'+str(i)] = NNBlock(base, base*2, nn='down_conv', norm=norm, activation=act,  w_init=w_init)
            base*=2

        layers['c_last'] = NNBlock(None, out_len, nn='linear', norm=None, activation=None, w_init=w_init)

        super(DCGANEncoder, self).__init__(**layers)

    def __call__(self, x, test=False):
        h = self.c_first(x, test=test)
        for i in range(self.down_layers-1):
            h = getattr(self, 'c'+str(i))(h, test=test)
        if not self.conv_as_last:
            _b, _ch, _w, _h = h.data.shape
            self.last_shape=(_b, _ch, _w, _h)
            h = F.reshape(h, (_b, _ch*_w*_h))
        h = self.c_last(h, test=test)
        return h
