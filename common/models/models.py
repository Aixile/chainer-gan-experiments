import numpy as np
import math
import chainer.functions as F
import chainer
import chainer.links as L
from chainer import cuda, optimizers, serializers, Variable
from chainer import function
from chainer.utils import type_check
from .ops import *
from chainer import cuda, optimizers, serializers, Variable

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

    def __call__(self, x, test=False, retain_forward=False):
        h = self.l0(x, test=test, retain_forward=retain_forward)
        h = self.l1(h, test=test, retain_forward=retain_forward)
        h = self.l2(h, test=test, retain_forward=retain_forward)
        return h
        
    def differentiable_backward(self, g):
        g = self.l2.differentiable_backward(g)
        g = self.l1.differentiable_backward(g)
        g = self.l0.differentiable_backward(g)
        return g

class DCGANEncoder(chainer.Chain):
    def __init__(self, in_ch=3, out_len=128, base_size=128, down_layers=4, use_bn=True, w_init=None):
        layers = {}

        self.down_layers = down_layers
        if use_bn:
            norm = 'bn'
        else:
            norm = None

        act = F.relu
        if w_init is None:
            w_init = chainer.initializers.Normal(0.02)

        layers['c_first'] = NNBlock(in_ch, base_size, nn='down_conv', norm=None, activation=act,  w_init=w_init)
        base = base_size

        for i in range(down_layers-1):
            layers['c'+str(i)] = NNBlock(base, base*2, nn='down_conv', norm=norm, activation=act,  w_init=w_init)
            base*=2

        layers['c_last'] = NNBlock(None, out_len, nn='linear', norm=None, activation=None, w_init=w_init)

        super(DCGANEncoder, self).__init__(**layers)

    def __call__(self, x, test=False, retain_forward=False):
        h = self.c_first(x, test=test, retain_forward=retain_forward)
        for i in range(self.down_layers-1):
            h = getattr(self, 'c'+str(i))(h, test=test, retain_forward=retain_forward)
        _b, _ch, _w, _h = h.data.shape
        self.last_shape=(_b, _ch, _w, _h)
        h = F.reshape(h, (_b, _ch*_w*_h))
        h = self.c_last(h, test=test, retain_forward=retain_forward)
        return h

    def differentiable_backward(self, g):
        g = self.c_last.differentiable_backward(g)
        _b, _ch, _w, _h = self.last_shape
        g = F.reshape(g, (_b, _ch, _w, _h))
        for i in reversed(range(self.down_layers-1)):
            g = getattr(self, 'c'+str(i)).differentiable_backward(g)
        g = self.c_first.differentiable_backward(g)
        return g
