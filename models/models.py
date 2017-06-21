import numpy as np
import math
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, optimizers, serializers, Variable
from chainer import function
from chainer.utils import type_check
from models.ops import *

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

class DCGANDiscriminator(chainer.Chain):
    def __init__(self, in_ch=3, base_size=64, down_layers=4, use_bn=True, noise_all_layer=False, conv_as_last=False):
        layers = {}

        self.down_layers = down_layers
        self.conv_as_last = conv_as_last

        if use_bn:
            norm = 'bn'
        else:
            norm = None

        act = F.leaky_relu
        w = chainer.initializers.Normal(0.02)

        layers['c_first'] = NNBlock(in_ch, base_size, nn='conv', norm=None, activation=act, noise=noise_all_layer, w_init=w)
        base = base_size

        for i in range(down_layers):
            layers['c'+str(i)] = NNBlock(base, base*2, nn='down_conv', norm=norm, activation=act, noise=noise_all_layer, w_init=w)
            base*=2

        if conv_as_last:
            layers['c_last'] = NNBlock(base, 1, nn='conv', norm=None, activation=None, w_init=w)
        else:
            layers['c_last'] = NNBlock(None, 1, nn='linear', norm=None, activation=None, w_init=w)

        super(DCGANDiscriminator, self).__init__(**layers)

    def __call__(self, x, test=False):
        h = self.c_first(x, test=test)
        for i in range(self.down_layers):
            h = getattr(self, 'c'+str(i))(h, test=test)
        if self.conv_as_last:
            h = self.c_last(h, test=test)
        else:
            _b, _ch, _w, _h = h.data.shape
            h = F.reshape(h, (_b, _ch*_w*_h))
            h = self.l0(h, test=test)
        return h

class DCGANGenerator(chainer.Chain):
    def __init__(self, latent_length=128, out_ch=3, base_size=1024, use_bn=True, up_layers=4, upsampling='up_deconv'):
        layers = {}
        self.up_layers = up_layers
        self.base_size = base_size
        self.latent_length = latent_length

        if use_bn:
            norm = 'bn'
        else:
            norm = None

        w = chainer.initializers.Normal(0.02)

        layers['c_first'] = NNBlock(latent_length, 4*4*base_size, nn='linear', norm=norm, w_init=w),

        for i in range(up_layers-1):
            layers['c'+str(i)] = NNBlock(base, base//2, nn=upsampling, norm=norm, w_init=w)
            base = base//2

        layers['c'+str(up_layers-1)] = NNBlock(base, out_ch, nn=upsampling, norm=None, w_init=w, activation=F.tanh)

        super(DCGANGenerator, self).__init__(**layers)

    def __call__(self, z, test=False):
        h = self.c_first(z, test=test)
        h = F.reshape(h, (h.data.shape[0], base_size, 4, 4))
        for i in range(self.up_layers):
            h = getattr(self, 'c'+str(i))(h, test=test)
        return h
