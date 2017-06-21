import numpy as np
import math
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, optimizers, serializers, Variable
from chainer import function
from chainer.utils import type_check
from .ops import *

class DCGANGenerator(chainer.Chain):
    def __init__(self, latent=128, out_ch=3, base_size=1024, use_bn=True, up_layers=4, upsampling='up_deconv'):
        layers = {}

        self.up_layers = up_layers
        self.base_size = base_size
        self.latent = latent

        if use_bn:
            norm = 'bn'
        else:
            norm = None

        w = chainer.initializers.Normal(0.02)

        base = base_size

        layers['c_first'] = NNBlock(latent, 4*4*base, nn='linear', norm=norm, w_init=w)

        for i in range(up_layers-1):
            layers['c'+str(i)] = NNBlock(base, base//2, nn=upsampling, norm=norm, w_init=w)
            base = base//2

        layers['c'+str(up_layers-1)] = NNBlock(base, out_ch, nn=upsampling, norm=None, w_init=w, activation=F.tanh)
        #print(layers)

        super(DCGANGenerator, self).__init__(**layers)

    def __call__(self, z, test=False):
        h = self.c_first(z, test=test)
        h = F.reshape(h, (h.data.shape[0], self.base_size, 4, 4))
        for i in range(self.up_layers):
            h = getattr(self, 'c'+str(i))(h, test=test)
        return h
