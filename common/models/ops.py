import numpy as np
import math
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, optimizers, serializers, Variable
from chainer import function
from chainer.utils import type_check
from .backwards import *

def add_noise(h, test, sigma=0.2):
    xp = cuda.get_array_module(h.data)
    if test:
        return h
    else:
        return h + sigma * xp.random.randn(*h.data.shape)

def selu(x):
    alpha = float(1.6732632423543772848170429916717)
    scale = float(1.0507009873554804934193349852946)
    return  scale * F.elu(x, alpha = alpha)

def weight_clipping(model, lower=-0.01, upper=0.01):
    for params in self.params():
        params_clipped = F.clip(params, lower, upper)
        params.data = params_clipped.data

class ResBlock(chainer.Chain):
    def __init__(self, ch, bn=True, activation=F.relu, k_size=3):
        self.bn = bn
        self.activation = activation
        layers = {}
        pad = k_size//2
        layers['c0'] = L.Convolution2D(ch, ch, 3, 1, pad)
        layers['c1'] = L.Convolution2D(ch, ch, 3, 1, pad)
        if bn:
            layers['bn0'] = L.BatchNormalization(ch)
            layers['bn1'] = L.BatchNormalization(ch)
        super(ResBlock, self).__init__(**layers)

    def __call__(self, x, test):
        h = self.c0(x)
        if self.bn:
            h = self.bn0(h, test=test)
        h = self.activation(h)
        h = self.c1(x)
        if self.bn:
            h = self.bn1(h, test=test)
        return h + x

class NNBlock(chainer.Chain):
    def __init__(self, ch0, ch1, \
                nn='conv', \
                norm='bn', \
                activation=F.relu, \
                dropout=False, \
                noise=None, \
                w_init=None, \
                k_size = 3, \
                normalize_input=False ):

        self.norm = norm
        self.normalize_input = normalize_input
        self.activation = activation
        self.dropout = dropout
        self.noise = noise
        self.nn = nn
        layers = {}

        if w_init == None:
            w = chainer.initializers.GlorotNormal()
        else:
            w = w_init

        if nn == 'down_conv':
            layers['c'] = L.Convolution2D(ch0, ch1, 4, 2, 1, initialW=w)

        elif nn == 'up_deconv':
            layers['c'] = L.Deconvolution2D(ch0, ch1, 4, 2, 1, initialW=w)

        elif nn == 'up_subpixel':
            pad = k_size//2
            layers['c'] = L.Convolution2D(ch0, ch1*4, k_size, 1, pad, initialW=w)

        elif nn=='conv' or nn=='up_unpooling':
            pad = k_size//2
            layers['c'] = L.Convolution2D(ch0, ch1, k_size, 1, pad, initialW=w)

        elif nn=='linear':
            layers['c'] = L.Linear(ch0, ch1, initialW=w)

        else:
            raise Exception("Cannot find method %s" % nn)

        if self.norm == 'bn':
            if self.noise:
                layers['n'] = L.BatchNormalization(ch1, use_gamma=False)
            else:
                layers['n'] = L.BatchNormalization(ch1)
        elif self.norm == 'ln':
                layers['n'] = L.LayerNormalization(ch1)

        super(NNBlock, self).__init__(**layers)

    def _do_normalization(self, x, test):
        if self.norm == 'bn':
            return self.n(x, test=test)
        elif self.norm == 'ln':
            return self.n(x)
        else:
            return x

    def _do_before_cal(self, x):
        if self.nn == 'up_unpooling':
            x = F.unpooling_2d(x, 2, 2, 0, cover_all=False)
        return x

    def _do_after_cal(self, x, test):
        if self.nn == 'up_subpixel':
            x = F.depth2space(self.c2(x), 2)
        if self.noise:
            x = add_noise(x, test=test)
        if self.dropout:
            x = F.dropout(x, train=not test)
        return x

    def __call__(self, x, test, retain_forward=False):
        if self.normalize_input:
            x = self._do_normalization(x, test)

        x = self._do_before_cal(x)
        x = self.c(x)

        if  not self.norm is None and not self.normalize_input:
            x = self._do_normalization(x, test)
        x = self._do_after_cal(x, test)

        if not self.activation is None:
            x = self.activation(x)

        if retain_forward:
            self.x = x
        return x

    def differentiable_backward(self, g):
        if not self.norm is None:
            raise NotImplementedError

        if self.activation is F.leaky_relu:
            g = backward_leaky_relu(self.x, g)
        elif self.activation is F.relu:
            g = backward_relu(self.x, g)
        elif self.activation is F.tanh:
            g = backward_tanh(self.x, g)
        elif self.activation is F.sigmoid:
            g = backward_sigmoid(self.x, g)
        elif not self.activation is None:
            raise NotImplementedError

        if self.nn == 'down_conv' or self.nn == 'conv':
            g = backward_convolution(None, g, self.c)
        elif self.nn == 'linear':
            g = backward_linear(None, g, self.c)
        elif self.nn == 'up_deconv':
            g = backward_deconvolution(None, g, self.c)
        else:
            raise NotImplementedError

        return g
