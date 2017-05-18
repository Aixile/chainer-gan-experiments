#!/usr/bin/env python

import numpy as np
import math
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, optimizers, serializers, Variable
from chainer import function
from chainer.utils import type_check

def add_noise(h, test, sigma=0.2):
    xp = cuda.get_array_module(h.data)
    if test:
        return h
    else:
        return h + sigma * xp.random.randn(*h.data.shape)

class CBR(chainer.Chain):
    def __init__(self, ch0, ch1, bn=True, sample='down', activation=F.relu, dropout=False, noise=False):
        self.bn = bn
        self.activation = activation
        self.dropout = dropout
        self.sample = sample
        self.noise = noise
        layers = {}
        w = chainer.initializers.Normal(0.02)
        if sample=='down':
            layers['c'] = L.Convolution2D(ch0, ch1, 4, 2, 1, initialW=w)
        elif sample=='none-9':
            layers['c'] = L.Convolution2D(ch0, ch1, 9, 1, 4, initialW=w)
        elif sample=='none-7':
            layers['c'] = L.Convolution2D(ch0, ch1, 7, 1, 3, initialW=w)
        elif sample=='none-5':
            layers['c'] = L.Convolution2D(ch0, ch1, 5, 1, 2, initialW=w)
        elif sample=='linear':
            layers['c'] = L.Linear(ch0, ch1, initialW=w)
        else:
            layers['c'] = L.Convolution2D(ch0, ch1, 3, 1, 1, initialW=w)
        if bn:
            if self.noise:
                layers['batchnorm'] = L.BatchNormalization(ch1, use_gamma=False)
            else:
                layers['batchnorm'] = L.BatchNormalization(ch1)
        super(CBR, self).__init__(**layers)

    def __call__(self, x, test):
        if self.sample=='linear' or self.sample=="down" or self.sample=="none" or self.sample=='none-9' or self.sample=='none-7' or self.sample=='none-5':
            h = self.c(x)
        elif self.sample=="up":
            h = F.unpooling_2d(x, 2, 2, 0, cover_all=False)
            h = self.c(h)
        else:
            print("unknown sample method %s"%self.sample)
        if self.bn:
            h = self.batchnorm(h, test=test)
        if self.noise:
            h = add_noise(h, test=test)
        if self.dropout:
            h = F.dropout(h, train=not test)
        if not self.activation is None:
            h = self.activation(h)
        return h



class Discriminator(chainer.Chain):
    def __init__(self, in_ch=3, n_down_layers=5):
        layers = {}
        w = chainer.initializers.Normal(0.02)
        self.n_down_layers = n_down_layers

        layers['c0'] = CBR(in_ch, 64, bn=False, sample='down', activation=F.leaky_relu, dropout=False, noise=True)
        base = 64

        for i in range(1, n_down_layers):
            layers['c'+str(i)] = CBR(base, base*2, bn=True, sample='down', activation=F.leaky_relu, dropout=False, noise=True)
            base*=2

        layers['c'+str(n_down_layers)] = CBR(base, 1, bn=False, sample='none', activation=None, dropout=False, noise=True)

        super(Discriminator, self).__init__(**layers)

    def __call__(self, x_0, test=False):
        h = self.c0(x_0, test=test)

        for i in range(1, self.n_down_layers+1):
            h = getattr(self, 'c'+str(i))(h, test=test)

        return h

class Generator(chainer.Chain):
    def __init__(self, latent_length=100):
         super(Generator, self).__init__(
            l0 = CBR(latent_length,  4*4*1024, bn=True, sample='linear'),
            c1 = CBR(1024, 512, bn=True, sample='up'), #4->8
            c2 = CBR(512, 256, bn=True, sample='up'), #8->16
            c3 = CBR(256, 128, bn=True, sample='up'), #16->32
            c4 = CBR(128, 3, bn=False, sample='up', activation=F.tanh) #32->64
        )
    def __call__(self, z, test=False):
        #print(z.data.shape)
        h = self.l0(z, test=test)
        h = F.reshape(h, (h.data.shape[0], 1024, 4, 4))
        h = self.c1(h, test=test)
        h = self.c2(h, test=test)
        h = self.c3(h, test=test)
        h = self.c4(h, test=test)
        return h


class DIS(chainer.Chain):

    def __init__(self):
        super(DIS, self).__init__(
            c1=L.Convolution2D(3, 64, 4, 2, 1),
            c2=L.Convolution2D(32, 32, 3, 1, 1),
            c3=L.Convolution2D(32, 64, 4, 2, 1),
            c4=L.Convolution2D(64, 64, 3, 1, 1),
            c5=L.Convolution2D(64, 128, 4, 2, 1),
            c6=L.Convolution2D(128, 128, 3, 1, 1),
            c7=L.Convolution2D(128, 256, 4, 2, 1),
            l8l=L.Linear(None, 1, wscale=0.02 * math.sqrt(8 * 8 * 256)),

            # remove bn in the input layer
            #bnc1=L.BatchNormalization(32),
            bnc2=L.BatchNormalization(32),
            bnc3=L.BatchNormalization(64),
            bnc4=L.BatchNormalization(64),
            bnc5=L.BatchNormalization(128),
            bnc6=L.BatchNormalization(128),
            bnc7=L.BatchNormalization(256),
        )

    def calc(self, x, test=False):
        #h = F.relu(self.bnc1(self.c1(x), test=test))
        h = F.relu(self.c1(x))
        h = F.relu(self.bnc2(self.c2(h), test=test))
        h = F.relu(self.bnc3(self.c3(h), test=test))
        h = F.relu(self.bnc4(self.c4(h), test=test))
        h = F.relu(self.bnc5(self.c5(h), test=test))
        h = F.relu(self.bnc6(self.c6(h), test=test))
        h = F.relu(self.bnc7(self.c7(h), test=test))
        return self.l8l(h)

    def __call__(self, x, t,loss_type='', test=False):
        h = self.calc(x, test)
        if loss_type=='ls':
            loss = F.mean_squared_error(h, t)
        else:
            loss = F.sigmoid_cross_entropy(h, t)

        return loss
