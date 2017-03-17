import numpy as np
import math
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, optimizers, serializers, Variable


from chainer import function
from chainer.utils import type_check

class GEN_64(chainer.Chain):

    def __init__(self, latent_length=100, channel_out=3 ,use_tanh_activation=True):
        super(GEN_64, self).__init__(
            l0 = L.Linear(latent_length, 4*4*1024，wscale=0.02*math.sqrt(latent_length)), #4*4
            dc1 = L.Deconvolution2D(1024, 512, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*1024)), #8*8
            dc2 = L.Deconvolution2D(512, 256, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*512)), #16*16
            dc3 = L.Deconvolution2D(256, 128, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*256)), #32*32
            dc4 = L.Deconvolution2D(128, channel_out, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*128)), #64*64
            bnl0 = L.BatchNormalization(4*4*1024),
            bn1 = L.BatchNormalization(512),
            bn2 = L.BatchNormalization(256),
            bn3 = L.BatchNormalization(128),
        )
        self._latent_length=latent_length
        self._use_tanh_activation = use_tanh_activation

    def __call__(self, z, test=False):
        h = self.bnl0(self.l0(z), test=test)
        h = F.reshape(h, (z.data.shape[0], 1024, 4, 4))
        h = F.relu(self.bn1(self.dc1(h), test=test))
        h = F.relu(self.bn2(self.dc2(h), test=test))
        h = F.relu(self.bn3(self.dc3(h), test=test))
        x = self.dc4(h)
        if self._use_tanh_activation:
            x = F.tanh(x)
        return x

    def gen_samples(self, cnt):
        z_rnd = self.xp.random.normal(size=(cnt,self._latent_length)).astype("f")
        return self.__call__(z_rnd, test=True)
