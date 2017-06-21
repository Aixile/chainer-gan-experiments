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
