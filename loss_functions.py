import numpy as np
import math
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, optimizers, serializers, Variable
from chainer import function
from chainer.utils import type_check

def LOSS_MSE(h, t):
    return F.mean_squared_error(h, t)

def LOSS_SCE(h, t):
    return F.sigmoid_cross_entropy(h, t)

def LOSS_L1(h, t):
    return F.mean_absolute_error(h, t)
