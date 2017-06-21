import numpy as np
import chainer
import os
import glob

from chainer import cuda, optimizers, serializers, Variable
import cv2
from .datasets_base import datasets_base

class mnist_train(datasets_base):
    def __init__(self, path=None, img_size=64):
        self.train, _ = chainer.datasets.get_mnist(withlabel=False, ndim=2, scale=255)
        super(mnist_train, self).__init__(flip=1, resize_to=img_size, crop_to=0)

    def __len__(self):
        return self.train.__len__()

    def do_resize(self, img, resize_to=32):
        return cv2.resize(img, (resize_to, resize_to), interpolation=cv2.INTER_AREA)

    def get_example(self, i):
        img = self.train.__getitem__(i)
        img = self.do_resize(img)
        img = np.expand_dims(img, axis=2)
        img = self.preprocess_image(img)
        return img
