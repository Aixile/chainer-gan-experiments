import numpy as np
import chainer
import os
import glob

from chainer import cuda, optimizers, serializers, Variable
import cv2
from .datasets_base import datasets_base

class aimasu_train(datasets_base):
    def __init__(self, path, img_size=64):
        self._paths = glob.glob(path + "/*.jpg")
        #self._img_size=img_size
        super(aimasu_train, self).__init__(flip=1, resize_to=img_size, crop_to=0, random_brightness=1)

    def __len__(self):
        return len(self._paths)

    def get_example(self, i):
        img = cv2.imread(self._paths[i])
        img = self.do_augmentation(img)
        img = self.preprocess_image(img)
        return img
