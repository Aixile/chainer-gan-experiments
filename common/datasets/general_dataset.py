import numpy as np
import chainer
import os
import glob

from chainer import cuda, optimizers, serializers, Variable
import cv2
from .datasets_base import datasets_base

class general_train(datasets_base):
    def __init__(self, path, img_size=64, flip=1, crop_to=0, random_brightness=0):
        self._paths = glob.glob(path + "/*.jpg")
        #self._img_size=img_size
        super(general_train, self).__init__(flip=1, resize_to=img_size, crop_to=crop_to, random_brightness=random_brightness)

    def __len__(self):
        return len(self._paths)

    def get_example(self, i):
        img = cv2.imread(self._paths[i])
        img = self.do_augmentation(img)
        img = self.preprocess_image(img)
        return img
