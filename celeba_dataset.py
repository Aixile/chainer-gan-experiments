import numpy as np
import chainer
import os

from chainer import cuda, optimizers, serializers, Variable
import cv2

class CelebaDataset(chainer.dataset.DatasetMixin):
    def __init__(self, path, range_min=1, range_max=202600, root='/img_align_celeba/', dtype=np.float32, img_size=64, scaled=True):
        self._paths = [path+root+str(i)+".jpg" for i in range(range_min,range_max)]
        self._dtype = dtype
        self._img_size = img_size
        self._scaled = scaled

    def __len__(self):
        return len(self._paths)

    def __do_image_resize(self, img):
        img = cv2.resize(img[20:198].astype(self._dtype),(self._img_size,self._img_size),interpolation=cv2.INTER_AREA)
        return img

    def get_example(i):
        img = cv2.imread(self._paths[i])
        img = self.__do_image_resize(img)
        if self._scaled:
            img = img/127.5 - 1
        return img.transpose(2, 0, 1)

    def __channels__(self):
        return 3
