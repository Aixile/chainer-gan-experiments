import numpy as np
import chainer
import os
import glob
import pandas
import pickle
from chainer import cuda, optimizers, serializers, Variable
import cv2
import datetime
import json
import pickle
from .datasets_base import datasets_base

class game_faces_train(datasets_base):
    def __init__(self, path, img_size=64, flip=1, crop_to=0, random_brightness=0):
        self._paths = glob.glob(path + "/images/*.jpg")
        self._datapath = path
        with open(path + "/games.pickle", 'rb') as f:
            self._gamedata=pickle.load(f)
        #self._img_size=img_size
        super(game_faces_train, self).__init__(flip=1, resize_to=img_size, crop_to=crop_to, random_brightness=random_brightness)

    def __len__(self):
        return len(self._paths)

    def get_getchu_id(self, path):
        path = os.path.basename(path)
        ans = path.split("chara")
        return int(ans[0])

    def try_get_example(self):
        id = np.random.randint(0, self.__len__())
        path = self._paths[id]
        g_id = self.get_getchu_id(path)
        if self._gamedata.loc[self._gamedata['getchu_id']==g_id].iloc[0]['sellday']<datetime.date(2007,1,1):
            return None
        img = cv2.imread(path)
        if img.shape[0] < 64:
            return None
        return img

    def get_example(self, i):
        np.random.seed(None)
        while True:
            img = self.try_get_example()
            if not img is None:
                break
        img = self.do_augmentation(img)
        img = self.preprocess_image(img)
        return img


class game_faces_tags_train(datasets_base):
    def __init__(self, path, img_size=64, flip=1, crop_to=0, random_brightness=0, threshold=0.25):
        self._paths = glob.glob(path + "/images/*.jpg")
        self._datapath = path
        with open(path + "/games.pickle", 'rb') as f:
            self._gamedata=pickle.load(f)
        with open(path + "/tags.pickle", 'rb') as f:
            self._tags = pickle.load(f)
        with open(path + "/attr.json") as f:
            self._attr = json.load(f)
        self._len_attr = len(self._attr)
        self._threshold = threshold
        super(game_faces_tags_train, self).__init__(flip=1, resize_to=img_size, crop_to=crop_to, random_brightness=random_brightness)

    def __len__(self):
        return len(self._paths)

    def get_getchu_id(self, path):
        path = os.path.basename(path)
        ans = path.split("chara")
        return int(ans[0])

    def get_tags(self, path):
        path = os.path.basename(path)
        prob = self._tags[path]
        prob2 = np.zeros((self._len_attr))
        for i in range(self._len_attr):
            prob2[i] = prob[self._attr[i][1]]

        tags = np.zeros((self._len_attr))
        tags[np.argmax(prob2[0:13])]=1.0
        tags[27 + np.argmax(prob2[27:])] = 1.0
        prob2[prob2<self._threshold] = 0.0
        prob2[prob2>=self._threshold] = 1.0
        for i in range(13, 27):
            tags[i] = prob2[i]
        return tags

    def try_get_example(self):
        id = np.random.randint(0, self.__len__())
        path = self._paths[id]
        g_id = self.get_getchu_id(path)
        if self._gamedata.loc[self._gamedata['getchu_id']==g_id].iloc[0]['sellday']<datetime.date(2007,1,1):
            return None, None
        img = cv2.imread(path)
        if img.shape[0] < 80:
            return None, None
        tags = self.get_tags(path)
        return img, tags

    def get_example(self, i):
        np.random.seed(None)
        while True:
            img, tags = self.try_get_example()
            if not img is None:
                break
        img = self.do_augmentation(img)
        img = self.preprocess_image(img)
        return img, tags
