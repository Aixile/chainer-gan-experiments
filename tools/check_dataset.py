import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, optimizers, serializers, Variable
from chainer import training
from chainer.training import extensions
import argparse
import sys
import json
sys.path.insert(0, '../')

import common.datasets as datasets
from common.models.generators import *
from common.models.discriminators import *
from common.utils import *
import settings

def main():
    parser = argparse.ArgumentParser(
        description='Check Dataset')
    parser.add_argument("--load_dataset", default='game_faces_tags_train', help='load dataset')
    parser.add_argument("--dataset_path", "-d", default=settings.GAME_FACE_PATH,
                        help='dataset directory')

    args = parser.parse_args()
    print(args)
    train_dataset = getattr(datasets, args.load_dataset)(path=args.dataset_path)
    with open(args.dataset_path + "/attr.json") as f:
        attr = json.load(f)
    for i in range(10):
        print("Image %d:" % i)
        img, tags = train_dataset.get_example(i)
        save_single_image(img, "samples_"+str(i)+".jpg")
        for j in range(len(attr)):
            print(j, end=" ")
            print(attr[j][0],end=" ")
            print(tags[j])
        print("")


if __name__ == '__main__':
    main()
