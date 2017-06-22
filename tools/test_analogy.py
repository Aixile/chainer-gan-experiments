import os
import numpy as np
import chainer
import chainer.cuda
from chainer import cuda, serializers, Variable
from chainer import training
import chainer.functions as F
import argparse

import sys
sys.path.insert(0, '../')
from common.models.generators import *
from common.utils import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='gan analogy testing script')
    parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--gen_class', '-c', default='', help='Default generator class')
    parser.add_argument("--load_gen_model", '-l', default='', help='load generator model')
    parser.add_argument('--out', '-o', default='output.jpg', help='output image path')
    parser.add_argument("--latent_len", type=int, default=128, help='latent vector length')
    parser.add_argument("--image_channels", type=int, default=3, help='number of image channels')

    args = parser.parse_args()
    print(args)

    if args.gen_class != '':
        gen = eval(args.gen_class)
    else:
        gen = DCGANGenerator(latent=args.latent_len, out_ch=args.image_channels)


    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
    serializers.load_npz(args.load_gen_model, gen)
    print("Generator model loaded")

    if args.gpu >= 0:
        gen.to_gpu()
        print("use gpu {}".format(args.gpu))
    analogy(gen, args.out, latent_len=args.latent_len)
