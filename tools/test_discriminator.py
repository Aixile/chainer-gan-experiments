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
from common.models.discriminators import *
from common.utils import *




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='discriminator testing script')
    parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--gen_class',  default='', help='Default generator class')
    parser.add_argument('--dis_class',  default='', help='Default discriminator class')
    parser.add_argument("--load_gen_model", '-l', default='', help='load generator model')
    parser.add_argument("--load_dis_model", default='', help='load discriminator model')
    parser.add_argument('--out', '-o', default='test', help='output image path')
    parser.add_argument("--latent_len", type=int, default=128, help='latent vector length')
    parser.add_argument("--image_channels", type=int, default=3, help='number of image channels')
    parser.add_argument("--image_size", type=int, default=64, help='image size')

    args = parser.parse_args()
    print(args)

    if args.gen_class != '':
        gen = eval(args.gen_class)
    else:
        gen = DCGANGenerator(latent=args.latent_len, out_ch=args.image_channels)

    if args.dis_class != '':
        dis = eval(args.dis_class)
    else:
        dis = DCGANDiscriminator(in_ch=args.image_channels, use_bn=False)


    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
    serializers.load_npz(args.load_gen_model, gen)
    print("Generator model loaded")
    serializers.load_npz(args.load_dis_model, dis)
    print("Discriminator model loaded")


    if args.gpu >= 0:
        gen.to_gpu()
        dis.to_gpu()
        print("use gpu {}".format(args.gpu))

    xp = gen.xp
    z_in = xp.random.normal(size=(36, args.latent_len)).astype("f")
    z = Variable(z_in, volatile=True)
    x = gen(z, test=True)
    result = dis(x, test=True)
    result = F.sigmoid(result)
    prop = xp.ones_like(x.data)
    for i in range(36):
        v = 2*result.data[i] - 1.0
        prop[i,:] = v
    save_images_grid(x, path=args.out+'.img.jpg', grid_w=6, grid_h=6)
    save_images_grid(prop, path=args.out+'.prob.jpg', grid_w=6, grid_h=6)
