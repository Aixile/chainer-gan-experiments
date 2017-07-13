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
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--gen_class',  default='', help='Default generator class')
    parser.add_argument("--load_gen_model", '-l', default='', help='load generator model')
    parser.add_argument('--out', '-o', default='test.jpg', help='output image path')
    parser.add_argument("--image_channels", type=int, default=3, help='number of image channels')
    parser.add_argument("--image_size", type=int, default=64, help='image size')
    parser.add_argument("--latent_len", type=int, default=128, help='latent vector length')
    parser.add_argument("--attr_len", type=int, default=38, help='attribute vector length')
    args = parser.parse_args()
    print(args)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()

    if args.gen_class != '':
        gen = eval(args.gen_class)
    else:
        gen = DCGANGenerator(latent=args.latent_len+args.attr_len, out_ch=args.image_channels)

    if args.load_gen_model != '':
        serializers.load_npz(args.load_gen_model, gen)
        print("Generator model loaded")

    if args.gpu >= 0:
        gen.to_gpu()
        print("use gpu {}".format(args.gpu))

    xp = gen.xp

    def gen_noise():
        return xp.random.normal(size=(6*6, args.latent_len)).astype("f")

    def parse_tags(t):
        ans = xp.zeros((6*6, len(t))).astype("f")
        ans = ans.transpose((1, 0))
        for i in range(len(t)):
            if t[i][1]>0:
                ans[i,:]=1.0
            else:
                ans[i,:]=-1.0
        return ans.transpose((1, 0))

    def get_fake_tag():
        prob2 = np.random.rand(args.attr_len)
        tags = np.zeros((args.attr_len)).astype("f")
        tags[:] = -1.0
        tags[np.argmax(prob2[0:13])]=1.0
        tags[27 + np.argmax(prob2[27:])] = 1.0
        prob2[prob2<0.75] = -1.0
        prob2[prob2>=0.75] = 1.0
        for i in range(13, 27):
            tags[i] = prob2[i]
        #tags[:] = -1
        return tags

    def get_fake_tag_batch():
        tags = xp.zeros((36, args.attr_len)).astype("f")
        d = xp.asarray(get_fake_tag())
        for i in range(36):
            tags[i] = d#xp.asarray(get_fake_tag())
        return tags

    tags = [["blonde hair", -1],
            ["brown hair", -1],
            ["black hair", -1],
            ["blue hair", -1],
            ["pink hair", -1],
            ["purple hair", -1],
            ["green hair", -1],
            ["red hair", -1],
            ["silver hair", -1],
            ["white hair", 1],
            ["orange hair", -1],
            ["aqua hair", -1],
            ["grey hair", -1],
            ["long hair", 1],
            ["short hair", -1],
            ["twintails", -1],
            ["very long hair", 1],
            ["drill hair", -1],
            ["ponytail", -1],
            ["side ponytail", -1],
            ["blush", -1],
            ["smile", 1],
            ["open mouth", 1],
            ["hat", -1],
            ["ribbon", -1],
            ["glasses", -1],
            ["lips", -1],
            ["blue eyes", -1],
            ["red eyes", 1],
            ["brown eyes", -1],
            ["green eyes", -1],
            ["purple eyes", -1],
            ["yellow eyes", -1],
            ["pink eyes", -1],
            ["aqua eyes", -1],
            ["black eyes", -1],
            ["orange eyes", -1],
            ["grey eyes", -1]]

    z = gen_noise()
    t = get_fake_tag_batch()#parse_tags(tags)
    for i in range(len(tags)):
        if t[0][i] > 0:
            print(tags[i][0])

    x_out = gen(F.concat([Variable(z, volatile=True),Variable(t, volatile=True)]), test=True)
    save_images_grid(x_out, args.out, 6, 6)
