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
#from common.models.discriminators import *
#from common.utils import *
os.environ["DEBUG"] = "1"
#os.environ["OPTIMIZE"] = "0"
from webdnn.frontend.chainer import ChainerConverter
from webdnn.backend.interface.generator import generate_descriptor

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='discriminator testing script')
    parser.add_argument('--gen_class',  default='', help='Default generator class')
    parser.add_argument("--load_gen_model", '-l', default='', help='load generator model')
    parser.add_argument('--out', '-o', default='test.jpg', help='output image path')
    parser.add_argument("--image_channels", type=int, default=3, help='number of image channels')
    parser.add_argument("--image_size", type=int, default=64, help='image size')
    parser.add_argument("--latent_len", type=int, default=128, help='latent vector length')
    parser.add_argument("--attr_len", type=int, default=38, help='attribute vector length')
    args = parser.parse_args()
    print(args)

    if args.gen_class != '':
        gen = eval(args.gen_class)
    else:
        gen = DCGANGenerator(latent=args.latent_len+args.attr_len, out_ch=args.image_channels)

    if args.load_gen_model != '':
        serializers.load_npz(args.load_gen_model, gen)
        print("Generator model loaded")
    np.random.seed(0)
    x =  chainer.Variable(np.empty((1, args.latent_len+args.attr_len), dtype=np.float32))
    y = gen(x, test=True)
    graph = ChainerConverter().convert_from_inout_vars([x], [y])
    exec_info = generate_descriptor("webassembly", graph)
    exec_info.save("./output_model")
