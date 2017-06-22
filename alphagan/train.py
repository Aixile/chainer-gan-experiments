import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, optimizers, serializers, Variable
from chainer import training
from chainer.training import extensions
import argparse
import sys
sys.path.insert(0, '../')

import common.datasets as datasets
from common.models.generators import *
from common.models.discriminators import *
from common.models.models import *
from common.utils import *
from updater import *
import settings

def main():
    parser = argparse.ArgumentParser(
        description='Train Alpha-GAN')
    parser.add_argument('--batch_size', '-b', type=int, default=64)
    parser.add_argument('--max_iter', '-m', type=int, default=50000)
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--eval_interval', type=int, default=200,
                        help='Interval of evaluating generator')

    parser.add_argument("--learning_rate_g", type=float, default=0.0002,
                        help="Learning rate for generator/encoder")
    parser.add_argument("--learning_rate_d", type=float, default=0.0002,
                        help="Learning rate for discriminator")

    parser.add_argument("--load_gen_model", default='', help='load generator model')
    parser.add_argument("--load_enc_model", default='', help='load encoder model')
    parser.add_argument("--load_dis_x_model", default='', help='load image discriminator model')
    parser.add_argument("--load_dis_z_model", default='', help='load code discriminator model')

    parser.add_argument("--lambda_l1", type=float, default=10, help='lambda for l1 loss')

    parser.add_argument("--image_size", type=int, default=64, help='image size')
    parser.add_argument("--image_channels", type=int, default=3, help='number of image channels')
    parser.add_argument("--latent_len", type=int, default=128, help='latent vector length')

    parser.add_argument("--load_dataset", default='celeba_train', help='load dataset')
    parser.add_argument("--dataset_path", "-d", default="/home/aixile/Workspace/dataset/celeba/",
                        help='dataset directory')

    args = parser.parse_args()
    print(args)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()

    gen = DCGANGenerator(latent=args.latent_len, out_ch=args.image_channels)
    enc = DCGANEncoder(out_len=args.latent_len, in_ch=args.image_channels)
    dis_x = DCGANDiscriminator(in_ch=args.image_channels, noise_all_layers=False, conv_as_last=False)
    dis_z = ThreeLayersMLP()

    if args.load_gen_model != '':
        serializers.load_npz(args.load_gen_model, gen)
        print("Generator model loaded")

    if args.load_enc_model != '':
        serializers.load_npz(args.load_dis_model, enc)
        print("Encoder model loaded")

    if args.load_dis_x_model != '':
        serializers.load_npz(args.load_dis_x_model, dis_x)
        print("Image discriminator model loaded")

    if args.load_enc_model != '':
        serializers.load_npz(args.load_dis_z_model, dis_z)
        print("Code discriminator model loaded")

    if args.gpu >= 0:
        gen.to_gpu()
        enc.to_gpu()
        dis_x.to_gpu()
        dis_z.to_gpu()
        print("use gpu {}".format(args.gpu))

    opt_gen = make_adam(gen, lr=args.learning_rate_g, beta1=0.5)
    opt_enc= make_adam(enc, lr=args.learning_rate_g, beta1=0.5)
    opt_dis_x = make_adam(dis_x, lr=args.learning_rate_d, beta1=0.5)
    opt_dis_z = make_adam(dis_z, lr=args.learning_rate_d, beta1=0.5)

    train_dataset = getattr(datasets, args.load_dataset)(path=args.dataset_path)
    train_iter = chainer.iterators.MultiprocessIterator(
        train_dataset, args.batch_size, n_processes=4)

    updater = Updater(
        models=(gen, enc, dis_x, dis_z),
        iterator={
            'main': train_iter,
        },
        optimizer={
            'gen': opt_gen,
            'enc': opt_enc,
            'dis_x': opt_dis_x,
            'dis_z': opt_dis_z
        },
        device=args.gpu,
        params={
            'batch_size': args.batch_size,
            'img_size': args.image_size,
            'img_chan': args.image_channels,
            'latent_len': args.latent_len,
            'lambda_l1': args.lambda_l1,
        },
    )

    trainer = training.Trainer(updater, (args.max_iter, 'iteration'), out=args.out)

    model_save_interval = (4000, 'iteration')
    eval_interval = (args.eval_interval, 'iteration')

    trainer.extend(extensions.snapshot_object(
        gen, 'gen_{.updater.iteration}.npz'), trigger=model_save_interval)
    trainer.extend(extensions.snapshot_object(
        enc, 'enc_{.updater.iteration}.npz'), trigger=model_save_interval)
    trainer.extend(extensions.snapshot_object(
        dis_x, 'dis_x_{.updater.iteration}.npz'), trigger=model_save_interval)
    trainer.extend(extensions.snapshot_object(
        dis_z, 'dis_z_{.updater.iteration}.npz'), trigger=model_save_interval)

    log_keys = ['epoch', 'iteration', 'gen/loss', 'enc/loss', 'dis_x/loss', 'dis_z/loss']
    trainer.extend(extensions.LogReport(keys=log_keys, trigger=(20, 'iteration')))
    trainer.extend(extensions.PrintReport(log_keys), trigger=(20, 'iteration'))
    trainer.extend(extensions.ProgressBar(update_interval=50))

    trainer.extend(
        Evaluation(gen, args.eval_folder, args.gpu, latent_len=args.latent_len), trigger=eval_interval
    )

    trainer.run()


if __name__ == '__main__':
    main()
