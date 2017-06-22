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
from common.utils import *
from updater import *
import settings

def main():
    parser = argparse.ArgumentParser(
        description='Train GAN')
    parser.add_argument('--batch_size', '-b', type=int, default=64)
    parser.add_argument('--max_iter', '-m', type=int, default=60000)
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--eval_interval', '-e', type=int, default=200,
                        help='Interval of evaluating generator')

    parser.add_argument("--learning_rate_g", type=float, default=0.0002,
                        help="Learning rate for generator")
    parser.add_argument("--learning_rate_d", type=float, default=0.0002,
                        help="Learning rate for discriminator")

    parser.add_argument('--gen_class', default='', help='generator class')
    parser.add_argument('--dis_class', default='', help='discriminator class')

    parser.add_argument("--load_gen_model", default='', help='load generator model')
    parser.add_argument("--load_dis_model", default='', help='load discriminator model')

    parser.add_argument("--lambda_gp", type=float, default=10, help='gradient penalty')

    parser.add_argument("--image_size", type=int, default=64, help='image size')
    parser.add_argument("--image_channels", type=int, default=3, help='number of image channels')
    parser.add_argument("--latent_len", type=int, default=128, help='latent vector length')

    parser.add_argument("--load_dataset", default='celeba_train', help='load dataset')
    parser.add_argument("--dataset_path", "-d", default=settings.CELEBA_PATH,
                        help='dataset directory')

    args = parser.parse_args()
    print(args)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()

    if args.gen_class != '':
        gen = eval(args.gen_class)
    else:
        gen = DCGANGenerator(latent=args.latent_len, out_ch=args.image_channels)

    if args.dis_class != '':
        dis = eval(args.dis_class)
    else:
        dis = DCGANDiscriminator(in_ch=args.image_channels, use_bn=False)

    if args.load_gen_model != '':
        serializers.load_npz(args.load_gen_model, gen)
        print("Generator model loaded")

    if args.load_dis_model != '':
        serializers.load_npz(args.load_dis_model, dis)
        print("Discriminator model loaded")

    if args.gpu >= 0:
        gen.to_gpu()
        dis.to_gpu()
        print("use gpu {}".format(args.gpu))

    opt_g = make_adam(gen, lr=args.learning_rate_g, beta1=0.5)
    opt_d = make_adam(dis, lr=args.learning_rate_d, beta1=0.5)

    train_dataset = getattr(datasets, args.load_dataset)(path=args.dataset_path)
    train_iter = chainer.iterators.MultiprocessIterator(
        train_dataset, args.batch_size, n_processes=4)

    updater = Updater(
        models=(gen, dis),
        iterator={
            'main': train_iter,
        },
        optimizer={
            'gen': opt_g,
            'dis': opt_d},
        device=args.gpu,
        params={
            'batch_size': args.batch_size,
            'img_size': args.image_size,
            'img_chan': args.image_channels,
            'lambda_gp': args.lambda_gp,
            'latent_len': args.latent_len,
        },
    )

    trainer = training.Trainer(updater, (args.max_iter, 'iteration'), out=args.out)

    model_save_interval = (4000, 'iteration')
    eval_interval = (args.eval_interval, 'iteration')

    trainer.extend(extensions.snapshot_object(
        gen, 'gen_{.updater.iteration}.npz'), trigger=model_save_interval)
    trainer.extend(extensions.snapshot_object(
        dis, 'dis_{.updater.iteration}.npz'), trigger=model_save_interval)

    log_keys = ['epoch', 'iteration', 'gen/loss', 'dis/loss', 'dis/loss_gp']
    trainer.extend(extensions.LogReport(keys=log_keys, trigger=(20, 'iteration')))
    trainer.extend(extensions.PrintReport(log_keys), trigger=(20, 'iteration'))
    trainer.extend(extensions.ProgressBar(update_interval=50))

    trainer.extend(
        gan_sampling(gen, args.out+"/preview/", args.gpu), trigger=eval_interval
    )

    trainer.run()


if __name__ == '__main__':
    main()
