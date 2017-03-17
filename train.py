#!/usr/bin/env python

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
import chainer.datasets.image_dataset as ImageDataset
import six
import os

from chainer import cuda, optimizers, serializers, Variable
from chainer import training
from chainer.training import extensions

import argparse
import generator_models
import discriminator_models

from training_visualizer import test_sampling

def main():
    parser = argparse.ArgumentParser(description='Chainer GAN Playground')

    parser.add_argument('--batchsize', '-b', type=int, default=32,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--dataset', '-i', default='',
                        help='Directory of image files.')

    parser.add_argument('--output_dir', '-o', default='result/',
                        help='Directory to output the result')

    parser.add_argument('--test_samples_output_dir', '-t', default='test/',
                        help='Directory to output test samples')

    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')

    parser.add_argument('--snapshot_interval', type=int, default=10000,
                        help='Interval of snapshot')
    parser.add_argument('--display_interval', type=int, default=100,
                        help='Interval of displaying log to console')
    parser.add_argument('--test_interval', type=int, default=1000,
                        help='Interval of testing generator')

    parser.add_argument('--gan', default='', help='Type of adversarial loss, dcgan loss for the default')

    parser.add_argument('--gen_class', default='GEN_64', help='Default generator class')
    parser.add_argument('--dis_class', default='DIS_64', help='Default discriminator class')
    parser.add_argument("--load_gen_model", default='', help='Load the generator model')
    parser.add_argument("--load_dis_model", default='', help='Load the discriminator model')

    parser.add_argument('--gen_dis_update_radio', type=int, default=1, help='Generator/Discriminator update radio')

    parser.add_argument("--use_tanh_activation", dest='use_tanh_activation', action="store_true")
    parser.set_defaults(use_tanh_activation=False)

    parser.add_argument("--use_rmsprop", dest='use_rmsprop', action="store_true")
    parser.set_defaults(use_tanh_activation=False)

    parser.add_argument("--learning_rate_g", type=float, default=0.0001, help="Learning rate for generator")
    parser.add_argument("--learning_rate_d", type=float, default=0.0001, help="Learning rate for discriminator")

    parser.add_argument("--weight_decay", type=float, default=0, help='Weight decay value')

    args = parser.parse_args()
    print(args)

    np.random.seed(args.seed)

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    dataset=CelebaDataset(self.dataset, scaled=args.use_tanh_activation)

    gen = getattr(generator_models, args.gen_class)(channel_out=dataset.__channels__, use_tanh_activation=args.use_tanh_activation)
    if args.load_gen_model != '':
        serializers.load_npz(args.load_gen_model, cnn)
        print("Generator model loaded")

    dis = getattr(discriminator_models, args.dis_class)(channel_in=dataset.__channels__)
    if args.load_dis_model != '':
        serializers.load_npz(args.load_dis_model, dis)
        print("Discriminator model loaded")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(args.test_samples_output_dir):
        os.makedirs(args.test_samples_output_dir)


    train_iter = chainer.iterators.MultiprocessIterator(dataset, args.batchsize,n_prefetch=args.batch_size)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        gen.to_gpu()
        dis.to_gpu()

    if args.use_rmsprop:
        opt = optimizers.RMSprop(alpha=self.learning_rate_g)
    else:
        opt = optimizers.Adam(alpha=self.learning_rate_g, beta1=0.5)
    opt.setup(gen)

    if args.use_rmsprop:
        opt_d = optimizers.RMSprop(alpha=self.learning_rate_d)
    else:
        opt_d = optimizers.Adam(alpha=self.learning_rate_d, beta1=0.5)
    opt_d.setup(dis)

    if args.weight_decay>0:
        opt.add_hook(chainer.optimizer.WeightDecay(args.weight_decay), 'hook_gen')
        opt_d.add_hook(chainer.optimizer.WeightDecay(args.weight_decay), 'hook_dec')

    updater = ganUpdater(
        models=(gen, dis),
        iterator={
            'main': train_iter,
        },
        optimizer={
            'gen': opt,
            'dis': opt_d},
        device=args.gpu,
        params={
            'gen_dis_update_radio': args.gen_dis_update_radio,
            'adv_loss_type': args.gan
        }
    )

    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    snapshot_interval = (args.snapshot_interval, 'iteration')
    snapshot_interval2 = (args.snapshot_interval * 2, 'iteration')

    test_visualizer_interval = (args.test_interval , 'iteration')

    trainer.extend(extensions.dump_graph('gen/loss'))
    trainer.extend(extensions.snapshot(), trigger=snapshot_interval2)
    trainer.extend(extensions.snapshot_object(
        gen, 'gen_iter_{.updater.iteration}'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        dis, 'dis_iter_{.updater.iteration}'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        opt, 'optimizer_'), trigger=snapshot_interval)
    trainer.extend(extensions.LogReport(trigger=(10, 'iteration'), ))
    trainer.extend(extensions.PrintReport(
        ['epoch',  'gen/loss', 'dis/loss']))
    trainer.extend(extensions.ProgressBar(update_interval=20))

    trainer.extend(
        test_sampling(updater, gen, args.test_samples_output_path,
        scaled=args.use_tanh_activation,
        ), trigger=test_visualizer_interval
    )

    trainer.run()

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    # Save the trained model
    chainer.serializers.save_npz(os.path.join(out_dir, 'generator_final'), gen)
    chainer.serializers.save_npz(os.path.join(out_dir, 'discriminator_final'), dis)
    chainer.serializers.save_npz(os.path.join(out_dir, 'optimizer_final'), opt)

if __name__ == '__main__':
    main()
