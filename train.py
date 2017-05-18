import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, optimizers, serializers, Variable
from chainer import training
from chainer.training import extensions
import argparse

from models import Generator, Discriminator
from updater import Updater
from evaluation import Evaluation
import datasets

def main():
    parser = argparse.ArgumentParser(
        description='Train GAN')
    parser.add_argument('--batch_size', '-b', type=int, default=32)
    parser.add_argument('--max_iter', '-m', type=int, default=120000)
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--eval_folder', '-e', default='test',
                        help='Directory to output the evaluation result')
    parser.add_argument('--eval_interval', type=int, default=200,
                        help='Interval of evaluating generator')

    parser.add_argument("--learning_rate_g", type=float, default=0.0002,
                        help="Learning rate for generator")
    parser.add_argument("--learning_rate_d", type=float, default=0.0002,
                        help="Learning rate for discriminator")

    parser.add_argument("--load_gen_model", default='', help='load generator model')
    parser.add_argument("--load_dis_model", default='', help='load discriminator model')

    parser.add_argument("--load_dataset", default='celeba_train', help='load dataset')
    parser.add_argument("--dataset_path", "-d", default="/home/aixile/Workspace/dataset/celeba/",
                        help='dataset directory')

    args = parser.parse_args()
    print(args)

    gen = Generator()
    dis = Discriminator()

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


    def make_optimizer(model, alpha=0.0002, beta1=0.5):
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1)
        optimizer.setup(model)
        return optimizer

    opt_g=make_optimizer(gen, alpha=args.learning_rate_g)
    opt_d=make_optimizer(dis, alpha=args.learning_rate_d)

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
            'img_size': 64,
            'latent_len': 100,
            'gan_type': 'ls'
        },
    )

    trainer = training.Trainer(updater, (args.max_iter, 'iteration'), out=args.out)

    model_save_interval = (4000, 'iteration')
    eval_interval = (args.eval_interval, 'iteration')

    trainer.extend(extensions.snapshot_object(
        gen, 'gen_{.updater.iteration}.npz'), trigger=model_save_interval)
    trainer.extend(extensions.snapshot_object(
        dis, 'dis_{.updater.iteration}.npz'), trigger=model_save_interval)

    log_keys = ['epoch', 'iteration', 'gen/loss', 'dis/loss']
    trainer.extend(extensions.LogReport(keys=log_keys, trigger=(20, 'iteration')))
    trainer.extend(extensions.PrintReport(log_keys), trigger=(20, 'iteration'))
    trainer.extend(extensions.ProgressBar(update_interval=50))

    trainer.extend(
        Evaluation(gen, args.eval_folder, args.gpu), trigger=eval_interval
    )

    trainer.run()


if __name__ == '__main__':
    main()
