import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
import chainer.datasets.image_dataset as ImageDataset
import six
import os

from chainer import cuda, optimizers, serializers, Variable
from chainer import training


def cal_l2_sum(h, t):
    return F.sum((h-t)**2)/ np.prod(h.data.shape)

def loss_func_adv_dis_fake_ls(y_fake):
    return cal_l2_sum(y_fake, 0.1)

def loss_func_adv_dis_real_ls(y_real):
    return cal_l2_sum(y_real, 0.9)

def loss_func_adv_dis_real(y_real):
    F.sum(F.softplus(-y_real)) / np.prod(y_real.data.shape)

def loss_func_adv_dis_fake(y_fake):
    F.sum(F.softplus(y_fake)) / np.prod(y_fake.data.shape)

class Updater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.gen, self.dis = kwargs.pop('models')
        self._iter = 0
        params = kwargs.pop('params')
        self._img_size = params['img_size']
        self._latent_len = params['latent_len']
        self._gan_type = params['gan_type']
        super(Updater, self).__init__(*args, **kwargs)

    def update_core(self):
        xp = self.gen.xp
        self._iter += 1

        batch = self.get_iterator('main').next()
        batchsize = len(batch)

        z_in = xp.random.normal(size=(batchsize, self._latent_len)).astype("f")

        W = self._img_size
        t_out = xp.zeros((batchsize, 3, W, W)).astype("f")
        for i in range(batchsize):
            t_out[i, :] = xp.asarray(batch[i])
        t_out = Variable(t_out)

        x_out = self.gen(z_in, test=False)

        opt_g = self.get_optimizer('gen')
        opt_d = self.get_optimizer('dis')

        opt_g.zero_grads()
        z_in = Variable(z_in)

        y_fake = self.dis(x_out, test=False)
        y_real = self.dis(t_out, test=False)

        if self._gan_type == 'ls':
            loss_gen =loss_func_adv_dis_real_ls(y_fake)
        else:
            loss_gen =loss_func_adv_dis_real(y_fake)

        chainer.report({'loss': loss_gen}, self.gen)
        loss_gen.backward()
        opt_g.update()

        opt_d.zero_grads()
        x_out.unchain_backward()

        if self._gan_type == 'ls':
            loss_dis = loss_func_adv_dis_fake_ls(y_fake) + loss_func_adv_dis_real_ls(y_real)
        else:
            loss_dis = loss_func_adv_dis_fake(y_fake) + loss_func_adv_dis_real(y_real)

        chainer.report({'loss': loss_dis}, self.dis)

        loss_dis.backward()
        opt_d.update()
