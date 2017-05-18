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

def loss_func_tv_l2(x_out):
    xp = cuda.get_array_module(x_out.data)
    b, ch, h, w = x_out.data.shape
    Wx = xp.zeros((ch, ch, 2, 2), dtype="f")
    Wy = xp.zeros((ch, ch, 2, 2), dtype="f")
    for i in range(ch):
        Wx[i,i,0,0] = -1
        Wx[i,i,0,1] = 1
        Wy[i,i,0,0] = -1
        Wy[i,i,1,0] = 1
    return F.sum(F.convolution_2d(x_out, W=Wx) ** 2) + F.sum(F.convolution_2d(x_out, W=Wy) ** 2)

def loss_func_tv_l1(x_out):
    xp = cuda.get_array_module(x_out.data)
    b, ch, h, w = x_out.data.shape
    Wx = xp.zeros((ch, ch, 2, 2), dtype="f")
    Wy = xp.zeros((ch, ch, 2, 2), dtype="f")
    for i in range(ch):
        Wx[i,i,0,0] = -1
        Wx[i,i,0,1] = 1
        Wy[i,i,0,0] = -1
        Wy[i,i,1,0] = 1
    return F.sum(F.absolute(F.convolution_2d(x_out, W=Wx))) + F.sum(F.absolute(F.convolution_2d(x_out, W=Wy)))

class Updater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.gen, self.dis = kwargs.pop('models')
        self._iter = 0
        params = kwargs.pop('params')
        self._img_size = params['img_size']
        self._latent_len = params['latent_len']
        self._gan_type = params['gan_type']
        self._lambda_tv = params['lambda_tv']

        self._batch_size = params['batch_size']
        self._use_buffer = params['use_buffer']
        self._buffer = [xp.zeros((self._batch_size, 3, self._img_size, self._img_size)).astype("f") for i in range(20)]

        super(Updater, self).__init__(*args, **kwargs)

    def get_and_update_buffer(self, data):
        if  self._iter < 20:
            self._buffer[self._iter][:] = data[:]
            return data

        self._buffer[self._iter%20][:] = data[:]

        if np.random.rand() < 0.5:
            return data
        id = np.random.randint(0, 19)
        return self._buffer[id][:]

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
            loss_gen_adv =loss_func_adv_dis_real_ls(y_fake)
        else:
            loss_gen_adv =loss_func_adv_dis_real(y_fake)

        loss_tv = self._lambda_tv*loss_func_tv_l2(x_out)

        chainer.report({'loss_adv': loss_gen_adv, 'loss_tv':loss_tv}, self.gen)
        loss_gen = loss_gen_adv + loss_tv

        loss_gen.backward()
        opt_g.update()

        opt_d.zero_grads()

        if self._use_buffer:
            x_out = Variable(self.get_and_update_buffer(x_out.data))
            y_fake = self.dis(x_out, test=False)
        else:
            x_out.unchain_backward()

        if self._gan_type == 'ls':
            loss_dis = loss_func_adv_dis_fake_ls(y_fake) + loss_func_adv_dis_real_ls(y_real)
        else:
            loss_dis = loss_func_adv_dis_fake(y_fake) + loss_func_adv_dis_real(y_real)

        chainer.report({'loss': loss_dis}, self.dis)

        loss_dis.backward()
        opt_d.update()
