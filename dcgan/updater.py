import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, optimizers, serializers, Variable
import sys
sys.path.insert(0, '../')
from common.loss_functions import *

class Updater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.gen, self.dis = kwargs.pop('models')
        self._iter = 0
        params = kwargs.pop('params')
        self._img_size = params['img_size']
        self._img_chan = params['img_chan']
        self._batch_size = params['batch_size']
        self._latent_len = params['latent_len']
        super(Updater, self).__init__(*args, **kwargs)

    def get_real_image_batch(self):
        xp = self.gen.xp
        batch = self.get_iterator('main').next()
        t_out = xp.zeros((self._batch_size, self._img_chan, self._img_size, self._img_size)).astype("f")
        for i in range(self._batch_size):
            t_out[i, :] = xp.asarray(batch[i])
        return t_out

    def get_fake_image_batch(self):
        z = self.get_latent_code_batch()
        x_out = self.gen(Variable(z, volatile=True), test=True).data
        return x_out

    def get_latent_code_batch(self):
        xp = self.gen.xp
        z_in = xp.random.normal(size=(self._batch_size, self._latent_len)).astype("f")
        return z_in

    def update_core(self):
        xp = self.gen.xp
        self._iter += 1

        opt_g = self.get_optimizer('gen')
        opt_d = self.get_optimizer('dis')

        data_z = self.get_latent_code_batch()
        data_x = self.get_real_image_batch()

        x_fake = self.gen(Variable(data_z))
        dis_fake = self.dis(x_fake)

        loss_gen = loss_func_dcgan_dis_real(dis_fake)
        chainer.report({'loss': loss_gen}, self.gen)

        opt_g.zero_grads()
        loss_gen.backward()
        opt_g.update()

        x_fake.unchain_backward()
        x_real = Variable(data_x)
        dis_real = self.dis(x_real)
        loss_dis = loss_func_dcgan_dis_real(dis_real) + loss_func_dcgan_dis_fake(dis_fake)

        opt_d.zero_grads()
        loss_dis.backward()
        opt_d.update()

        chainer.report({'loss': loss_dis}, self.dis)
