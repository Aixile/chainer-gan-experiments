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
        self.gen, self.enc, self.dis_x, self.dis_z = kwargs.pop('models')
        self._iter = 0
        params = kwargs.pop('params')
        self._img_size = params['img_size']
        self._img_chan = params['img_chan']
        self._batch_size = params['batch_size']
        self._latent_len = params['latent_len']
        self._lambda_l1 = params['lambda_l1']

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
        self._iter += 1

        opt_enc = self.get_optimizer('enc')
        opt_gen = self.get_optimizer('gen')

        opt_enc.zero_grads()
        d_x_real = self.get_real_image_batch()
        x_real = Variable(d_x_real)
        z_fake = self.enc(x_real)
        x_rec = self.gen(z_fake)
        dis_z_fake = self.dis_z(z_fake)

        loss_enc = self._lambda_l1 * loss_l1(x_rec, x_real) + loss_func_dcgan_dis_real(dis_z_fake)
        loss_enc.backward()
        opt_enc.update()

        z_fake.unchain_backward()
        opt_gen.zero_grads()

        d_z_real = self.get_latent_code_batch()
        z_real = Variable(d_z_real)
        x_fake = self.gen(z_real)
        dis_x_fake = self.dis_x(x_fake)
        dis_x_rec = self.dis_x(x_rec)

        loss_gen =  self._lambda_l1 * loss_l1_sum(x_rec, x_real) + loss_func_dcgan_dis_real(dis_x_fake) + loss_func_dcgan_dis_real(dis_x_rec)

        loss_gen.backward()
        opt_gen.update()

        chainer.report({'loss': loss_gen}, self.gen)
        chainer.report({'loss': loss_enc}, self.enc)

        if self._iter % 2 == 0:
            opt_dis_x = self.get_optimizer('dis_x')
            opt_dis_z = self.get_optimizer('dis_z')
            opt_dis_x.zero_grads()
            opt_dis_z.zero_grads()

            x_rec.unchain_backward()
            x_fake.unchain_backward()
            z_fake.unchain_backward()

            dis_x_real = self.dis_x(x_real)
            dis_z_real = self.dis_z(z_real)

            loss_dis_x = loss_func_dcgan_dis_fake(dis_x_fake) + loss_func_dcgan_dis_fake(dis_x_rec) + loss_func_dcgan_dis_real(dis_x_real)
            loss_dis_z = loss_func_dcgan_dis_fake(dis_z_fake) + loss_func_dcgan_dis_real(dis_z_real)

            loss_dis_x.backward()
            loss_dis_z.backward()
            opt_dis_x.update()
            opt_dis_z.update()

            chainer.report({'loss': loss_dis_x}, self.dis_x)
            chainer.report({'loss': loss_dis_z}, self.dis_z)
