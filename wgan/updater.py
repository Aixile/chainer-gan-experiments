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
        self._latent_len = params['latent_len']
        self._dis_iter = params['dis_iter']
        self._batch_size = params['batch_size']
        self._lambda_gp = params['lambda_gp']
        self._mode = params['mode']

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

        opt_d = self.get_optimizer('dis')
        for i in range(self._dis_iter):
            d_fake = self.get_fake_image_batch()
            d_real = self.get_real_image_batch()

            y_fake = self.dis(Variable(d_fake), test=False)
            y_real = self.dis(Variable(d_real), test=False)

            w1 = F.average(y_fake-y_real)

            loss_dis = w1

            if self._mode == 'gp':
                eta = np.random.rand()
                c = (d_real * eta + (1.0 - eta) * d_fake).astype('f')
                y = self.dis(Variable(c), test=False, retain_forward=True)

                g = xp.ones_like(y.data)
                grad_c = self.dis.differentiable_backward(Variable(g))
                grad_c_l2 = F.sqrt(F.sum(grad_c**2, axis=(1, 2, 3)))

                loss_gp = loss_l2(grad_c_l2, 1.0)

                loss_dis += self._lambda_gp * loss_gp

            opt_d.zero_grads()
            loss_dis.backward()
            opt_d.update()

            if self._mode == 'clip':
                self.dis.clip()

        chainer.report({'loss': loss_dis,'loss_w1': w1}, self.dis)

        z_in = self.get_latent_code_batch()
        x_out = self.gen(Variable(z_in), test=False)

        opt_g = self.get_optimizer('gen')
        y_fake = self.dis(x_out, test=False)
        loss_gen = - F.average(y_fake)

        chainer.report({'loss': loss_gen}, self.gen)

        opt_g.zero_grads()
        loss_gen.backward()
        opt_g.update()
