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
        self._lambda_gp = params['lambda_gp']
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

        data_z0 = self.get_latent_code_batch()
        x_fake0 = self.gen(Variable(data_z0))
        data_z1 = self.get_latent_code_batch()
        x_fake1 = self.gen(Variable(data_z1))
        data_x = self.get_real_image_batch()
        x_real = Variable(data_x)

        eta = np.random.rand()
        x_inter = Variable((data_x * eta + (1.0 - eta) * x_fake0.data).astype('f'))

        dis_x_fake0 = self.dis(x_fake0)
        dis_x_fake1 = self.dis(x_fake1)
        dis_x_real = self.dis(x_real)

        loss_gen = loss_l2_norm(dis_x_fake0, dis_x_real) + \
                    loss_l2_norm(dis_x_fake1, dis_x_real) - \
                    loss_l2_norm(dis_x_fake0, dis_x_fake1)
        #print(loss_gen.data)

        chainer.report({'loss': loss_gen}, self.gen)
        opt_g.zero_grads()
        loss_gen.backward()
        opt_g.update()

        x_fake0.unchain_backward()
        x_fake1.unchain_backward()

        loss_surrogate = loss_l2_norm(dis_x_fake0, dis_x_fake1) - \
                    loss_l2_norm(dis_x_fake0, 0.0) + \
                    loss_l2_norm(dis_x_real, 0.0) - \
                    loss_l2_norm(dis_x_real, dis_x_fake1)

        dis_x_inter = self.dis(x_inter, retain_forward=True)
        g = xp.ones_like(dis_x_inter.data)
        t0 = dis_x_inter.data - dis_x_fake1.data
        t0_norm = xp.sum(t0**2, axis=(1)) ** 0.5
        t1_norm = xp.sum(dis_x_inter.data**2, axis=(1)) ** 0.5
        t_g = ((t0.transpose() / t0_norm) - (dis_x_inter.data.transpose()) / t1_norm).transpose()
        g = g * t_g

        grad = self.dis.differentiable_backward(Variable(g))
        grad_l2 = F.sqrt(F.sum(grad**2, axis=(1, 2, 3)))
        loss_gp = self._lambda_gp * loss_l2(grad_l2, 1.0)

        loss_dis = loss_surrogate + loss_gp

        opt_d.zero_grads()
        loss_dis.backward()
        opt_d.update()

        chainer.report({'loss': loss_dis, 'loss_gp': loss_gp}, self.dis)
