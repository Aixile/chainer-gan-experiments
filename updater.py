import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
import chainer.datasets.image_dataset as ImageDataset
import six
import os

from chainer import cuda, optimizers, serializers, Variable
from chainer import training

from loss_functions import *

class ganUpdater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.gen, self.dis = kwargs.pop('models')#,self.l = kwargs.pop('models')
        params = kwargs.pop('params')
        self._iter = 0
        self._gen_dis_update_radio = params['gen_dis_update_radio']
        self._loss_type = params['adv_loss_type']
        super(ganUpdater, self).__init__(*args, **kwargs)

    def update_core(self):
        xp = self.gen.xp
        self._iter += 1

        batch = self.get_iterator('main').next()
        batchsize = len(batch)

        t_out = Variable(xp.asarray(batch).astype("f"))

        gen_optimizer = self.get_optimizer('gen')
        dis_optimizer = self.get_optimizer('dis')

        x_out = self.gen_samples(batchsize)
        y_fake = self.dis(x_out, test=False)

        if self._iter % self._gen_dis_update_radio == 0:

            if self._loss_type == 'ls':
                gen_adv_loss = LOSS_MSE(y_fake, 1)
            else:
                gen_adv_loss = LOSS_SCE(y_fake, Variable(xp.ones([batchsize,1], dtype=np.int32)))

            chainer.report({'loss': gen_adv_loss}, self.gen)
            cnn_optimizer.update(lossfun=lambda: gen_adv_loss)

        #x_out.unchain_backward()

        #y_fake = self.dis(x_out, test=False)
        y_real = self.dis(t_out, test=False)

        if args.adv_loss_type == 'ls':
            dis_adv_loss = LOSS_MSE(y_real,1) + LOSS_MSE(y_fake,0)
        else:
            dis_adv_loss = LOSS_SCE(y_fake, Variable(xp.zeros([batchsize,1], dtype=np.int32)))+LOSS_SCE(y_real, Variable(xp.ones([batchsize,1], dtype=np.int32)))

        chainer.report({'loss': dis_adv_loss}, self.dis)

        dis_optimizer.update(lossfun=lambda: dis_adv_loss)
