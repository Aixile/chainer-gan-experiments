import os
import chainer
from chainer.training import extension
from chainer import Variable, cuda
import chainer.functions as F
import numpy as np
import os
import cv2
from utils import save_images_grid

def Evaluation(gen, eval_folder, gpu, rows=6, cols=6, latent_len=100):
    @chainer.training.make_extension()

    def samples_generation(trainer):

        if not os.path.exists(eval_folder):
            os.makedirs(eval_folder)

        z = np.random.normal(size=(rows*cols, latent_len)).astype("f")
        if gpu>=0:
            z = cuda.to_gpu(z)
        z = Variable(z, volatile=True)
        imgs = gen(z, test=True)

        save_images_grid(imgs, path=eval_folder+"/iter_"+str(trainer.updater.iteration)+".jpg",
            grid_w=rows, grid_h=cols)

    return samples_generation
