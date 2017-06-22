import os
import chainer
from chainer.training import extension
from chainer import Variable, cuda
import chainer.functions as F
import numpy as np
import os
import cv2
from .save_images import save_images_grid

def gan_sampling(gen, eval_folder, gpu, rows=6, cols=6, latent_len=128):
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

def analogy(gen, output, samples=12, latent_len=128, points=10):
    xp = gen.xp
    z0 = xp.random.normal(size=(samples, latent_len)).astype("f")
    z1 = xp.random.normal(size=(samples, latent_len)).astype("f")
    values = np.linspace(0.0, 1.0, num=points)
    results = []
    for i in range(points):
        z = (values[i]*z0 + (1.0-values[i])*z1).astype("f")
        z = Variable(z, volatile=True)
        imgs = gen(z, test=True)
        results.append(imgs.data)
    results = xp.concatenate(results, axis=0)
    save_images_grid(results, path=output, grid_w=points, grid_h=samples, transposed=True)
