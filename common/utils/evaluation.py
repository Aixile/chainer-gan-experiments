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

def gan_sampling_tags(gen, eval_folder, gpu, rows=6, cols=6, latent_len=128, attr_len=38, threshold=0.25):
    @chainer.training.make_extension()
    def get_fake_tag():
        prob2 = np.random.rand(attr_len)
        tags = np.zeros((attr_len)).astype("f")
        tags[:] = -1.0
        tags[np.argmax(prob2[0:13])]=1.0
        tags[27 + np.argmax(prob2[27:])] = 1.0
        prob2[prob2<threshold] = -1.0
        prob2[prob2>=threshold] = 1.0
        for i in range(13, 27):
            tags[i] = prob2[i]
        return tags

    def get_fake_tag_batch():
        xp = gen.xp
        batch = rows*cols
        tags = xp.zeros((batch, attr_len)).astype("f")
        for i in range(batch):
            tags[i] = xp.asarray(get_fake_tag())
        return tags

    def samples_generation(trainer):
        if not os.path.exists(eval_folder):
            os.makedirs(eval_folder)
        z = np.random.normal(size=(rows*cols, latent_len)).astype("f")
        if gpu>=0:
            z = cuda.to_gpu(z)
        tags =get_fake_tag_batch()
        z = Variable(z, volatile=True)
        tags = Variable(tags, volatile=True)
        imgs = gen(F.concat([z,tags]), test=True)
        save_images_grid(imgs, path=eval_folder+"/iter_"+str(trainer.updater.iteration)+".jpg",
            grid_w=rows, grid_h=cols)

    return samples_generation

def ae_reconstruction(enc, dec, eval_folder, gpu, data_iter, batch_size=32, img_chan=3, img_size=64):
    @chainer.training.make_extension()
    def sample_reconstruction(trainer):
        xp = enc.xp
        batch = data_iter.next()
        d_real = xp.zeros((batch_size, img_chan, img_size, img_size)).astype("f")
        for i in range(batch_size):
            d_real[i, :] = xp.asarray(batch[i])
        x = Variable(d_real, volatile=True)
        imgs = dec(enc(x, test=True), test=True)
        save_images_grid(imgs, path=eval_folder+"/iter_"+str(trainer.updater.iteration)+".rec.jpg",
            grid_w=batch_size//8, grid_h=8)
        save_images_grid(d_real, path=eval_folder+"/iter_"+str(trainer.updater.iteration)+".real.jpg",
            grid_w=batch_size//8, grid_h=8)

    return sample_reconstruction

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
