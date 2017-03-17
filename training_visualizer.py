import os
import numpy as np

import chainer
import chainer.cuda
from chainer import cuda, serializers, Variable
import chainer.functions as F
import cv2
# def samples_colorization(updater):

def test_sampling(updater, generator, output_path, sample_batch_size=32, scaled=True):
    @chainer.training.make_extension()

    def batch_sampling(trainer):
        samples = generator.gen_samples(sample_batch_size)
        samples = cuda.to_cpu(samples)
        if scaled:
            samples = (samples + 1) * 127.5

        samples = samples.clip(0, 255).astype(np.uint8)

        row_n = (sample_batch_size//8)+1
        if row_n%8 == 0:
            row_n -= 1

        img_size = samples.shape[2]
        channls = samples.shape[1]

        result_col_size = 8*(img_size+5)
        result_row_size = row_n*(img_size+5)
        img_grid = np.zeros((result_row_size, result_col_size, channls))

        for i in range(0,len(samples)):
            img = sampels[i]
            img = img.transpose(1, 2, 0)
            row = (i//8)*(img_size+5)
            col = (i%8)*(img_size+5)
            img_grid[row:row+img_size, col:col+img_size] = img

        cv2.imwrite(output_path+"iter_"+str(trainer.updater.iteration)+".jpg", img_grid)

    return batch_sampling
