# Chainer GAN implementation

**Dependence**: Chainer, OpenCV


1. Use Least-Square GAN by default.
2. Patch Discriminiator: For 64*64 images, use 5 down-size convolution layers.
3. Add total variation regurization may help ?
4. Data augmentation with random fliping and random brightness adjusting
5. Unpooling2d + Convolution as Deconvolution layers
