# srgan

This is a minimal Tensorflow 2.0 implementation of super-resolution generative adversarial network (SRGAN) architecture based on [Ledig et al. (2016)](https://arxiv.org/abs/1609.04802).
The idea of GAN comes from [Goodfellow et al. (2014)](https://arxiv.org/abs/1406.2661) and the generator in SRGAN adopts an SRResNet architecture.

Note that we avoided the use of sub-pixel convolution to get rid of checkerboard artifects.
