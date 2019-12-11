import tensorflow as tf

class ResBlock(tf.keras.Model):
    def __init__(self, k=3, n=64, s=1):
        super(ResBlock, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(filters=n, kernel_size=k, strides=s, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.prelu = tf.keras.layers.PReLU()
        self.conv2 = tf.keras.layers.Conv2D(filters=n, kernel_size=k, strides=s, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()

    def call(self, x0):
        x = self.conv1(x0)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return x + x0

# PixelShuffler introduced in SRGAN paper
class PixelShuffler(tf.keras.layers.Layer):
    def __init__(self, scale=2):
        super(PixelShuffler, self).__init__()
        self.scale = scale
    def call(self, x0):
        return tf.nn.depth_to_space(x0, self.scale)

# We use transposed conv layer to replace PixelShuffler
class UpsamplingBlock(tf.keras.Model):
    #def __init__(self, k=3, n=256, s=1, upscale=2):
    def __init__(self, k=3, n=256, s=2):
        super(UpsamplingBlock, self).__init__()

        #self.conv = tf.keras.layers.Conv2D(filters=n, kernel_size=k, strides=s, padding='same')
        #self.ps = PixelShuffler(upscale)
        self.trans_conv = tf.keras.layers.Conv2DTranspose(filters=n, kernel_size=k*s, strides=s, padding='same')
        self.prelu = tf.keras.layers.PReLU()

    def call(self, x0):
        #x = self.conv(x0)
        #x = self.ps(x)
        x = self.trans_conv(x0)
        x = self.prelu(x)
        return x

class SRResNet(tf.keras.Model):
    def __init__(self, n_channel, final_bias=0):
        super(SRResNet, self).__init__()

        self.conv_pre = tf.keras.layers.Conv2D(filters=64, kernel_size=9, strides=1, padding='same')
        self.bn_pre = tf.keras.layers.BatchNormalization()

        self.res01 = ResBlock()
        self.res02 = ResBlock()
        self.res03 = ResBlock()
        self.res04 = ResBlock()
        self.res05 = ResBlock()
        self.res06 = ResBlock()
        self.res07 = ResBlock()
        self.res08 = ResBlock()
        self.res09 = ResBlock()
        self.res10 = ResBlock()
        self.res11 = ResBlock()
        self.res12 = ResBlock()
        self.res13 = ResBlock()
        self.res14 = ResBlock()
        self.res15 = ResBlock()
        self.res16 = ResBlock()

        self.conv_post = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')
        self.bn_post = tf.keras.layers.BatchNormalization()

        self.upsampling1 = UpsamplingBlock()
        self.upsampling2 = UpsamplingBlock()

        final_bias_init = tf.keras.initializers.Constant(value=final_bias)
        self.conv_final = tf.keras.layers.Conv2D(filters=n_channel, kernel_size=9, strides=1, padding='same', bias_initializer=final_bias_init)

    def call(self, x0):
        pre = self.conv_pre(x0)
        pre = self.bn_pre(pre)

        x = self.res01(pre)
        x = self.res02(x)
        x = self.res03(x)
        x = self.res04(x)
        x = self.res05(x)
        x = self.res06(x)
        x = self.res07(x)
        x = self.res08(x)
        x = self.res09(x)
        x = self.res10(x)
        x = self.res11(x)
        x = self.res12(x)
        x = self.res13(x)
        x = self.res14(x)
        x = self.res15(x)
        x = self.res16(x)

        x = self.conv_post(x)
        x = self.bn_post(x)

        x = x + pre
        x = self.upsampling1(x)
        x = self.upsampling2(x)
        x = self.conv_final(x)
        return x

# Downsample high-resolution data to generate low-resolution inputs
class DownsamplingNet(tf.keras.Model):
    def __init__(self, scale=4):
        super(DownsamplingNet, self).__init__()
        self.downsampling = tf.keras.layers.AveragePooling2D(scale)

    def call(self, x0):
        x = self.downsampling(x0)
        x = (x+1.0) * 0.5  # shift domain to [0,1]
        return x

def ReconNet(n_channel, final_bias=0):
    return tf.keras.Sequential([DownsamplingNet(4), SRResNet(n_channel, final_bias=final_bias)])


class DisBlock(tf.keras.Model):
    def __init__(self, k, n, s, bn_layer=True, alpha=0.2):
        super(DisBlock, self).__init__()

        self.conv = tf.keras.layers.Conv2D(filters=n, kernel_size=k, strides=s, padding='same')
        self.bn_layer = bn_layer
        if self.bn_layer:
            self.bn = tf.keras.layers.BatchNormalization()
        self.lrelu = tf.keras.layers.LeakyReLU(alpha=alpha)

    def call(self, x0):
        x = self.conv(x0)
        if self.bn_layer:
            x = self.bn(x)
        x = self.lrelu(x)
        return x

class Discriminator(tf.keras.Model):
    def __init__(self, alpha=0.2):
        super(Discriminator, self).__init__()

        self.dis01 = DisBlock(k=3, n=64, s=1, bn_layer=False, alpha=alpha)
        self.dis02 = DisBlock(k=3, n=64, s=2, alpha=alpha)
        self.dis03 = DisBlock(k=3, n=128, s=1, alpha=alpha)
        self.dis04 = DisBlock(k=3, n=128, s=2, alpha=alpha)
        self.dis05 = DisBlock(k=3, n=256, s=1, alpha=alpha)
        self.dis06 = DisBlock(k=3, n=256, s=2, alpha=alpha)
        self.dis07 = DisBlock(k=3, n=512, s=1, alpha=alpha)
        self.dis08 = DisBlock(k=3, n=512, s=2, alpha=alpha)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(1024)
        self.lrelu1 = tf.keras.layers.LeakyReLU(alpha=alpha)
        self.dense2 = tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)

    def call(self, x0):
        x = self.dis01(x0)
        x = self.dis02(x)
        x = self.dis03(x)
        x = self.dis04(x)
        x = self.dis05(x)
        x = self.dis06(x)
        x = self.dis07(x)
        x = self.dis08(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.lrelu1(x)
        x = self.dense2(x)
        return x
