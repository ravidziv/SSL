"""File that stores layers object and functions for data augmentation"""
import tensorflow as tf
import tensorflow_addons as tfa


def gaussian_kernel(kernel_size: int, sigma: float):
    """Defines gaussian kernel

    Args:
        kernel_size: Python int, size of the Gaussian kernel
        sigma: Python int, standard deviation of the Gaussian kernel
    Returns:
        2-D Tensor of gaussian kernel
    """

    sigma = tf.cast(sigma, tf.float32)
    x = tf.linspace(-kernel_size / 2, kernel_size / 2, kernel_size)
    [y, x] = tf.meshgrid(x, x)
    kernel = tf.math.exp(-(tf.math.square(x) + tf.math.square(y)) / (2 * tf.square(sigma)))
    kernel = kernel / tf.reduce_sum(kernel)
    return kernel


def gaussian_blur(image: tf.Tensor, kernel_size: int = 3, sigma: float = 3.):
    """Convolve a gaussian kernel with input image
    Convolution is performed depthwise

    Args:
        image: 3-D Tensor of image, should by floats
        kernel_size: float  for the gaussian kernel size
        sigma: standard deviation of the Gaussian kernel
    Returns:
        3-D Tensor image convolved with gaussian kernel
    """

    kernel = gaussian_kernel(kernel_size, sigma)
    kernel = tf.expand_dims(tf.stack([kernel, kernel, kernel], axis=-1), axis=-1)
    pointwise_filter = tf.eye(3, batch_shape=[1, 1])
    image = tf.expand_dims(image, axis=0)
    image = tf.nn.separable_conv2d(image, kernel, pointwise_filter, strides=[1, 1, 1, 1], padding='SAME')
    image = tf.squeeze(image, axis=0)
    return image


def _gaussian_kernel(kernel_size: int, sigma: float, n_channels: int, dtype: tf.dtypes.DType):
    """Defines gaussian kernel

    Args:
        kernel_size: Python int, size of the Gaussian kernel
        sigma: Python int, standard deviation of the Gaussian kernel
        n_channels: Number of channels in the image
        dtype: type of the original tensor
    Returns:
        2-D Tensor of gaussian kernel
    """

    x = tf.range(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=dtype)
    g = tf.math.exp(-(tf.pow(x, 2) / (2 * tf.pow(tf.cast(sigma, dtype), 2))))
    g_norm2d = tf.pow(tf.reduce_sum(g), 2)
    g_kernel = tf.tensordot(g, g, axes=0) / g_norm2d
    g_kernel = tf.expand_dims(g_kernel, axis=-1)
    return tf.expand_dims(tf.tile(g_kernel, (1, 1, n_channels)), axis=-1)


def apply_blur(img: tf.Tensor, kernel_size: int, sigma: float):
    """Convolve a gaussian kernel with input image
    Convolution is performed depthwise

    Args:
        img: 3-D Tensor of image, should by floats
        kernel_size: Python int, size of the Gaussian kernel
        sigma: Python int, standard deviation of the Gaussian kernel
    Returns:
        img: 3-D Tensor image convolved with gaussian kernel
    """

    blur = _gaussian_kernel(kernel_size, sigma, 3, img.dtype)
    img = tf.nn.depthwise_conv2d(img, blur, [1, 1, 1, 1], 'SAME')
    return img


class ColorDrop(tf.keras.layers.Layer):
    """Layer that convert to grayscale"""

    def __init__(self, color_prob: float, **kwargs):
        super().__init__(**kwargs)
        self.color_prob = color_prob

    def call(self, inputs, training=True):
        if training:
            random_apply = tf.random.uniform(minval=0, maxval=1, shape=[1])
            inputs = tf.cond(self.color_prob > random_apply[0],
                             lambda: color_drop(inputs),
                             lambda: tf.identity(inputs))
            inputs = tf.clip_by_value(inputs, 0, 1)

        return inputs


class CutOut(tf.keras.layers.Layer):
    """Layer that cutouts the image """

    def __init__(self, color_prob: float, crop_size: [int, int], cutout_prob: float, **kwargs):
        super().__init__(**kwargs)
        self.color_prob = color_prob
        self.crop_size = crop_size
        self.cutout_prob = cutout_prob

    def call(self, inputs, training=True, ):
        if training:
            random_apply = tf.random.uniform(minval=0, maxval=1, shape=[1])
            inputs = cutout(inputs, random_apply, self.crop_size, self.cutout_prob)
            inputs = tf.clip_by_value(inputs, 0, 1)
        return inputs


class RandomJitter(tf.keras.layers.Layer):
    """Layer for applying color jitter noise"""

    def __init__(self, jitter_prob: float = 0.5, strength: float = 0., **kwargs):
        super().__init__(**kwargs)
        self.jitter_prob = jitter_prob
        self.strength = strength

    def call(self, images, training=True):
        if training:
            random_apply = tf.random.uniform(minval=0, maxval=1, shape=[1])
            images = tf.cond(self.jitter_prob > random_apply[0],
                             lambda: color_jitter(images, strength=self.strength),
                             lambda: tf.identity(images))
            images = tf.clip_by_value(images, 0, 1)
        return images


def color_jitter(image, strength: float = 1.) -> tf.Tensor:
    """Apply color jitter operation"""
    image = tf.image.random_brightness(image, max_delta=0.8 * strength)
    image = tf.image.random_contrast(image, lower=1 - 0.8 * strength,
                                     upper=1 + 0.8 * strength)
    image = tf.image.random_saturation(image, lower=1 - 0.8 * strength,
                                       upper=1 + 0.8 * strength)
    image = tf.image.random_hue(image, max_delta=0.2 * strength)
    return image


def color_drop(image: tf.Tensor) -> tf.Tensor:
    """drop color channel by convert to grayscale"""
    image = tf.image.rgb_to_grayscale(image)
    image = tf.tile(image, [1, 1, 1, 3])
    return image


def random_blur(image: tf.Tensor, random_apply: [float, float], crop_size: [int, int], blur_prob: float) -> tf.Tensor:
    """Applies gaussian blurring randomly"""
    sigma = tf.random.uniform(minval=0.01, maxval=2.0, shape=[1])
    kernel_size = crop_size[0] // 10
    image = tf.cond(blur_prob > random_apply[0],
                    lambda: apply_blur(image, kernel_size, sigma),
                    lambda: tf.identity(image))
    return image


class RandomBlur(tf.keras.layers.Layer):
    """Layer for random blurring"""

    def __init__(self, crop_size: [int, int], blur_prob: float, **kwargs):
        super().__init__(**kwargs)
        self.crop_size = crop_size
        self.blur_prob = blur_prob

    def call(self, inputs, training=True):
        if training:
            random_apply = tf.random.uniform(minval=0, maxval=1, shape=[1])
            inputs = random_blur(inputs, random_apply, self.crop_size, self.blur_prob)
            inputs = tf.clip_by_value(inputs, 0, 1)
        return inputs


def cutout(image: tf.Tensor, random_apply: [int, int], crop_size: [int, int],
           cutout_prob: float) -> tf.Tensor:
    """Randomly applies cutout of image, cutout is 3 times smaller than image"""
    mask_size = [int(crop_size[0] / 6) * 2] * 2  # 3 times smaller, must be divisible by 2
    offset = tf.random.uniform(minval=crop_size[0] // 10,
                               maxval=crop_size[0] - crop_size // 10,
                               shape=[2], dtype=tf.int32)
    image = tf.cond(cutout_prob > random_apply[0],
                    lambda: tfa.image.cutout(image, [mask_size[0], mask_size[1]],
                                             [offset[0], offset[1]]),
                    lambda: tf.identity(image))
    return image


class AugmentationModel(tf.keras.Model):
    """Model that stores all the augmentation layers based on the config dict"""

    def get_config(self):
        return {'config': self.config, 'model': self.model}

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        layers = []
        if self.config.crop:
            layers.append(tf.keras.layers.experimental.preprocessing.RandomCrop(height=self.config.crop_size[0],
                                                                                width=self.config.crop_size[1]))
        if self.config.flip:
            layers.append(tf.keras.layers.experimental.preprocessing.RandomFlip())
        if self.config.blur:
            layers.append(RandomBlur(crop_size=self.config.crop_size, blur_prob=self.config.blur_prob))
        if self.config.noise:
            layers.append(tf.keras.layers.GaussianNoise(stddev=0.025))
        if self.config.rotate:
            layers.append(tf.keras.layers.experimental.preprocessing.RandomRotation(factor=self.config.rotation_factor))
        if self.config.jitter:
            layers.append(RandomJitter(jitter_prob=self.config.jitter_prob, strength=self.config.strength))
        if self.config.colordrop:
            layers.append(ColorDrop(color_prob=self.config.color_prob))
        if self.config.cutout:
            layers.append(CutOut(color_prob=self.config.color_prob, crop_size=self.config.crop_size,
                                 cutout_prob=self.config.cutout_prob))
        self.model = tf.keras.models.Sequential(layers)

    def call(self, image, training=None, mask=None):
        image_aug = self.model(image, training=training)
        image_aug2 = self.model(image, training=training)
        return {'x':image_aug, 'x2':image_aug2}
