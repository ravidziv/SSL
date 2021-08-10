"""
Residual networks (ResNets) from:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""

import tensorflow as tf
import logging
BATCH_NORM_EPSILON = 1e-5


class BatchNormRelu(tf.keras.layers.Layer):  # pylint: disable=missing-docstring

    def __init__(self,
                 relu=True,
                 init_zero=False,
                 center=True,
                 scale=True,
                 global_bn=True,
                 data_format='channels_last',
                 batch_norm_decay=0.9,
                 **kwargs):
        """

        :param relu:
        :param init_zero:
        :param center:
        :param scale:
        :param global_bn:
        :param data_format:
        :param batch_norm_decay:
        :param kwargs:
        """
        super(BatchNormRelu, self).__init__(**kwargs)
        self.relu = relu
        if init_zero:
            gamma_initializer = tf.zeros_initializer()
        else:
            gamma_initializer = tf.ones_initializer()
        if data_format == 'channels_first':
            axis = 1
        else:
            axis = -1
        if global_bn:
            # Batch normalization layers with fused=True only support 4D input
            # tensors.
            self.bn = tf.keras.layers.experimental.SyncBatchNormalization(
                axis=axis,
                momentum=batch_norm_decay,
                epsilon=BATCH_NORM_EPSILON,
                center=center,
                scale=scale,
                gamma_initializer=gamma_initializer)
        else:
            # Batch normalization layers with fused=True only support 4D input
            # tensors.
            self.bn = tf.keras.layers.BatchNormalization(
                axis=axis,
                momentum=batch_norm_decay,
                epsilon=BATCH_NORM_EPSILON,
                center=center,
                scale=scale,
                fused=False,
                gamma_initializer=gamma_initializer)

    def call(self, inputs, training=None, **kwargs):
        inputs = self.bn(inputs, training=training)
        if self.relu:
            inputs = tf.nn.relu(inputs)
        return inputs


class DropBlock(tf.keras.layers.Layer):  # pylint: disable=missing-docstring

    def __init__(self,
                 keep_prob,
                 drop_block_size,
                 data_format='channels_last',
                 **kwargs):
        self.keep_prob = keep_prob
        self.drop_block_size = drop_block_size
        self.data_format = data_format
        super(DropBlock, self).__init__(**kwargs)

    def call(self, net, training=None, **kwargs):
        keep_prob = self.keep_prob
        drop_block_size = self.drop_block_size
        data_format = self.data_format
        if not training or keep_prob is None:
            return net

        logging.info(
            'Applying DropBlock: drop_block_size {}, net.shape {}'.format(
                drop_block_size, net.shape))

        if data_format == 'channels_last':
            _, width, height, _ = net.get_shape().as_list()
        else:
            _, _, width, height = net.get_shape().as_list()
        if width != height:
            raise ValueError('Input tensor with width!=height is not supported.')

        drop_block_size = min(drop_block_size, width)
        # seed_drop_rate is the gamma parameter of Drop block.
        seed_drop_rate = (1.0 - keep_prob) * width ** 2 / drop_block_size ** 2 / (
                width - drop_block_size + 1) ** 2

        # Forces the block to be inside the feature map.
        w_i, h_i = tf.meshgrid(tf.range(width), tf.range(width))
        valid_block_center = tf.logical_and(
            tf.logical_and(w_i >= int(drop_block_size // 2),
                           w_i < width - (drop_block_size - 1) // 2),
            tf.logical_and(h_i >= int(drop_block_size // 2),
                           h_i < width - (drop_block_size - 1) // 2))

        valid_block_center = tf.expand_dims(valid_block_center, 0)
        valid_block_center = tf.expand_dims(
            valid_block_center, -1 if data_format == 'channels_last' else 0)

        rand_noise = tf.random.uniform(net.shape, dtype=tf.float32)
        block_pattern = (1 - tf.cast(valid_block_center, dtype=tf.float32) +
                         tf.cast((1 - seed_drop_rate), dtype=tf.float32) + rand_noise) >= 1
        block_pattern = tf.cast(block_pattern, dtype=tf.float32)

        if drop_block_size == width:
            block_pattern = tf.reduce_min(
                block_pattern,
                axis=[1, 2] if data_format == 'channels_last' else [2, 3],
                keepdims=True)
        else:
            if data_format == 'channels_last':
                ksize = [1, drop_block_size, drop_block_size, 1]
            else:
                ksize = [1, 1, drop_block_size, drop_block_size]
            block_pattern = -tf.nn.max_pool(
                -block_pattern,
                ksize=ksize,
                strides=[1, 1, 1, 1],
                padding='SAME',
                data_format='NHWC' if data_format == 'channels_last' else 'NCHW')

        percent_ones = (
                tf.cast(tf.reduce_sum(block_pattern), tf.float32) /
                tf.cast(tf.size(block_pattern), tf.float32))

        net = net / tf.cast(percent_ones, net.dtype) * tf.cast(
            block_pattern, net.dtype)
        return net


class FixedPadding(tf.keras.layers.Layer):  # pylint: disable=missing-docstring

    def __init__(self, kernel_size, data_format='channels_last', **kwargs):
        super(FixedPadding, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.data_format = data_format

    def call(self, inputs, training=None, **kwargs):
        kernel_size = self.kernel_size
        data_format = self.data_format
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        if data_format == 'channels_first':
            padded_inputs = tf.pad(
                inputs, [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]])
        else:
            padded_inputs = tf.pad(
                inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])

        return padded_inputs


class Conv2dFixedPadding(tf.keras.layers.Layer):  # pylint: disable=missing-docstring

    def __init__(self,
                 filters,
                 kernel_size,
                 strides,
                 weights_decay=0.,
                 data_format='channels_last',
                 **kwargs):
        super(Conv2dFixedPadding, self).__init__(**kwargs)
        if strides > 1:
            self.fixed_padding = FixedPadding(kernel_size, data_format=data_format)
        else:
            self.fixed_padding = None
        self.conv2d = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=('SAME' if strides == 1 else 'VALID'),
            use_bias=False,
            kernel_initializer=tf.keras.initializers.VarianceScaling(),
            kernel_regularizer=tf.keras.regularizers.l2(weights_decay),
            data_format=data_format)

    def call(self, inputs, training=None, **kwargs):
        if self.fixed_padding:
            inputs = self.fixed_padding(inputs, training=training)
        return self.conv2d(inputs, training=training)


class IdentityLayer(tf.keras.layers.Layer):

    def call(self, inputs, training=None, **kwargs):
        return tf.identity(inputs)


class SK_Conv2D(tf.keras.layers.Layer):  # pylint: disable=invalid-name
    """Selective kernel convolutional layer (https://arxiv.org/abs/1903.06586)."""

    def __init__(self,
                 filters,
                 strides,
                 sk_ratio,
                 weights_decay=0.,
                 min_dim=32,
                 data_format='channels_last',
                 **kwargs):
        super(SK_Conv2D, self).__init__(**kwargs)
        self.data_format = data_format
        self.filters = filters
        self.sk_ratio = sk_ratio
        self.min_dim = min_dim

        # Two stream conv (using split and both are 3x3).
        self.conv2d_fixed_padding = Conv2dFixedPadding(
            weights_decay=weights_decay,
            filters=2 * filters,
            kernel_size=3,
            strides=strides,
            data_format=data_format)
        self.batch_norm_relu = BatchNormRelu(data_format=data_format)

        # Mixing weights for two streams.
        mid_dim = max(int(filters * sk_ratio), min_dim)
        self.conv2d_0 = tf.keras.layers.Conv2D(
            filters=mid_dim,
            kernel_size=1,
            strides=1,
            kernel_initializer=tf.keras.initializers.VarianceScaling(),
            use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l2(weights_decay),
            data_format=data_format)
        self.batch_norm_relu_1 = BatchNormRelu(data_format=data_format)
        self.conv2d_1 = tf.keras.layers.Conv2D(
            filters=2 * filters,
            kernel_size=1,
            strides=1,
            kernel_initializer=tf.keras.initializers.VarianceScaling(),
            kernel_regularizer=tf.keras.regularizers.l2(weights_decay),
            use_bias=False,
            data_format=data_format)

    def call(self, inputs, training=None, **kwargs):
        channel_axis = 1 if self.data_format == 'channels_first' else 3
        pooling_axes = [2, 3] if self.data_format == 'channels_first' else [1, 2]

        # Two stream conv (using split and both are 3x3).
        inputs = self.conv2d_fixed_padding(inputs, training=training)
        inputs = self.batch_norm_relu(inputs, training=training)
        inputs = tf.stack(tf.split(inputs, num_or_size_splits=2, axis=channel_axis))

        # Mixing weights for two streams.
        global_features = tf.reduce_mean(
            tf.reduce_sum(inputs, axis=0), pooling_axes, keepdims=True)
        global_features = self.conv2d_0(global_features, training=training)
        global_features = self.batch_norm_relu_1(global_features, training=training)
        mixing = self.conv2d_1(global_features, training=training)
        mixing = tf.stack(tf.split(mixing, num_or_size_splits=2, axis=channel_axis))
        mixing = tf.nn.softmax(mixing, axis=0)

        return tf.reduce_sum(inputs * mixing, axis=0)


class SE_Layer(tf.keras.layers.Layer):  # pylint: disable=invalid-name
    """Squeeze and Excitation layer (https://arxiv.org/abs/1709.01507)."""

    def __init__(self, filters, se_ratio, weights_decay=0., data_format='channels_last', **kwargs):
        super(SE_Layer, self).__init__(**kwargs)
        self.data_format = data_format
        self.se_reduce = tf.keras.layers.Conv2D(
            max(1, int(filters * se_ratio)),
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=tf.keras.initializers.VarianceScaling(),
            kernel_regularizer=tf.keras.regularizers.l2(weights_decay),
            bias_regularizer=tf.keras.regularizers.l2(weights_decay),

            padding='same',
            data_format=data_format,
            use_bias=True)
        self.se_expand = tf.keras.layers.Conv2D(
            None,  # This is filled later in build().
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=tf.keras.initializers.VarianceScaling(),
            kernel_regularizer=tf.keras.regularizers.l2(weights_decay),
            bias_regularizer=tf.keras.regularizers.l2(weights_decay),

            padding='same',
            data_format=data_format,
            use_bias=True)

    def build(self, input_shape):
        self.se_expand.filters = input_shape[-1]
        super(SE_Layer, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        spatial_dims = [2, 3] if self.data_format == 'channels_first' else [1, 2]
        se_tensor = tf.reduce_mean(inputs, spatial_dims, keepdims=True)
        se_tensor = self.se_expand(tf.nn.relu(self.se_reduce(se_tensor)))
        return tf.sigmoid(se_tensor) * inputs


class ResidualBlock(tf.keras.layers.Layer):  # pylint: disable=missing-docstring

    def __init__(self,
                 filters,
                 strides,
                 use_projection=False,
                 data_format='channels_last',
                 drop_block_keep_prob=None,
                 drop_block_size=None,
                 weights_decay=0.,
                 se_ratio=0.,
                 sk_ratio=0.,
                 **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        del drop_block_keep_prob
        del drop_block_size
        self.conv2d_bn_layers = []
        self.shortcut_layers = []
        if use_projection:
            if sk_ratio > 0:  # Use ResNet-D (https://arxiv.org/abs/1812.01187)
                if strides > 1:
                    self.shortcut_layers.append(FixedPadding(2, data_format))
                self.shortcut_layers.append(
                    tf.keras.layers.AveragePooling2D(
                        pool_size=2,
                        strides=strides,
                        padding='SAME' if strides == 1 else 'VALID',
                        data_format=data_format))
                self.shortcut_layers.append(
                    Conv2dFixedPadding(
                        weights_decay=weights_decay,
                        filters=filters,
                        kernel_size=1,
                        strides=1,
                        data_format=data_format))
            else:
                self.shortcut_layers.append(
                    Conv2dFixedPadding(
                        weights_decay=weights_decay,

                        filters=filters,
                        kernel_size=1,
                        strides=strides,
                        data_format=data_format)),
            self.shortcut_layers.append(
                BatchNormRelu(relu=False, data_format=data_format))

        self.conv2d_bn_layers.append(
            Conv2dFixedPadding(
                weights_decay=weights_decay,

                filters=filters,
                kernel_size=3,
                strides=strides,
                data_format=data_format))
        self.conv2d_bn_layers.append(BatchNormRelu(data_format=data_format))
        self.conv2d_bn_layers.append(
            Conv2dFixedPadding(
                filters=filters, kernel_size=3, strides=1, weights_decay=weights_decay, data_format=data_format))
        self.conv2d_bn_layers.append(
            BatchNormRelu(relu=False, init_zero=True, data_format=data_format))
        if se_ratio > 0:
            self.se_layer = SE_Layer(filters, se_ratio, data_format=data_format)
        self.se_ratio = se_ratio

    def call(self, inputs, training=None, **kwargs):
        shortcut = inputs
        for layer in self.shortcut_layers:
            # Projection shortcut in first layer to match filters and strides
            shortcut = layer(shortcut, training=training)

        for layer in self.conv2d_bn_layers:
            inputs = layer(inputs, training=training)

        if self.se_ratio > 0:
            inputs = self.se_layer(inputs, training=training)

        return tf.nn.relu(inputs + shortcut)


class BottleneckBlock(tf.keras.layers.Layer):
    """BottleneckBlock."""

    def __init__(self,
                 filters,
                 strides,
                 use_projection=False,
                 data_format='channels_last',
                 drop_block_keep_prob=None,
                 drop_block_size=None,
                 sk_ratio=0.,
                 se_ratio=0.,
                 weights_decay=0.,
                 **kwargs):
        super(BottleneckBlock, self).__init__(**kwargs)
        self.projection_layers = []
        self.sk_ratio = sk_ratio
        self.se_ratio = se_ratio
        if use_projection:
            filters_out = 4 * filters
            if sk_ratio > 0:  # Use ResNet-D (https://arxiv.org/abs/1812.01187)
                if strides > 1:
                    self.projection_layers.append(FixedPadding(2, data_format))
                self.projection_layers.append(
                    tf.keras.layers.AveragePooling2D(
                        pool_size=2,
                        strides=strides,
                        padding='SAME' if strides == 1 else 'VALID',
                        data_format=data_format))
                self.projection_layers.append(
                    Conv2dFixedPadding(
                        weights_decay=weights_decay,
                        filters=filters_out,
                        kernel_size=1,
                        strides=1,
                        data_format=data_format))
            else:
                self.projection_layers.append(
                    Conv2dFixedPadding(
                        weights_decay=weights_decay,
                        filters=filters_out,
                        kernel_size=1,
                        strides=strides,
                        data_format=data_format))
            self.projection_layers.append(
                BatchNormRelu(relu=False, data_format=data_format))
        self.shortcut_drop_block = DropBlock(
            data_format=data_format,
            keep_prob=drop_block_keep_prob,
            drop_block_size=drop_block_size)

        self.conv_relu_drop_block_layers = []

        self.conv_relu_drop_block_layers.append(
            Conv2dFixedPadding(
                filters=filters, kernel_size=1, strides=1, weights_decay=weights_decay, data_format=data_format))
        self.conv_relu_drop_block_layers.append(
            BatchNormRelu(data_format=data_format))
        self.conv_relu_drop_block_layers.append(
            DropBlock(
                data_format=data_format,
                keep_prob=drop_block_keep_prob,
                drop_block_size=drop_block_size))

        if sk_ratio > 0:
            self.conv_relu_drop_block_layers.append(
                SK_Conv2D(filters, strides, sk_ratio, weights_decay=weights_decay, data_format=data_format))
        else:
            self.conv_relu_drop_block_layers.append(
                Conv2dFixedPadding(
                    weights_decay=weights_decay,
                    filters=filters,
                    kernel_size=3,
                    strides=strides,
                    data_format=data_format))
            self.conv_relu_drop_block_layers.append(
                BatchNormRelu(data_format=data_format))
        self.conv_relu_drop_block_layers.append(
            DropBlock(
                data_format=data_format,
                keep_prob=drop_block_keep_prob,
                drop_block_size=drop_block_size))

        self.conv_relu_drop_block_layers.append(
            Conv2dFixedPadding(
                weights_decay=weights_decay,
                filters=4 * filters,
                kernel_size=1,
                strides=1,
                data_format=data_format))
        self.conv_relu_drop_block_layers.append(
            BatchNormRelu(relu=False, init_zero=True, data_format=data_format))
        self.conv_relu_drop_block_layers.append(
            DropBlock(
                data_format=data_format,
                keep_prob=drop_block_keep_prob,
                drop_block_size=drop_block_size))

        if self.se_ratio > 0:
            self.conv_relu_drop_block_layers.append(
                SE_Layer(filters, self.se_ratio, data_format=data_format))

    def call(self, inputs, training=None, **kwargs):
        shortcut = inputs
        for layer in self.projection_layers:
            shortcut = layer(shortcut, training=training)
        shortcut = self.shortcut_drop_block(shortcut, training=training)

        for layer in self.conv_relu_drop_block_layers:
            inputs = layer(inputs, training=training)

        return tf.nn.relu(inputs + shortcut)


class BlockGroup(tf.keras.layers.Layer):  # pylint: disable=missing-docstring

    def __init__(self,
                 filters,
                 block_fn,
                 blocks,
                 strides,
                 data_format='channels_last',
                 drop_block_keep_prob=None,
                 drop_block_size=None,
                 sk_ratio=0.,
                 se_ratio=0.,
                 weights_decay=0.,
                 **kwargs):
        self._name = kwargs.get('name')
        super(BlockGroup, self).__init__(**kwargs)

        self.layers = []
        self.layers.append(
            block_fn(
                filters,
                strides,
                weights_decay=weights_decay,
                use_projection=True,
                data_format=data_format,
                drop_block_keep_prob=drop_block_keep_prob,
                drop_block_size=drop_block_size,
                sk_ratio=sk_ratio,
                se_ratio=se_ratio,
            ))

        for _ in range(1, blocks):
            self.layers.append(
                block_fn(
                    filters,
                    1,
                    weights_decay=weights_decay,
                    data_format=data_format,
                    drop_block_keep_prob=drop_block_keep_prob,
                    drop_block_size=drop_block_size))

    def call(self, inputs, training=None, **kwargs):
        for layer in self.layers:
            inputs = layer(inputs, training=training)
        return tf.identity(inputs, self._name)


class Resnet(tf.keras.models.Model):  # pylint: disable=missing-docstring

    def get_config(self):
        return {'train_mode':self.train_mode}

    def __init__(self,
                 block_fn,
                 layers,
                 width_multiplier,
                 cifar_stem=False,
                 data_format='channels_last',
                 drop_block_keep_probs=None,
                 drop_block_size=None,
                 train_mode=None,
                 fine_tune_after_block=-1,
                 sk_ratio=0.,
                 se_ratio=0.,
                 weights_decay=0.,
                 **kwargs):
        super(Resnet, self).__init__(**kwargs)
        self.train_mode = train_mode
        self.fine_tune_after_block = fine_tune_after_block
        self.data_format = data_format
        if drop_block_keep_probs is None:
            drop_block_keep_probs = [None] * 4
        if not isinstance(drop_block_keep_probs,
                          list) or len(drop_block_keep_probs) != 4:
            raise ValueError('drop_block_keep_probs is not valid:',
                             drop_block_keep_probs)
        trainable = (
                train_mode != 'finetune' or fine_tune_after_block == -1)
        self.initial_conv_relu_max_pool = []
        if cifar_stem:
            self.initial_conv_relu_max_pool.append(
                Conv2dFixedPadding(
                    weights_decay=weights_decay,
                    filters=64 * width_multiplier,
                    kernel_size=3,
                    strides=1,
                    data_format=data_format,
                    trainable=trainable))
            self.initial_conv_relu_max_pool.append(
                IdentityLayer(name='initial_conv', trainable=trainable))
            self.initial_conv_relu_max_pool.append(
                BatchNormRelu(data_format=data_format, trainable=trainable))
            self.initial_conv_relu_max_pool.append(
                IdentityLayer(name='initial_max_pool', trainable=trainable))
        else:
            if sk_ratio > 0:  # Use ResNet-D (https://arxiv.org/abs/1812.01187)
                self.initial_conv_relu_max_pool.append(
                    Conv2dFixedPadding(
                        weights_decay=weights_decay,
                        filters=64 * width_multiplier // 2,
                        kernel_size=3,
                        strides=2,
                        data_format=data_format,
                        trainable=trainable))
                self.initial_conv_relu_max_pool.append(
                    BatchNormRelu(data_format=data_format, trainable=trainable))
                self.initial_conv_relu_max_pool.append(
                    Conv2dFixedPadding(
                        weights_decay=weights_decay,
                        filters=64 * width_multiplier // 2,
                        kernel_size=3,
                        strides=1,
                        data_format=data_format,
                        trainable=trainable))
                self.initial_conv_relu_max_pool.append(
                    BatchNormRelu(data_format=data_format, trainable=trainable))
                self.initial_conv_relu_max_pool.append(
                    Conv2dFixedPadding(
                        weights_decay=weights_decay,
                        filters=64 * width_multiplier,
                        kernel_size=3,
                        strides=1,
                        data_format=data_format,
                        trainable=trainable))
            else:
                self.initial_conv_relu_max_pool.append(
                    Conv2dFixedPadding(
                        weights_decay=weights_decay,
                        filters=64 * width_multiplier,
                        kernel_size=7,
                        strides=2,
                        data_format=data_format,
                        trainable=trainable))
            self.initial_conv_relu_max_pool.append(
                IdentityLayer(name='initial_conv', trainable=trainable))
            self.initial_conv_relu_max_pool.append(
                BatchNormRelu(data_format=data_format, trainable=trainable))

            self.initial_conv_relu_max_pool.append(
                tf.keras.layers.MaxPooling2D(
                    pool_size=3,
                    strides=2,
                    padding='SAME',
                    data_format=data_format,
                    trainable=trainable))
            self.initial_conv_relu_max_pool.append(
                IdentityLayer(name='initial_max_pool', trainable=trainable))

        self.block_groups = []
        # fine_tune_after_block != 4. In that case earlier BN stats were getting
        # updated. Now they will not be. Check with Ting to make sure this is ok.
        if train_mode == 'finetune' and fine_tune_after_block == 0:
            trainable = True

        self.block_groups.append(
            BlockGroup(
                weights_decay=weights_decay,
                filters=64 * width_multiplier,
                block_fn=block_fn,
                blocks=layers[0],
                strides=1,
                name='block_group1',
                data_format=data_format,
                drop_block_keep_prob=drop_block_keep_probs[0],
                drop_block_size=drop_block_size,
                trainable=trainable,
                sk_ratio=sk_ratio,
                se_ratio=se_ratio))

        if train_mode == 'finetune' and fine_tune_after_block == 1:
            trainable = True

        self.block_groups.append(
            BlockGroup(
                filters=128 * width_multiplier,
                block_fn=block_fn,
                blocks=layers[1],
                strides=2,
                name='block_group2',
                data_format=data_format,
                drop_block_keep_prob=drop_block_keep_probs[1],
                drop_block_size=drop_block_size,
                trainable=trainable))

        if train_mode == 'finetune' and fine_tune_after_block == 2:
            trainable = True

        self.block_groups.append(
            BlockGroup(
                filters=256 * width_multiplier,
                block_fn=block_fn,
                blocks=layers[2],
                strides=2,
                name='block_group3',
                data_format=data_format,
                drop_block_keep_prob=drop_block_keep_probs[2],
                drop_block_size=drop_block_size,
                trainable=trainable))

        if train_mode == 'finetune' and fine_tune_after_block == 3:
            trainable = True

        self.block_groups.append(
            BlockGroup(
                filters=512 * width_multiplier,
                block_fn=block_fn,
                blocks=layers[3],
                strides=2,
                name='block_group4',
                data_format=data_format,
                drop_block_keep_prob=drop_block_keep_probs[3],
                drop_block_size=drop_block_size,
                trainable=trainable))

    def call(self, inputs, training=None, **kwargs):

        for layer in self.initial_conv_relu_max_pool:
            inputs = layer(inputs, training=training)

        for i, layer in enumerate(self.block_groups):
            if self.train_mode == 'finetune' and self.fine_tune_after_block == i:
                inputs = tf.stop_gradient(inputs)
            inputs = layer(inputs, training=training)
        if self.train_mode == 'finetune' and self.fine_tune_after_block == 4:
            inputs = tf.stop_gradient(inputs)
        if self.data_format == 'channels_last':
            inputs = tf.reduce_mean(inputs, [1, 2])
        else:
            inputs = tf.reduce_mean(inputs, [2, 3])

        inputs = tf.identity(inputs, 'final_avg_pool')
        return inputs


def resnet(resnet_depth,
           width_multiplier,
           cifar_stem=False,
           data_format='channels_last',
           drop_block_keep_probs=None,
           drop_block_size=None,
           train_mode=None,
           weights_decay=0.,
           fine_tune_after_block=-1,
           sk_ratio=0.,
           se_ratio=0.,
           ):
    """Returns the ResNet model for a given size and number of output classes."""
    model_params = {
        18: {
            'block': ResidualBlock,
            'layers': [2, 2, 2, 2]
        },
        34: {
            'block': ResidualBlock,
            'layers': [3, 4, 6, 3]
        },
        50: {
            'block': BottleneckBlock,
            'layers': [3, 4, 6, 3]
        },
        101: {
            'block': BottleneckBlock,
            'layers': [3, 4, 23, 3]
        },
        152: {
            'block': BottleneckBlock,
            'layers': [3, 8, 36, 3]
        },
        200: {
            'block': BottleneckBlock,
            'layers': [3, 24, 36, 3]
        }
    }

    if resnet_depth not in model_params:
        raise ValueError('Not a valid resnet_depth:', resnet_depth)

    params = model_params[resnet_depth]
    return Resnet(
        train_mode=train_mode,
        fine_tune_after_block=fine_tune_after_block,
        sk_ratio=sk_ratio,
        se_ratio=se_ratio,
        block_fn=params['block'],
        layers=params['layers'],
        width_multiplier=width_multiplier,
        cifar_stem=cifar_stem,
        drop_block_keep_probs=drop_block_keep_probs,
        drop_block_size=drop_block_size,
        data_format=data_format,
        weights_decay=weights_decay)
