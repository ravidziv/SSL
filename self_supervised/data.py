"""Data pipeline."""

import functools
import logging
from typing import Tuple
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_datasets.core import DatasetBuilder
import self_supervised.data_utils as data_utils


def build_input_fn(builder: DatasetBuilder, global_batch_size: int, topology: tf.tpu.experimental.Topology,
                   is_training: bool, train_mode: str, train_split: str, eval_split: str,
                   cache_dataset: bool, image_size: int, color_jitter_strength: float):
    """Build input function.
    :param color_jitter_strength: the strength of the jitter for each image
    :param image_size: the size of the image
    :param eval_split: the split for eval to take
    :param train_split: the split for train to take
    :param train_mode: pretrain or classification
    :param builder: FDS builder for specified dataset.
    :param cache_dataset: if we want to cached the dataset
    :param global_batch_size: Global batch size.
    :param topology: An instance of `tf.tpu.experimental.Topology` or None.
    :param is_training: Whether to build in training mode.

    Returns:
      A function that accepts a dict of params and returns a tuple of images and
      features, to be used as the input_fn in TPUEstimator.

    """

    def _input_fn(input_context):
        """Inner input function."""
        batch_size = input_context.get_per_replica_batch_size(global_batch_size)
        logging.info('Global batch size: %d', global_batch_size)
        logging.info('Per-replica batch size: %d', batch_size)
        preprocess_fn_pretrain = get_preprocess_fn(is_training, is_pretrain=True, image_size=image_size,
                                                   color_jitter_strength=color_jitter_strength)
        preprocess_fn_finetune = get_preprocess_fn(is_training, is_pretrain=False, image_size=image_size,
                                                   color_jitter_strength=color_jitter_strength)
        num_classes = builder.info.features['label'].num_classes

        def map_fn(image, label):
            """Produces multiple transformations of the same batch."""
            if is_training and train_mode == 'pretrain':
                xs = []
                for _ in range(2):  # Two transformations
                    xs.append(preprocess_fn_pretrain(image))
                image = tf.concat(xs, -1)
            else:
                image = preprocess_fn_finetune(image)
            label = tf.one_hot(label, num_classes)
            return image, label

        logging.info('num_input_pipelines: %d', input_context.num_input_pipelines)
        dataset = builder.as_dataset(
            split=train_split if is_training else eval_split,
            shuffle_files=is_training,
            as_supervised=True,
            # Passing the input_context to TFDS makes TFDS read different parts
            # of the dataset on different workers. We also adjust the interleave
            # parameters to achieve better performance.
            read_config=tfds.ReadConfig(
                interleave_cycle_length=32,
                interleave_block_length=1,
                input_context=input_context))
        if cache_dataset:
            dataset = dataset.cache()
        if is_training:
            options = tf.data.Options()
            options.experimental_deterministic = False
            options.experimental_slack = True
            dataset = dataset.with_options(options)
            buffer_multiplier = 50 if image_size <= 32 else 10
            dataset = dataset.shuffle(batch_size * buffer_multiplier)
            dataset = dataset.repeat(-1)
        dataset = dataset.map(
            map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        # dataset = dataset.batch(batch_size, drop_remainder=is_training)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

    return _input_fn


def build_distributed_dataset(builder: DatasetBuilder, batch_size: int, topology: tf.tpu.experimental.Topology,
                              train_mode: str, train_split: str, eval_split: str, cache_dataset: bool, image_size: int,
                              strategy: tf.distribute.Strategy, color_jitter_strength: float, is_training: bool) \
        -> tf.data.Dataset:
    """

    :param batch_size:
    :param builder:
    :param is_training:
    :param strategy:
    :param topology:
    :param train_mode:
    :param train_split:
    :param eval_split:
    :param cache_dataset:
    :param image_size:
    :param color_jitter_strength:
    :return:
    """
    input_fn = build_input_fn(builder=builder, topology=topology, is_training=is_training,
                              train_mode=train_mode, train_split=train_split, eval_split=eval_split,
                              cache_dataset=cache_dataset, image_size=image_size,
                              color_jitter_strength=color_jitter_strength, global_batch_size=batch_size)
    return strategy.distribute_datasets_from_function(input_fn)


def get_preprocess_fn(is_training: bool, is_pretrain: bool, image_size: int, color_jitter_strength: float):
    """Get function that accepts an image and returns a preprocessed image.
    :param is_training: if it is a train dataset or eval
    :param is_pretrain: if we want to to self supervised pre-training
    :param image_size: the size of the image
    :param color_jitter_strength:  the strength of the jitter for each image
    :return:
    """
    # Disable test cropping for small images (e.g. CIFAR)
    if image_size <= 32:
        test_crop = False
    else:
        test_crop = True
    return functools.partial(
        data_utils.preprocess_image,
        height=image_size,
        width=image_size,
        is_training=is_training,
        color_distort=is_pretrain,
        test_crop=test_crop,
        color_jitter_strength=color_jitter_strength)


def read_dataset(train_batch_size: int = 32, eval_batch_size: int = 128,
                 train_mode: str = 'pretrain', strategy: tf.distribute.Strategy = None, topology=None,
                 dataset: str = 'cifar10', train_split: str = 'train', eval_split: str = 'test',
                 data_dir: str = None, image_size: int = 32, cache_dataset: bool = True,
                 color_jitter_strength: float = 1.0) -> Tuple[tf.data.Dataset, tf.data.Dataset, int, int, int]:
    """Reads dataset from tfds and processes it either for pretraining including augmentation
    :type color_jitter_strength: object
    :param train_batch_size: batch size of train dataset
    :param eval_batch_size: batch size of eval dataset
    :param train_mode: pretrain ir supervised classification
    :param strategy: the strategy for distribute leaning over gpus/cpus
    :param topology:  if we want to use TPU
    :param dataset: which dataset to load - cifar or imagenet
    :param train_split: which split take for the training
    :param eval_split: which split take for the eval
    :param data_dir: where to store/load the dataset
    :param image_size: the size of image
    :param cache_dataset: if we want to cache the data before using it
    :param color_jitter_strength: the strength  of the jitter
    :return:
        train_dataset: tf dataset with augmented train features
        test_dataset: tf dataset with augmented test features
        num_train_examples: the size of the train dataset
        num_eval_examples : the size of the test dataset
        num_classes : number of classes in the data
    """
    builder: DatasetBuilder = tfds.builder(dataset, data_dir=data_dir)
    builder.download_and_prepare()
    num_train_examples = builder.info.splits[train_split].num_examples
    num_eval_examples = builder.info.splits[eval_split].num_examples
    num_classes = builder.info.features['label'].num_classes
    train_dataset = build_distributed_dataset(builder=builder, batch_size=train_batch_size, is_training=True,
                                              strategy=strategy, topology=topology, train_mode=train_mode,
                                              train_split=train_split, eval_split=eval_split,
                                              cache_dataset=cache_dataset,
                                              image_size=image_size, color_jitter_strength=color_jitter_strength)
    test_dataset = build_distributed_dataset(builder=builder, batch_size=eval_batch_size, is_training=False,
                                             strategy=strategy, topology=topology, train_mode=train_mode,
                                             train_split=train_split, eval_split=eval_split,
                                             cache_dataset=cache_dataset,
                                             image_size=image_size, color_jitter_strength=color_jitter_strength)
    return train_dataset, test_dataset, num_train_examples, num_eval_examples, num_classes
