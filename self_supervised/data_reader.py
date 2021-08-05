from typing import Tuple
import tensorflow as tf
from self_supervised.image_utils import AugmentationModel


def make_datasets(images: tf.Tensor, labels: tf.Tensor, task: str, config_augment: dict,
                  num_epochs: int, batch_size: int) -> tf.data.Dataset:
    """Make tf datasets from the given iamges and labels"""
    if task == 'pretrain':
        augmentation_model = AugmentationModel(config_augment)
        data = tf.data.Dataset.from_tensor_slices(images)
        data = data.shuffle(5000, reshuffle_each_iteration=True)
        data = data.map(lambda i: tf.cast(i, tf.float32) / 255)
        data = data.batch(batch_size, drop_remainder=True).repeat(num_epochs)
        data = data.map(lambda x: augmentation_model(x, training=True),
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    elif task == 'classification':
        data = tf.data.Dataset.from_tensor_slices((images, labels))
        data = data.shuffle(5000, reshuffle_each_iteration=True)
        data = data.map(lambda x, y: (tf.cast(x, tf.float32) / 255, y))
        data = data.batch(batch_size, drop_remainder=True).repeat(num_epochs)
    return data


def read_cifar10(batch_size: int = 32, num_epochs: int = 1, task: str = 'pretrain', config_augment: dict = None) -> \
        Tuple[tf.data.Dataset, tf.data.Dataset, int]:
    """Reads cifar10 data and and processes data either for pretraining or classification

    Args:
        batch_size: batch size for the given dataset
        num_epochs: Epoch at which epoch counter finishes
        config_augment: dict for all the configs
        task:  'pretrain or classification task
    Returns:
        data: tf dataset with augmented cifar10 images
    """
    (cifar10_images, cifar10_labels), (cifar10_images_test, cifar10_labels_test) = tf.keras.datasets.cifar10.load_data()
    num_of_examples = cifar10_images.shape[0]
    train_dataset = make_datasets(images=cifar10_images, labels=cifar10_labels, task=task,
                                  config_augment=config_augment, num_epochs=num_epochs, batch_size=batch_size)

    test_dataset = make_datasets(images=cifar10_images_test, labels=cifar10_labels_test, task=task,
                                 config_augment=config_augment, num_epochs=num_epochs, batch_size=batch_size)
    return train_dataset, test_dataset, num_of_examples
