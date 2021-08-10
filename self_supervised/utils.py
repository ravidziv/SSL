"""Utils for self supervised learning"""
import math
from typing import Tuple, List
import tensorflow as tf
from self_supervised import lars_optimizer


def load_callbacks(logdir: str, save_freq: int) -> Tuple[List[tf.keras.callbacks.Callback], str]:
    """Load tensorboard and checkpoint callbacks based on the log dir
    Args:
        :param logdir: the path to store the model
        :param save_freq: The interval for saving checkpoints
     """
    file_writer = tf.summary.create_file_writer(logdir + "/metrics")
    file_writer.set_as_default()
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, profile_batch=0)
    checkpoint_path = f"{logdir}/cp.ckpt"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1, save_freq=save_freq)
    callbacks = [tensorboard_callback, cp_callback]
    return callbacks


def build_optimizers(contrastive_learning_rate, probe_learning_rate: float, contrastive_optimizer_type: str,
                     probe_optimizer_type: str, contrastive_momentum: float, probe_momentum: float,
                     contrastive_weight_decay: float = None, probe_weight_decay: float = None) \
        -> Tuple[tf.keras.optimizers.Optimizer, tf.keras.optimizers.Optimizer]:
    """Returns the optimizers of the contrastive learning task and for the supervised head.
    :rtype: Tuple[tf.keras.optimizers.Optimizer, tf.keras.optimizers.Optimizer]
    """
    contrastive_optimizer = build_optimizer(learning_rate=contrastive_learning_rate,
                                            optimizer=contrastive_optimizer_type,
                                            momentum=contrastive_momentum, weight_decay=contrastive_weight_decay)
    probe_optimizer = build_optimizer(learning_rate=probe_learning_rate,
                                      optimizer=probe_optimizer_type,
                                      momentum=probe_momentum, weight_decay=probe_weight_decay)
    return contrastive_optimizer, probe_optimizer


def build_optimizer(learning_rate: float, optimizer: str, momentum: float,
                    weight_decay: float = None) -> tf.keras.optimizers.Optimizer:
    """Returns an optimizer.
    :param learning_rate: the learning rate for the optimizer
    :param optimizer:  the type of the optimizer (Adam, SGD...)
    :param momentum: The momentum parameter for the optimizer
    :param weight_decay: Add weight decay for the LARSO optimizer
    :return: The new optimizer
    """
    if optimizer == 'momentum':
        return tf.keras.optimizers.SGD(learning_rate, momentum, nesterov=True)
    elif optimizer == 'adam':
        return tf.keras.optimizers.Adam(learning_rate)
    elif optimizer == 'lars':
        return lars_optimizer.LARSOptimizer(
            learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            exclude_from_weight_decay=[
                'batch_normalization', 'bias', 'head_supervised'
            ])
    else:
        raise ValueError('Unknown optimizer {}'.format(optimizer))


def get_train_steps(num_examples: int, train_steps: int, train_epochs: int, train_batch_size: int) -> int:
    """Determine the number of training steps."""
    return train_steps or (
            num_examples * train_epochs // train_batch_size + 1)


class WarmUpAndCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Applies a warmup schedule on a given learning rate decay schedule."""

    def __init__(self, base_learning_rate: float, num_examples: int, warmup_epochs: int, train_batch_size: int,
                 learning_rate_scaling: str, train_steps: int, train_epochs: int, name: str = None):
        """

        :param base_learning_rate: the initial learning rate
        :param num_examples: number of total examples
        :param warmup_epochs: the number of epochs for the initial warm up scheduling
        :param train_batch_size: the batch size
        :param learning_rate_scaling: linear or sqrt update
        :param train_steps:  total number of train steps
        :param train_epochs: number of epochs
        :param name: if we want a name for the scheduler
        """
        super(WarmUpAndCosineDecay, self).__init__()
        self.base_learning_rate = base_learning_rate
        self.num_examples = num_examples
        self._name = name
        self.warmup_epochs = warmup_epochs
        self.train_batch_size = train_batch_size
        self.learning_rate_scaling = learning_rate_scaling
        self.train_steps = train_steps
        self.train_epochs = train_epochs

    def __call__(self, step):
        with tf.name_scope(self._name or 'WarmUpAndCosineDecay'):
            warmup_steps = int(
                round(self.warmup_epochs * self.num_examples //
                      self.train_batch_size))
            if self.learning_rate_scaling == 'linear':
                scaled_lr = self.base_learning_rate * self.train_batch_size / 256.
            elif self.learning_rate_scaling == 'sqrt':
                scaled_lr = self.base_learning_rate * math.sqrt(self.train_batch_size)
            else:
                raise ValueError('Unknown learning rate scaling {}'.format(
                    self.learning_rate_scaling))
            learning_rate = (
                step / float(warmup_steps) * scaled_lr if warmup_steps else scaled_lr)
            # Cosine decay learning rate schedule
            total_steps = get_train_steps(num_examples=self.num_examples,
                                          train_steps=self.train_steps, train_epochs=self.train_epochs,
                                          train_batch_size=self.train_batch_size)
            cosine_decay = tf.keras.experimental.CosineDecay(
                scaled_lr, total_steps - warmup_steps)
            learning_rate = tf.where(step < warmup_steps, learning_rate,
                                     cosine_decay(step - warmup_steps))
            return learning_rate

    def get_config(self):
        return {
            'base_learning_rate': self.base_learning_rate,
            'num_examples': self.num_examples,
        }
