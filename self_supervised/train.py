"""Train from scratch or fine-tune SSL model with tf and keras"""
import datetime
from typing import Tuple, List

import tensorflow as tf

from self_supervised.SimCLR.SimCLRModel import SimCLR
from self_supervised.data_reader import read_cifar10
from self_supervised.loss import nt_xent_func


def load_callbacks(train_path: str, save_freq: int) -> Tuple[List[tf.keras.callbacks.Callback], str]:
    """Load tensorboard and checkpoint callbacks based on the log dir
    Args:
        :param save_freq: The interval for saving checkpoints
        :param train_path: the path to store the model
     """
    logdir = train_path + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    file_writer = tf.summary.create_file_writer(logdir + "/metrics")
    file_writer.set_as_default()
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    checkpoint_path = f"{logdir}/cp.ckpt"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1, save_freq=save_freq)
    callbacks = [tensorboard_callback, cp_callback]
    return callbacks, logdir


def pretrain_func(encoder: tf.keras.Model, decoder: tf.keras.Model, config):
    """Pretrains the model based on config settings
    First it load cifar data (with augmentation), then create the SimCLR model from the encoder and decoder
    and then train the  model
    Args:
        :param encoder: The encoder for the simCLR model
        :param decoder: The decoder for the simCLR model
        :param config: dict with all the configs
    """
    load_data_func = None
    if config.dataset == 'cifar10':
        load_data_func = read_cifar10
    train_dataset, test_dataset, train_num_of_examples = load_data_func(batch_size=config.batch_size,
                                                                        num_epochs=config.num_epochs,
                                                                        task=config.task, config_augment=config)
    iterations_per_epoch = train_num_of_examples // config.batch_size
    total_iterations = iterations_per_epoch * config.num_epochs
    learning_rate = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=config.learning_rate,
                                                              decay_steps=total_iterations)

    optimizer = tf.keras.optimizers.SGD(learning_rate, momentum=config.momentum)
    model = SimCLR(encoder=encoder, decoder=decoder)
    loss = nt_xent_func(temperature=config.temperature)

    model.compile(loss=loss, optimizer=optimizer)
    callbacks, logdir = load_callbacks(train_path=config.train_file_path, save_freq=config.save_freq)
    # model.run_eagerly = True
    print(f'Log dir: {logdir}')
    model.fit(train_dataset, validation_data=train_dataset, validation_steps=10, steps_per_epoch=200, epochs=2,
              callbacks=callbacks)


def finetune_func(encoder: tf.keras.Model, decoder: tf.keras.Model, readout_model: tf.keras.Model, config):
    """fine-tuning the given model
    Args:
        :param encoder: The encoder for the simCLR model
        :param decoder: The decoder for the simCLR model
        :param readout_model: The readout for the simCLR model
        :param config: dict with all the configs
    """
    load_data_func = None
    if config.dataset == 'cifar10':
        load_data_func = read_cifar10
    train_dataset, test_dataset, train_num_of_examples = load_data_func(batch_size=config.batch_size,
                                                                        num_epochs=config.num_epochs,
                                                                        task=config.task, config_augment=config)
    iterations_per_epoch = train_num_of_examples // config.batch_size
    total_iterations = iterations_per_epoch * config.num_epochs
    model = SimCLR(encoder=encoder, decoder=decoder)
    learning_rate = tf.keras.experimental.CosineDecay(config.learning_rate, total_iterations)
    model.load_weights(config.logdir_model).expect_partial()
    # train only the readout
    if config.freeze:
        model.trainable = False
    model_new = tf.keras.Sequential([model.encoder, readout_model])
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=config.momentum)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model_new.compile(loss=loss, optimizer=optimizer, metrics='acc')
    callbacks, logdir = load_callbacks(train_path=config.train_file_path, save_freq=config.save_freq)
    #model_new.run_eagerly = True
    model_new.fit(train_dataset, validation_data=test_dataset, callbacks=callbacks)
