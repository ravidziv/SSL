"""Train from scratch or fine-tune SSL model with tf and keras"""
import math
import os
from datetime import datetime

import tensorflow as tf

import self_supervised.utils as utils
from bayesian_deep_learning.SWAG.callbacks import SWAGCallback
from bayesian_deep_learning.SWAG.models import SWAG_model
from self_supervised.SimCLR.SimCLRModel import SimCLR
from self_supervised.data import read_dataset
from self_supervised.utils import load_callbacks


def pretrain_func(config):
    """Pre trains the model based on config settings
    First it load cifar data (with augmentation), then create the SimCLR model from the encoder and decoder
    and then train the  model
    Args:
        :param config: dict with all the configs
    """
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        logdir = os.path.join(config.log_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
        train_dataset, test_dataset, num_train_examples, num_eval_examples, num_classes = read_dataset(
            train_batch_size=config.train_batch_size,
            color_jitter_strength=config.color_jitter_strength,
            train_mode=config.train_mode,
            eval_batch_size=config.eval_batch_size,
            strategy=strategy,
            image_size=config.image_size,
            cache_dataset=config.cache_dataset)

        steps_per_epoch = config.train_steps or int(
            math.ceil(num_train_examples / config.train_batch_size))

        eval_steps = config.eval_steps or int(
            math.ceil(num_eval_examples / config.eval_batch_size))
        # Get optimizers
        contrastive_learning_rate = utils.WarmUpAndCosineDecay(base_learning_rate=config.learning_rate,
                                                               num_examples=num_train_examples,
                                                               warmup_epochs=config.warmup_epochs,
                                                               train_batch_size=config.train_batch_size,
                                                               learning_rate_scaling=config.learning_rate_scaling,
                                                               train_steps=config.train_steps,
                                                               train_epochs=config.train_epochs)

        contrastive_optimizer, probe_optimizer = utils.build_optimizers(
            contrastive_learning_rate=contrastive_learning_rate,
            probe_learning_rate=config.probe_learning_rate,
            contrastive_optimizer_type=config.contrastive_optimizer,
            probe_optimizer_type=config.probe_optimizer,
            contrastive_momentum=config.momentum, probe_momentum=config.probe_momentum,
            probe_weight_decay=config.probe_weight_decay, contrastive_weight_decay=config.contrastive_weight_decay,
        )
        # The SSL model
        model = SimCLR(num_classes=num_classes,
                       projection_head_args=config.projection_head_args, resent_head_args=config.resent_head_args,
                       train_mode=config.train_mode, lineareval_while_pretraining=config.lineareval_while_pretraining)
        # The callbacks
        callbacks = load_callbacks(logdir=logdir, save_freq=config.save_freq)
        # If we want SWAG train
        if config.use_swag:
            clone_model = model = SimCLR(num_classes=num_classes,
                                         projection_head_args=config.projection_head_args,
                                         resent_head_args=config.resent_head_args,
                                         train_mode=config.train_mode,
                                         lineareval_while_pretraining=config.lineareval_while_pretraining)
            clone_model.set_weights(model.get_weights())
            old_model = model
            old_model.compile(contrastive_optimizer=contrastive_optimizer, probe_optimizer=probe_optimizer,
                              temperature=config.temperature)
            clone_model.compile(contrastive_optimizer=contrastive_optimizer, probe_optimizer=probe_optimizer,
                                temperature=config.temperature)
            model = SWAG_model(base_model=old_model, swag_model=clone_model)
            swag = SWAGCallback(start_epoch=config.swag_start_epoch)
            # swag_sch = SWAGRScheduler(start_epoch=config.swag_start_epoch, lr_schedule=config.swag_lr_schedule,
            #                          swag_lr2=config.swag_lr2, swag_lr=config.swag_lr,
            #                          swag_freq=config.swag_freq)
            callbacks.append(swag)
            model.compile()
        else:
            model.compile(contrastive_optimizer=contrastive_optimizer, probe_optimizer=probe_optimizer,
                          temperature=config.temperature)

        # model.run_eagerly = True
        print(f'Log dir: {logdir}')
        # Train
        model.fit(train_dataset, validation_data=test_dataset, validation_steps=eval_steps,
                  epochs=config.train_epochs, callbacks=callbacks, steps_per_epoch=steps_per_epoch)
