"""Run SSL model (pretrain/ supervised train)"""
import argparse
from self_supervised.train import pretrain_func, finetune_func
from self_supervised.models.resnet_loader import load_ResNet
import tensorflow as tf
import importlib


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def run(config):
    """Runs the model for either pretrain or segmentation/classification tasks
    
    Loads the ResNet backbone and based on task specification performs either
    pretrain or segmentation/classification fine-tuning.

    Args:
        config: instance of config class
    Raises:
        ValueError: In segmentation/classification tasks if number of classes are not specified
    """

    # load the Resnet backbone
    encoder = load_ResNet(config.model, config.imagenet_path, include_top=False, cifar10=config.dataset == 'cifar10',
                          weight_decay=config.weight_decay)
    decoder = tf.keras.Sequential([tf.keras.layers.GlobalAveragePooling2D(name='GAP'),
                                   tf.keras.layers.Dense(units=config.zdim * 2, activation='relu'),
                                   tf.keras.layers.Dense(units=config.zdim)
                                   ])

    if config.task == 'pretrain':
        print("Pretraining model")
        pretrain_func(encoder=encoder, decoder=decoder, config=config)
    elif config.task == 'segmentation' or 'classification':
        print("Fine-tuning model")
        layer1 = tf.keras.layers.GlobalAveragePooling2D()
        model_output = tf.keras.layers.Dense(units=config.num_classes, use_bias=False, name='output')
        readout_model = tf.keras.models.Sequential([layer1, model_output])
        finetune_func(encoder=encoder, decoder=decoder, readout_model= readout_model, config=config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="main for self-supervised learning")
    parser.add_argument('--task', type=str, default='classification',
                        help="Specify the task; either pretrain, classification or segmentation")
    parser.add_argument('--model', type=str, default='18', help="Specify the ResNet model to be used; either 18 or 34")
    parser.add_argument('--imagenet_path', type=str, default='', help="Specify the path pretrained Imagenet weights")

    parser.add_argument('--batch_size', type=int, default=16, help="Specify the batch size")
    parser.add_argument('--weight_decay', type=float, default=1e-6, help="Specify the value for weight decay")
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help="Specify the value for the initial learning rate")
    parser.add_argument('--train_file_path', type=str, default='/data/SSL/log/',
                        help="Specify the path to the training files containing a list of image filenames to be "
                             "loaded, where each line is an image path")
    parser.add_argument('--val_file_path', type=str, default='',
                        help="Specify the path to the validation files containing a list of image filenames to be "
                             "loaded, where each line is an image path")
    parser.add_argument('--pretrain_save_path', type=str, default='logs/pretrain',
                        help="Specify the path to the directory containing the pretrain save checkpoint")
    parser.add_argument('--finetune_save_path', type=str, default='logs/finetune',
                        help="Specify the path to the directory containing the finetune save checkpoint")
    parser.add_argument('--dataset', type=str, default='cifar10', help="which dataset to use")
    parser.add_argument('--num_classes', type=int, default=10,
                        help="Specify the number of classes for classification or segmentation")
    parser.add_argument('--num_epochs', type=int, default=100,
                        help="Specify the number of epochs to train the model for")
    parser.add_argument('--freeze', type=bool, default=True, help="Freezes base model weights")
    parser.add_argument('--blur', default=True, help="Random gaussian blur image augmentation")
    parser.add_argument('--rotate', default=True, help="Applies random rotation image augmentation")
    parser.add_argument('--jitter', default=True, help="Applies jitter image augmentation")
    parser.add_argument('--crop', default=True, help="Applies random crop image augmentation")
    parser.add_argument('--flip', default=True, help="Applies random flip image augmentation")
    parser.add_argument('--noise', default=True, help="Applies Gaussian noise image augmentation")
    parser.add_argument('--colordrop', default=True, help="Applies color drop image augmentation")
    parser.add_argument('--temperature', type=float, default=1,
                        help="Sets the temperature for cross-entropy loss function")
    parser.add_argument('--strength', default=1.0, type=float, help="Sets the strength of augmentations")
    parser.add_argument('--save_freq', default=100, type=int, help="Sets the strength of augmentations")
    parser.add_argument('--trial_module', default='config_file.py', type=str, help="Dimension of the z space")
    parser.add_argument('--logdir_model', default='/data/SSL/log/20210805-134031/cp.ckpt', type=str,
                        help="which model to load")

    args = parser.parse_args()
    code = importlib.import_module(args.trial_module.replace(".py", ""))
    # merge args with config file args
    config_args = code.__getattribute__("config")
    dict_arg = vars(args)
    for key in dict_arg:
        val = dict_arg[key]
        config_args[key] = val

    run(AttrDict(config_args))
