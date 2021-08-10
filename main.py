"""Run SSL model (pretrain/ supervised train)"""
import argparse
import importlib
from self_supervised.train import pretrain_func


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def main(config):
    """Runs the model for either pretrain or segmentation/classification tasks
    
    Loads the ResNet backbone and based on task specification performs either
    pretrain or segmentation/classification fine-tuning.

    Args:
        config: instance of config class
    """

    print("Pretraining model")
    pretrain_func(config=config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="main for self-supervised learning")
    parser.add_argument('--learning_rate_scaling', type=str, default='linear',
                        help='linear or sqrt, How to scale the learning rate as a function of batch size.')
    parser.add_argument('--warmup_epochs', type=int, default=10,
                        help='Number of epochs of warmup.')
    parser.add_argument('--train_mode', type=str, default='pretrain',
                        help="Specify the task; either pretrain, finetune")
    parser.add_argument('--model', type=str, default='18', help="Specify the ResNet model to be used; either 18 or 34")
    parser.add_argument('--imagenet_path', type=str, default='', help="Specify the path pretrained Imagenet weights")

    parser.add_argument('--train_batch_size', type=int, default=512, help="Specify the train  batch size")
    parser.add_argument('--eval_batch_size', type=int, default=512, help="Specify the eval batch size")
    parser.add_argument('--train_epochs', type=int, default=1000, help="Number of epochs to train for.")
    parser.add_argument('--contrastive_optimizer', type=str, default='momentum',
                        help="['momentum', 'adam', 'lars] Optimizer to use")
    parser.add_argument('--probe_optimizer', type=str, default='adam',
                        help="['momentum', 'adam', 'lars] Optimizer to use for the supervised probe")
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum parameter.')

    parser.add_argument('--train_steps', type=int, default=0,
                        help="'Number of steps to train for. If provided, overrides train_epochs.'")
    parser.add_argument('--contrastive_weight_decay', type=float, default=1e-6, help=" weight decay for the optimizer")
    parser.add_argument('--weights_decay', type=float, default=1e-6, help="weight decay for the encoder")
    parser.add_argument('--probe_weight_decay', type=float, default=1e-6, help="weight decay for the probe classifier")
    parser.add_argument('--learning_rate', type=float, default=1,  help="the initial learning rate")
    parser.add_argument('--probe_learning_rate', type=float, default=1e-3,  help="initial learning rate for the probe")
    parser.add_argument('--probe_momentum', type=float, default=0.9,  help="momentum for the probe")
    parser.add_argument('--log_dir', type=str, default='/data/SSL/log/',
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
    parser.add_argument('--freeze', type=bool, default=True, help="Freezes base model weights")

    parser.add_argument('--temperature', type=float, default=0.5,
                        help="Sets the temperature for cross-entropy loss function")
    parser.add_argument('--save_freq', default=1000, type=int, help="Sets the strength of augmentations")
    parser.add_argument('--trial_module', default='config_file.py', type=str, help="Dimension of the z space")
    parser.add_argument('--logdir_model', default='/data/SSL/log/20210805-134031/cp.ckpt', type=str,
                        help="which model to load")
    parser.add_argument('--use_swag', default=True, type=bool, help="Run SWAG model on the top")
    parser.add_argument('--swag_start_epoch', default=3, type=int, help="the start epoch If we run swag")
    parser.add_argument('--swag_lr2', default=0.02, type=float, help="max lr for cyclic swag scheduler")
    parser.add_argument('--swag_lr', default=0.1, type=float, help="SWAG lr")
    parser.add_argument('--swag_freq', default=1, type=int, help="The interval (in epochs) for update")
    parser.add_argument('--eval_steps', default=0, type=int, help="The number of validation steps if 0 go over all the dataset")
    parser.add_argument('--steps_per_epoch', default=100, type=int, help="The number of train steps per epoch")
    parser.add_argument('--swag_lr_schedule', default='constant', type=str, help="constant or cyclic")
    parser.add_argument('--image_size', default=32, type=int, help="image input size")
    parser.add_argument('--cache_dataset', default=False, type=bool, help="cache the dataset")
    parser.add_argument('--lineareval_while_pretraining', default=True, type=bool,
                        help="Whether to finetune supervised head while pretraining.")
    parser.add_argument('--batch_norm_decay', default=0.9, type=float, help="Batch norm decay parameter.")
    parser.add_argument('--color_jitter_strength', default=0.5, type=float, help="The strength of color jittering.")
    parser.add_argument('--proj_out_dim', default=128, type=int, help='Number of head projection dimension.')
    parser.add_argument('--num_proj_layers', default=3, type=int, help='Number of non-linear head layers.')
    parser.add_argument('--ft_proj_selector', default=0, type=int,
                        help='Which layer of the projection head to use during fine-tuning. '
                             '0 means no projection head, and -1 means the final layer.')
    parser.add_argument('--proj_head_mode', default='nonlinear', type=str, help="['none', 'linear', 'nonlinear'],"
                                                                                "How the head projection is done.")
    parser.add_argument('--resnet_depth', default=18, type=int, help='Depth of ResNet.')
    parser.add_argument('--width_multiplier', default=1, type=int, help='Multiplier to change width of network.')
    parser.add_argument('--sk_ratio', default=0., type=float,
                        help='If it is bigger than 0, it will enable SK. Recommendation: 0.0625.')
    parser.add_argument('--se_ratio', default=0., type=float, help='If it is bigger than 0, it will enable SE.')
    parser.add_argument('--use_blur', default=False, type=bool,
                        help='Whether or not to use Gaussian blur for augmentation during pretraining.')
    parser.add_argument('--fine_tune_after_block', default=-1, type=int,
                        help='The layers after which block that we will fine-tune. -1 means fine-tuning '
                             'everything. 0 means fine-tuning after stem block. 4 means fine-tuning '
                             'just the linear head.')

    args = parser.parse_args()
    code = importlib.import_module(args.trial_module.replace(".py", ""))
    config_args = vars(args)
    # Group args for the encoder and decoder
    projection_head_args_keys = ['proj_head_mode', 'num_proj_layers', 'ft_proj_selector', 'proj_out_dim']
    resent_head_args_keys = ['resnet_depth', 'width_multiplier', 'train_mode', 'fine_tune_after_block', 'sk_ratio',
                             'se_ratio', 'weights_decay']

    projection_head_dict = {key: config_args[key] for key in projection_head_args_keys}
    resent_head_head_dict = {key: config_args[key] for key in resent_head_args_keys}
    config_args['projection_head_args'] = projection_head_dict
    config_args['resent_head_args'] = resent_head_head_dict
    main(AttrDict(config_args))
