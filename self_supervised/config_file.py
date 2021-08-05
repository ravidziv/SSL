config = {'task': 'pretrain',
          'cifar10': False,
          'model': '18',
          'num_classes': 10,
          'train_file_path': '/data/SSL/logs',
          'learning_rate': 0.1,
          'momentum':0.9,
          'blur': True,
          'noise': True,
          'flip': True,
          'crop': True,
          'rotate': False,
          'jitter': True,
          'colordrop': True,
          'cutout': False,
          'batch_size': 16,
          'num_epochs': 1000,
          'rotate_prob': 0.5,
          'max_rotation': 30,
          'rotation_factor': 0.2,
          'flip_prob': 0.5,
          'blur_prob': 0.5,
          'color_prob': 0.2,
          'jitter_prob': 0.8,
          'cutout_prob': 0.5,
          'strength': 1.0,
          'pretrain_save_path': 'logs/pretrain',
          'input_size': [32, 32],
          'crop_size': [32, 32],
          'pretrained': False,
          'freeze': False,
          'include_nonlinearity': False,
          'temperature': 0.5,
          'weight_decay': 1e-6,
          'zdim': 128
          }
''''  
elif 'task == 'segmentation' or 'task == 'classification':
    'cifar10' :False
    'model' :'18'
    'num_classes' :5
    'imagenet_path' :''
    'train_file_path' :''
    'val_file_path' :''
    'learning_rate' :0.001
    'momentum': 0.9
    'blur' :False
    'noise' :False
    'flip' :False
    'crop' :False
    'rotate' :False
    'jitter' :False
    'colordrop' :False
    'cutout' :False
    'batch_size' :64
    'num_epochs' :100
    'rotate_prob' :0.5
    'max_rotation' :30
    'flip_prob' :0.5
    'blur_prob' :0.5
    'color_prob' :0.2
    'jitter_prob' :0.8
    'cutout_prob' :0.5
    'strength' :1.0
    'pretrain_save_path' :'logs/pretrain'
    'finetune_save_path' :'logs/finetune'
    'input_size' :[500, 500],
    'crop_size' :[240, 240],
    'pretrained' :False,
    'freeze' :False,
    'include_nonlinearity' :False,
    'weight_decay' :0,
    'zdim' :128,
'''
