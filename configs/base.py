from .config import Config

class BaseConfig(Config):

    def __init__(self):
        self.config = {
            # MODEL SETTINGS #
            'backbone': 'resnet50',
            'num_classes': 7000,
            'use_se': False,
            'save_center': False,

            # OPTIMIZER SETTINGS #
            'optimizer': 'sgd',

            # TRAIN LOOP SETTINGS #
            'max_epoch': 50,
            'save_interval': 10,
            'print_freq': 100,

            # DATA_LOADER SETTINGS #
            'train_batch_size': 128,
            'test_batch_size': 512,
            'input_shape': (3, 128, 128),
            'num_workers': 8,

            # DEFAULT EXPERIMENT #
            'trial_num': 0,
            'train': True,
            'test': True,
            'eval_each_epoch': False,
            'experiment_name': 'single_race',
            'train_race': 'Caucasian',
            'noise_mix':False,
        }
