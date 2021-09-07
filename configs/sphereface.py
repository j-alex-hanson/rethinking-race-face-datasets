from .config import Config

class SpherefaceConfig(Config):

    def __init__(self):
        self.config = {
            'method': 'sphereface',
            'metric': 'sphere',
            'loss': 'focal_loss',

            'lr': 1e-1,
            'lr_step': 10,
            'lr_decay': 0.95,
            'weight_decay': 5e-4,
        }
