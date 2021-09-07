from .config import Config

class CenterlossConfig(Config):

    def __init__(self):
        self.config = {
            'method': 'centerloss',
            'save_center': True,
            'loss': 'cross_entropy_loss',
            'lda': 0.1,
            'alpha': 0.1,

            'lr': 3e-1,
            'lr_step': 10,
            'lr_decay': 0.95,
            'weight_decay': 5e-4
        }
