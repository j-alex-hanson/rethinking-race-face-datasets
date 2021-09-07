from .config import Config

class VggFace2Config(Config):

    def __init__(self):
        self.config = {
            'method': 'vggface2',
            'backbone': 'senet50',
            'loss': 'cross_entropy_loss',

            'lr': 0.3,
            'lr_step': 10,
            'lr_decay': 0.95,
            'weight_decay': 5e-4,
        }
