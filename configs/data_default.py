
from .config import Config

class DefaultDataConfig(Config):

    def __init__(self):
        self.config = {
            'checkpoints_path': '~/checkpoints',
            'features_dir': '~/RFW-features/',
            'train_root': '~/BUPT-Balancedface/race_per_7000/',
            'test_root': '~/RFW/images/test/',
            'identities_per_race_path': '~/RFW-images-per-race/',
            'rfw_test_list': '_pairs.txt',
            'simplex_file_path': 'configs/points_config.json'
        }
