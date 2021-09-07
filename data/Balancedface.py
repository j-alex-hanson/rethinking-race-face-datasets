import os
from PIL import Image
from torch.utils import data
from torchvision import transforms as T


class Balancedface(data.Dataset):

    def __init__(self, root, input_shape=(3, 128, 128)):
        self.input_shape = input_shape
        self.transforms = T.Compose([
            T.Resize(int(self.input_shape[1] * 156 / 128)),
            T.RandomCrop(self.input_shape[1:]),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5],
                            std=[0.5, 0.5, 0.5])
        ])
        self.img_paths = []
        self.labels = []

        cur_label = 0
        for race in ['African', 'Asian', 'Caucasian', 'Indian']:
            img_paths = []
            labels = []
            race_root = os.path.join(root, race)
            for dir in os.listdir(os.path.join(race_root)):
                for img in os.listdir(os.path.join(race_root, dir)):
                    img_paths.append(os.path.join(race_root, dir, img))
                    labels.append(cur_label)
                cur_label += 1
            self.img_paths.extend(img_paths)
            self.labels.extend(labels)


    def __getitem__(self, index):
        imgPath = self.img_paths[index]
        label = self.labels[index]
        data = Image.open(imgPath)
        data = data.convert('RGB')
        data = self.transforms(data)
        return data.float(), label


    def __len__(self):
        return len(self.img_paths)
