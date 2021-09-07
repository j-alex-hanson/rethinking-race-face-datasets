import os
from PIL import Image
import random
from torch.utils import data
from torchvision import transforms as T


class BalancedfacePair(data.Dataset):

    def __init__(self, root, race_a='Caucasian', race_b='African', percent_a=50, input_shape=(3, 128, 128)):
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

        random.seed(42)
        root_a = os.path.join(root, race_a)
        root_b = os.path.join(root, race_b)

        label_to_images_a, label_buckets_a, labels_a = self.fetch_race_data(root_a)
        label_to_images_b, label_buckets_b, labels_b = self.fetch_race_data(root_b)

        images_per_label = min(label_buckets_a[2000], label_buckets_b[2000])
        num_labels_a = int(5000 * percent_a / 100)
        num_labels_b = 5000 - num_labels_a

        cur_label = 0
        for label in random.sample(list(labels_a[2000:]), num_labels_a):
            self.img_paths.extend(random.sample(label_to_images_a[label], images_per_label))
            self.labels.extend([cur_label] * images_per_label)
            cur_label += 1
        for label in random.sample(list(labels_b[2000:]), num_labels_b):
            self.img_paths.extend(random.sample(label_to_images_b[label], images_per_label))
            self.labels.extend([cur_label] * images_per_label)
            cur_label += 1


    def __getitem__(self, index):
        img_path = self.img_paths[index]
        label = self.labels[index]
        data = Image.open(img_path)
        data = data.convert('RGB')
        data = self.transforms(data)
        return data.float(), label


    def __len__(self):
        return len(self.img_paths)


    def fetch_race_data(self, race_root):
        label_to_images = {}
        label_buckets = []
        cur_label = 0
        for dir in os.listdir(os.path.join(race_root)):
            label_to_images[cur_label] = []
            cur_num_images = 0
            for img in os.listdir(os.path.join(race_root, dir)):
                label_to_images[cur_label].append(os.path.join(race_root, dir, img))
                cur_num_images += 1
            label_buckets.append(cur_num_images)
            cur_label += 1
        label_buckets, labels = zip(*sorted(zip(label_buckets, range(7000))))
        return label_to_images, label_buckets, labels
