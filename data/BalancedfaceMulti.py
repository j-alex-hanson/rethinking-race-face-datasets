import os
from PIL import Image
import random
from torch.utils import data
from torchvision import transforms as T


class BalancedfaceMulti(data.Dataset):

    def __init__(self, root, identities_per_race, images_per_identity=None, input_shape=(3, 128, 128)):
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
        PERSON_START_INDEX = 2000
        african_dir = os.path.join(root, 'African')
        asian_dir = os.path.join(root, 'Asian')
        caucasian_dir = os.path.join(root, 'Caucasian')
        indian_dir = os.path.join(root, 'Indian')

        african_label_buckets, african_labels_to_images, african_labels = self.get_sorted_labels(african_dir)
        asian_label_buckets, asian_labels_to_images, asian_labels = self.get_sorted_labels(asian_dir)
        caucasian_label_buckets, caucasian_labels_to_images, caucasian_labels = self.get_sorted_labels(caucasian_dir)
        indian_label_buckets, indian_labels_to_images, indian_labels = self.get_sorted_labels(indian_dir)

        if images_per_identity is None:
            images_per_label = min(african_label_buckets[PERSON_START_INDEX], asian_label_buckets[PERSON_START_INDEX], caucasian_label_buckets[PERSON_START_INDEX], indian_label_buckets[PERSON_START_INDEX])
            images_per_identity = {}
            images_per_identity['african'] = images_per_label
            images_per_identity['asian'] = images_per_label
            images_per_identity['caucasian'] = images_per_label
            images_per_identity['indian'] = images_per_label

        cur_label = 0
        for label in list(african_labels)[PERSON_START_INDEX:PERSON_START_INDEX+identities_per_race['african']]:
            self.img_paths.extend(random.sample(african_labels_to_images[label], images_per_identity['african']))
            self.labels.extend([cur_label] * images_per_identity['african'])
            cur_label += 1
        for label in list(asian_labels)[PERSON_START_INDEX:PERSON_START_INDEX+identities_per_race['asian']]:
            self.img_paths.extend(random.sample(asian_labels_to_images[label], images_per_identity['asian']))
            self.labels.extend([cur_label] * images_per_identity['asian'])
            cur_label += 1
        for label in list(caucasian_labels)[PERSON_START_INDEX:PERSON_START_INDEX+identities_per_race['caucasian']]:
            self.img_paths.extend(random.sample(caucasian_labels_to_images[label], images_per_identity['caucasian']))
            self.labels.extend([cur_label] * images_per_identity['caucasian'])
            cur_label += 1
        for label in list(indian_labels)[PERSON_START_INDEX:PERSON_START_INDEX+identities_per_race['indian']]:
            self.img_paths.extend(random.sample(indian_labels_to_images[label], images_per_identity['indian']))
            self.labels.extend([cur_label] * images_per_identity['indian'])
            cur_label += 1


    def get_sorted_labels(self, root_dir):
        label_to_images = {}
        label_buckets = []
        cur_label = 0
        for dir in os.listdir(os.path.join(root_dir)):
            label_to_images[cur_label] = []
            cur_num_images = 0
            for img in os.listdir(os.path.join(root_dir, dir)):
                label_to_images[cur_label].append(os.path.join(root_dir, dir, img))
                cur_num_images += 1
            label_buckets.append(cur_num_images)
            cur_label += 1
        label_buckets, labels = zip(*sorted(zip(label_buckets, range(7000))))
        return label_buckets, label_to_images, labels


    def __getitem__(self, index):
        img_path = self.img_paths[index]
        label = self.labels[index]
        data = Image.open(img_path)
        data = data.convert('RGB')
        data = self.transforms(data)
        return data.float(), label


    def __len__(self):
        return len(self.img_paths)
