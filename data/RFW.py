import os
from PIL import Image
from torch.utils import data
from torchvision import transforms as T


def getRFWList(pair_list_file):
    with open(pair_list_file, 'r') as fd:
        pair_lines = fd.readlines()
    pair_list = []
    for pair_line in pair_lines:
        pair_items = pair_line.split('\t')
        if len(pair_items) == 3:
            identity = pair_items[0].strip()
            first_image = identity + '_' + pair_items[1].strip().zfill(4)
            second_image = identity + '_' + pair_items[2].strip().zfill(4)
            pair_list.append((first_image, second_image, 1))
        elif len(pair_items) == 4:
            first_identity = pair_items[0].strip()
            first_image = first_identity + '_' + pair_items[1].strip().zfill(4)
            second_identity = pair_items[2].strip()
            second_image = second_identity + '_' + pair_items[3].strip().zfill(4)
            pair_list.append((first_image, second_image, 0))
        else:
            raise Exception('pair file not following expected format')

    return pair_list


class RFW(data.Dataset):

    def __init__(self, root, race='Caucasian', input_shape=(3, 128, 128)):
        self.input_shape = input_shape
        self.transforms = T.Compose([
            T.Resize(int(self.input_shape[1] * 156 / 128)),
            T.CenterCrop(self.input_shape[1:]),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5],
                            std=[0.5, 0.5, 0.5])
        ])
        self.img_paths = []

        root = os.path.join(root, 'data', race)
        for dir in os.listdir(os.path.join(root)):
            for img in os.listdir(os.path.join(root, dir)):
                self.img_paths.append(os.path.join(root, dir, img))


    def __getitem__(self, index):
        img_path = self.img_paths[index]
        data = Image.open(img_path)
        data = data.convert('RGB')
        data = self.transforms(data)
        return data.float(), img_path.split('/')[-1][:-4]


    def __len__(self):
        return len(self.img_paths)
