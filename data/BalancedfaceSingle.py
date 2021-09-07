import cv2
import numpy as np
import os
from PIL import Image
import random
from torch.utils import data
from torchvision import transforms as T


class BalancedfaceSingle(data.Dataset):

    def __init__(self, root, race='Caucasian', input_shape=(3, 128, 128), noise_mixing=False, noise=0.25):
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
        self.noise_mixing = noise_mixing
        self.noise = noise

        root = os.path.join(root, race)

        cur_label = 0
        for dir in os.listdir(os.path.join(root)):
            for img in os.listdir(os.path.join(root, dir)):
                self.img_paths.append(os.path.join(root, dir, img))
                self.labels.append(cur_label)
            cur_label += 1

        if self.noise_mixing:
            self.face_cascade = cv2.CascadeClassifier('./data/face_detector.xml')


    def __getitem__(self, index):
        img_path = self.img_paths[index]
        label = self.labels[index]
        data = Image.open(img_path)
        data = data.convert('RGB')

        if self.noise_mixing and random.random() < self.noise:
            img = np.array(data)
            faces = self.face_cascade.detectMultiScale(img, 1.1, 4)
            rint = random.randint(5,10)
            ksize = 2*rint+1
            blurrer = T.GaussianBlur(kernel_size=ksize, sigma=1.5)
            blur_img = np.array(blurrer(data))
            for (x, y, w, h) in faces:
                face = img[x:x+w,y:y+h,:]
                rx = random.randint(0,w-(w//4))
                ry = random.randint(0,h-(h//4))
                face[rx:rx+(w//4), ry:ry+(h//4)] = blur_img[x+rx:x+rx+(w//4), y+ry:y+ry+(h//4)]
                img[x:x+w,y:y+h] = face
            data = Image.fromarray(np.uint8(img))
        
        data = self.transforms(data)
        return data.float(), label


    def __len__(self):
        return len(self.img_paths)

