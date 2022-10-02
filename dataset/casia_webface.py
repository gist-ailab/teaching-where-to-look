import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np
import cv2
import os
import torch
import torch.nn.functional as F
from utility.noise import get_gaussian_kernel


def img_loader(path):
    try:
        with open(path, 'rb') as f:
            img = cv2.imread(path)
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            return img
    except IOError:
        print('Cannot load image ' + path)


class CASIAWebFace(data.Dataset):
    def __init__(self, root, file_list, down_size=[112], single=True, transform=None, loader=img_loader, flip=True):
        self.root = root
        self.transform = transform
        self.loader = loader
        self.down_size = down_size
        self.single = single
        
        image_list = []
        label_list = []
        with open(file_list) as f:
            img_label_list = f.read().splitlines()

        for info in img_label_list:
            image_path, label_name = info.split('  ')
            image_list.append(image_path)
            label_list.append(int(label_name))

        self.image_list = image_list
        self.label_list = label_list
        self.class_nums = len(np.unique(self.label_list))

        self.flip = flip
        print("dataset size: ", len(self.image_list), '/', self.class_nums)

    def __getitem__(self, index):
        img_path = self.image_list[index]
        ind = img_path.find('faces_webface_112x112')
        img_path = img_path[ind+28:]
        
        label = self.label_list[index]

        HR_img = self.loader(os.path.join(self.root, img_path))

        # random flip with ratio of 0.5
        if self.flip:
            flip = np.random.choice(2) * 2 - 1
            if flip == 1:
                HR_img = cv2.flip(HR_img, 1)

        if self.transform is not None:
            HR_img = self.transform(HR_img)
        else:
            HR_img = torch.from_numpy(HR_img)
            
        # DownSampling
        if len(self.down_size) == 1 and self.down_size[0] == 112:
            return HR_img, label
        else:
            img_size = HR_img.size(-1)

            if len(self.down_size) == 1:
                down_size_select = self.down_size[0]
            else:
                down_size_select = np.random.choice(self.down_size, 1, replace=False)[0]
            
            LR_img = HR_img.unsqueeze(0)
            LR_img = F.interpolate(LR_img, size=int(down_size_select))
            LR_img = F.interpolate(LR_img, size=img_size)
            LR_img = get_gaussian_kernel()(LR_img)
            LR_img = LR_img.squeeze(0)

            if self.single:
                return LR_img, label
            else:
                return HR_img, LR_img, label

    def __len__(self):
        return len(self.image_list)


if __name__ == '__main__':
    root = 'D:/data/webface_align_112'
    file_list = 'D:/data/webface_align_train.list'

    transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])
    dataset = CASIAWebFace(root, file_list, transform=transform)
    trainloader = data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2, drop_last=False)
    print(len(dataset))
    for data in trainloader:
        print(data[0].shape)