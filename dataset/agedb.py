import numpy as np
import cv2
import os
import torch.utils.data as data

import torch
import torchvision.transforms as transforms
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

class AgeDB30(data.Dataset):
    def __init__(self, root, file_list, down_size=112, transform=None, loader=img_loader):
        self.root = root
        self.file_list = file_list
        self.transform = transform
        self.loader = loader
        self.nameLs = []
        self.nameRs = []
        self.folds = []
        self.flags = []

        self.down_size = down_size
        assert type(self.down_size) == int
        
        with open(file_list) as f:
            pairs = f.read().splitlines()
        for i, p in enumerate(pairs):
            p = p.split(' ')
            nameL = p[0]
            nameR = p[1]
            fold = i // 600
            flag = int(p[2])

            self.nameLs.append(nameL)
            self.nameRs.append(nameR)
            self.folds.append(fold)
            self.flags.append(flag)

    def __getitem__(self, index):
        HR_img_l = self.loader(os.path.join(self.root, self.nameLs[index]))
        HR_img_r = self.loader(os.path.join(self.root, self.nameRs[index]))
        HR_imglist = [self.transform(HR_img_l), self.transform(cv2.flip(HR_img_l, 1)), self.transform(HR_img_r), self.transform(cv2.flip(HR_img_r, 1))]
        
        img_size = HR_imglist[0].size(-1)
        if self.down_size != 112:
            LR_imglist = []
            for ix in range(len(HR_imglist)):
                LR_imp = HR_imglist[ix].unsqueeze(0)

                LR_imp = F.interpolate(LR_imp, size=int(self.down_size))
                LR_imp = F.interpolate(LR_imp, size=img_size)
                LR_imp = get_gaussian_kernel()(LR_imp)

                LR_imp = LR_imp.squeeze(0)
                
                LR_imglist.append(LR_imp)
            return LR_imglist
        
        else:
            return HR_imglist


    def __len__(self):
        return len(self.nameLs)


if __name__ == '__main__':
    root = '/media/sda/AgeDB-30/agedb30_align_112'
    file_list = '/media/sda/AgeDB-30/agedb_30_pair.txt'

    transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])

    dataset = AgeDB30(root, file_list, transform=transform)
    trainloader = data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2, drop_last=False)
    for data in trainloader:
        for d in data:
            print(d[0].shape)