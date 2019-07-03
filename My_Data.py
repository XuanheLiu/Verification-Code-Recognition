#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File : My_Model.py
# Author: LiuXuanhe
# Date : 2019/7/2
from torch.utils.data import Dataset
import os
from PIL import Image
import time
import numpy as np

class My_Data(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.img = []
        self.tar = []
        [(self.img.append(os.path.join(self.root, name)),
          self.tar.append(name.split(".")[0]))
         for name in os.listdir(root)]

    def __len__(self):
        return len(self.img)

    def __getitem__(self, index):
        image, target = self.img[index], self.tar[index]
        image = Image.open(image)
        assert image.size[0] == 120 and image.size[1] == 60
        target = self._label_to_onehot(target)
        if self.transform is not None:
           image = self.transform(image)

        return image, target

    def _label_to_onehot(self, target):
        trans = {"0":0, "1":1, "2":2, "3":3, "4":4, "5":5, "6":6, "7":7, "8":8, "9":9,
         "a":10, "b":11, "c":12, "d":13, "e":14, "f":15, "g":16, "h":17, "i":18, "j":19,
         "k":20, "l":21, "m":22, "n":23, "o":24, "p":25, "q":26, "r":27, "s":28, "t":29,
         "u":30, "v":31, "w":32, "x":33, "y":34, "z":35, "A":36, "B":37, "C":38, "D":39,
         "E":40, "F":41, "G":42, "H":43, "I":44, "J":45, "K":46, "L":47, "M":48, "N":49,
         "O":50, "P":51, "Q":52, "R":53, "S":54, "T":55, "U":56, "V":57, "W":58, "X":59,
         "Y":60, "Z":61}
        label = np.zeros(shape=(4, 62), dtype=np.float32)
        for i, j in enumerate(target):
            label[i][trans[j]] = 1
        return label
if __name__ == '__main__':
    from torchvision import transforms
    from torch.utils.data import DataLoader
    transf = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                       std=(0.5, 0.5, 0.5))])
    my_data = My_Data(r"E:\data\Verification_Code\V3\test", transform=transf)
    dataloader = DataLoader(my_data, batch_size=64, shuffle=True)
    for da, ta in dataloader:
        print(da.size(), type(ta), ta.shape, ta.dtype)
        print(ta)
        time.sleep(1)