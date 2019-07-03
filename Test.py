#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File : Test.py
# Author: LiuXuanhe
# Date : 2019/7/3
from My_Data import My_Data
from My_Model2 import CNN2SEQ
from torch.utils.data import DataLoader
from torch import optim, nn
import torch
from torchvision import transforms
import os

def correct_num(out, label):
    count = 0
    for i in range(out.size(0)):
        out_label = "".join(list(trans.keys())[list(trans.values()).index(value)] for value in
                            list(torch.squeeze(torch.argmax(out, dim=2)[i]).cpu().data.numpy()))
        real_label = "".join(list(trans.keys())[list(trans.values()).index(value)] for value in
                             list(torch.squeeze(torch.argmax(label, dim=2)[i]).cpu().data.numpy()))
        if out_label == real_label:
            count += 1
    return count

trans = {"0":0, "1":1, "2":2, "3":3, "4":4, "5":5, "6":6, "7":7, "8":8, "9":9,
         "a":10, "b":11, "c":12, "d":13, "e":14, "f":15, "g":16, "h":17, "i":18, "j":19,
         "k":20, "l":21, "m":22, "n":23, "o":24, "p":25, "q":26, "r":27, "s":28, "t":29,
         "u":30, "v":31, "w":32, "x":33, "y":34, "z":35, "A":36, "B":37, "C":38, "D":39,
         "E":40, "F":41, "G":42, "H":43, "I":44, "J":45, "K":46, "L":47, "M":48, "N":49,
         "O":50, "P":51, "Q":52, "R":53, "S":54, "T":55, "U":56, "V":57, "W":58, "X":59,
         "Y":60, "Z":61}

batch_size = 128
epochs = 300
img_path = r"E:\data\Verification_Code\V3"
save_path = r"Ckpt2"
if not os.path.exists(save_path):
    os.makedirs(save_path)
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                         std=[0.5, 0.5, 0.5])])
    data = My_Data(root=os.path.join(img_path, "test"), transform=transform)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=3, drop_last=True)
    data_size = len(data)
    cnn2seq = CNN2SEQ().to(device)
    save_weight = os.path.join(save_path, r"ckpt.pth")
    cnn2seq.load_state_dict(torch.load(save_weight))
    cnn2seq.eval()
    chr_correct = 0
    code_correct = 0.0
    for img, label in dataloader:
        img = img.to(device)
        label = label.to(device)
        with torch.set_grad_enabled(False):
            out = cnn2seq(img)
            out_label = "".join(list(trans.keys())[list(trans.values()).index(value)] for value in list(torch.squeeze(torch.argmax(out, dim=2)[0]).cpu().data.numpy()))
            real_label = "".join(list(trans.keys())[list(trans.values()).index(value)] for value in list(torch.squeeze(torch.argmax(label, dim=2)[0]).cpu().data.numpy()))
            count = correct_num(out, label)
            code_correct += count
            chr_correct += torch.sum(torch.argmax(label, dim=2) == torch.argmax(out, dim=2))
    chr_acc = chr_correct.double() / data_size / 4
    code_acc = code_correct / data_size
    print("out_label:{}, real_label:{}".format(out_label, real_label))
    print("chr_acc:{:.4f}, code_acc:{:.4f}".format(chr_acc, code_acc))
    with open("test_log.txt", "a+") as f:
        f.write("out_label:{}, real_label:{}\n".format(out_label, real_label))
        f.write("chr_acc:{:.4f}, code_acc:{:.4f}\n".format(chr_acc, code_acc))
