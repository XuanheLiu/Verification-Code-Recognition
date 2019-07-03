#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File : Train.py
# Author: LiuXuanhe
# Date : 2019/7/2
from My_Data import My_Data
from My_Model2 import CNN2SEQ
from torch.utils.data import DataLoader
from torch import optim, nn
import torch
from torchvision import transforms
import os


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
    input("请先设置开始的轮次和best_acc，请查阅log2.txt")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                         std=[0.5, 0.5, 0.5])])
    data = {"train":My_Data(root=os.path.join(img_path, "train"), transform=transform),
            "val":My_Data(root=os.path.join(img_path, "val"), transform=transform)}
    dataloader = {"train":DataLoader(data["train"], batch_size=batch_size, shuffle=True, num_workers=3, drop_last=True),
                  "val":DataLoader(data["val"], batch_size=batch_size, shuffle=True, num_workers=3, drop_last=True)}
    data_size = {"train":len(data["train"]), "val":len(data["val"])}
    cnn2seq = CNN2SEQ().to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam([{"params":cnn2seq.encoder.parameters()},
                            {"params":cnn2seq.decoder.parameters()}])

    save_weight = os.path.join(save_path, r"ckpt.pth")
    if os.path.exists(save_weight):
        cnn2seq.load_state_dict(torch.load(save_weight))
    best_acc = 0.9882
    for i in range(9, epochs):
        for flag in ["train", "val"]:
            if flag == "train":
                cnn2seq.train()
            else:
                cnn2seq.eval()
            runing_loss = 0.0
            runing_correct = 0
            for img, label in dataloader[flag]:
                img = img.to(device)
                label = label.to(device)
                with torch.set_grad_enabled(flag == "train"):
                    out = cnn2seq(img)
                    out_label = "".join(list(trans.keys())[list(trans.values()).index(value)] for value in list(torch.squeeze(torch.argmax(out, dim=2)[0]).cpu().data.numpy()))
                    real_label = "".join(list(trans.keys())[list(trans.values()).index(value)] for value in list(torch.squeeze(torch.argmax(label, dim=2)[0]).cpu().data.numpy()))
                    loss = loss_fn(out, label)
                    if flag == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                runing_loss += loss.item()*img.size(0)
                runing_correct += torch.sum(torch.argmax(label, dim=2) == torch.argmax(out, dim=2))
            epoch_loss = runing_loss / data_size[flag]
            epoch_acc = runing_correct.double() / data_size[flag] / 4
            print("-"*5,"第{}轮{}集合".format(str(i+1), flag),"-"*5)
            print("out_label:{}, real_label:{}".format(out_label, real_label))
            print("epoch_loss:{:.4f}, epoch_acc:{:.4f}".format(epoch_loss, epoch_acc))
            with open("log2.txt", "a+") as f:
                f.write("-"*5+"第{}轮{}集合".format(str(i+1), flag)+"-"*5+"\n")
                f.write("out_label:{}, real_label:{}\n".format(out_label, real_label))
                f.write("epoch_loss:{:.4f}, epoch_acc:{:.4f}\n".format(epoch_loss, epoch_acc))
            if flag=="val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(cnn2seq.state_dict(), save_weight)
                print("第{}轮权重已保存".format(str(i+1)))
                with open("log2.txt", "a+") as f:
                    f.write("第{}轮权重已保存\n".format(str(i+1)))
        print("*"*15)
        with open("log2.txt", "a+") as f:
            f.write("*"*15+"\n")
