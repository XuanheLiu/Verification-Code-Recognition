#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File : My_Model2.py
# Author: LiuXuanhe
# Date : 2019/7/3
from torch import nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1),
                                   nn.BatchNorm2d(num_features=32),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 3, 1, 1),
                                   nn.BatchNorm2d(num_features=64),
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, 3, 2, 1),
                                   nn.BatchNorm2d(num_features=128),
                                   nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1),
                                   nn.BatchNorm2d(num_features=64),
                                   nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1),
                                   nn.BatchNorm2d(num_features=32),
                                   nn.ReLU(inplace=True))
        self.fc = nn.Sequential(nn.Linear(32 * 15 * 30, 256),
                                nn.BatchNorm1d(num_features=256))

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x5 = x5.view(-1, 32*15*30)
        y = self.fc(x5)
        return y


class SEQ(nn.Module):
    def __init__(self):
        super(SEQ, self).__init__()
        self.lstm = nn.LSTM(input_size=256, hidden_size=128, num_layers=2, batch_first=True)
        self.fc = nn.Linear(128, 62)

    def forward(self, x):
        x = x.view(-1, 1, 256)
        x = x.expand(-1, 4, 256)
        lstm, _ = self.lstm(x)
        x = lstm.contiguous().view(-1, 128)
        y = self.fc(x)
        out = y.view(-1, 4, 62)
        output = F.softmax(out, dim=2)
        return output


class CNN2SEQ(nn.Module):
    def __init__(self):
        super(CNN2SEQ, self).__init__()
        self.encoder = CNN()
        self.decoder = SEQ()

    def forward(self, x):
        encoder = self.encoder(x)
        decoder = self.decoder(encoder)
        return decoder