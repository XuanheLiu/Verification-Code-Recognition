#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File : My_Model.py
# Author: LiuXuanhe
# Date : 2019/7/2
from PIL import Image, ImageDraw, ImageFont
import random
import os

def rndCHR():
    return chr(random.randint(67, 90))

def rndchr():
    return chr(random.randint(97, 122))

def rndnum():
    return str(random.randint(0, 9))

def rndcode():
    choice = random.choice([rndCHR(), rndchr(), rndnum()])
    return choice

def rndcolor():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(50, 255))

def rndcond(x, y, a1=1, a2=4, b1=5, b2=15):
    return (x+random.randint(a1, a2), y+random.randint(b1, b2))

def creatcode(codelen, path):
    base = Image.new("RGBA", (120, 40),(255, 255, 255))
    img = Image.new("RGB", (120, 60), (255, 255, 255))
    font = ImageFont.truetype("simsun.ttc", 36)
    draw = ImageDraw.Draw(base)
    imgname = ""
    for i in range(codelen):
        rd = rndcode()
        draw.text((i*30+5, 5), rd, font=font, fill=rndcolor())
        med = base.crop((i*30, 0, (i+1)*30, 40))
        med = med.rotate(random.randint(-40, 40))
        r, g, b, a = med.split()
        img.paste(med, rndcond(i*30, 0), mask=a)
        imgname += rd
    drawline = ImageDraw.Draw(img)
    for i in range(random.randint(2, 3)):
        x1, y1 = random.randint(5, 15), random.randint(5, 55)
        x2, y2 = random.randint(105, 115), random.randint(5, 55)
        drawline.line((x1, y1, x2, y2), fill=rndcolor())
    imgname += ".jpg"
    if not os.path.exists(path):
        os.makedirs(path)
    pathfn = os.path.join(path, imgname)
    img.save(pathfn, "JPEG")

if __name__ == '__main__':
    path = r"E:\data\Verification_Code\V3"
    dirname = {"train":128000, "val":16000, "test":16000}
    for k, v in dirname.items():
        tarpath = os.path.join(path, k)
        while True:
            creatcode(4, tarpath)
            if len(os.listdir(tarpath)) == v:
                print("{}数据集已完成".format(k))
                break