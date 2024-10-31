#coding=utf-8
import struct
import os
from PIL import Image

#DATA_PATH = r"C:/Users/alexx/$ML_PATH/chinese/Gnt1.0Test/"
DATA_PATH = r"C:/Users/alexx/$ML_PATH/chinese/Gnt1.0TrainPart3" #gnt数据文件路径
#IMG_PATH = r"C:/Users/alexx/$ML_PATH/chinese/data/test"
IMG_PATH = r"C:/Users/alexx/$ML_PATH/chinese/data/train" #解析后的图片存放路径

files = os.listdir(DATA_PATH)

num = 1
total = len(files)
counts = {}
for file in files:
    print(str(num) + "/" + str(total))
    tag = []
    img_bytes = []
    img_wid = []
    img_hei = []
    f=open(DATA_PATH+"/"+file, "rb")
    while f.read(4):
        tag_code = ord(f.read(2).decode("gbk").rstrip("\x00"))
        tag.append(tag_code)
        if tag_code not in counts:
            if os.path.exists(os.path.join(IMG_PATH, str(tag_code))):
                _, _, files = next(os.walk(os.path.join(IMG_PATH, str(tag_code))))
                counts[tag_code] = len(files) + 1
            else:
                os.mkdir(os.path.join(IMG_PATH, str(tag_code)))
                counts[tag_code] = 1
        width = struct.unpack('<h', bytes(f.read(2)))
        height = struct.unpack('<h', bytes(f.read(2)))
        img_hei.append(height[0])
        img_wid.append(width[0])
        data=f.read(width[0]*height[0])
        img_bytes.append(data)
    f.close()
    for k in range(0, len(tag)):
        im = Image.frombytes('L', (img_wid[k], img_hei[k]), img_bytes[k])
        if not os.path.exists(os.path.join(IMG_PATH, str(tag[k]), str(num), ".png")):
            im.save(IMG_PATH + "/" + str(tag[k]) + "/" + str(counts[tag[k]]) + ".png")
            counts[tag[k]] += 1
    num = num + 1

files=os.listdir(IMG_PATH)
n=0
f=open("label.txt","w") #创建用于训练的标签文件
for file in files:
    files_d=os.listdir(IMG_PATH+"/"+file)
    for file1 in files_d:
        f.write(file+"/"+file1+" "+str(n)+"\n")
    n = n+1