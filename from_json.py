#!/usr/bin/env python
# coding: utf-8

import os
import cv2
import json
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw
import torch
from pycocotools import coco

def init(work_dir = '', max_no_objects = 11):

    colors = []
    for r in range(250,50,-25):
        for g in range(250,50,-25):
            for b in range(250,50,-25):
                colors.append([r,g,b])



    if not os.path.exists(os.path.join(work_dir,'output','coco')):
        os.mkdir(os.path.join(work_dir,'output','coco'))
    img_path = os.path.join(work_dir,'data','rgb')
    json_dir = os.path.join(work_dir,'data','labels_json')
    save_path = os.path.join(work_dir,'output','coco')
    
    return True


def dil2D(arr):
    Kx_weight = torch.tensor(
                     [[8, 4],
                      [2, 1]])
    Kx_weight = Kx_weight.repeat(1,1,1,1)
    nc = torch.nn.functional.conv2d(input=arr.unsqueeze(0).unsqueeze(0).float(),
                                                weight=Kx_weight.float(), stride=(1,1), padding=(1,1))
    be = (nc != 0) & (nc != 15)
    be = be.squeeze(0).squeeze(0)
    return be
    
    
def json2coco(json_dir):
    for j in tqdm(os.listdir(json_dir)):
        with open(os.path.join(json_dir,j)) as j_file:
            data = json.load(j_file)
        color_index = 0
        image = cv2.imread(os.path.join(img_path,j.replace('.json','.png')))
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        labels = np.zeros((data['imageHeight'],data['imageWidth'],3),dtype=np.uint8)
        edges = np.zeros((data['imageHeight'],data['imageWidth'],3),dtype=np.uint8)
        edge = np.zeros((data['imageHeight'],data['imageWidth']),dtype=np.uint8)
        for c in data['shapes']:
            img = Image.new('L', (data['imageWidth'],data['imageHeight']), 0)
            coco_points = []
            for p in c['points']:
                coco_points.append(p[0])
                coco_points.append(p[1])
            status = ImageDraw.Draw(img).polygon(coco_points, outline=1, fill=1)
            mask = np.array(img,dtype= np.bool)
            labels[mask,:] = colors[color_index]
            edge = edge|(dil2D(torch.tensor(mask)).int().numpy()*255)[1:,1:]
            color_index += 1
        edges[:,:,0] = edge
        edges[:,:,1] = edge
        edges[:,:,2] = edge
    

def json2coco(json_dir):
    images_train = []
    annotations_train = []
    images_val = []
    annotations_val = []
    img_id = 1
    ann_id = 1
    train_list = []
    val_list = []
    for j in tqdm(os.listdir(json_dir)):
        with open(os.path.join(json_dir,j)) as j_file:
            data = json.load(j_file)
        img = {}
        img['license'] = None
        img['file_name'] = j.replace('.json','.png')
        img['coco_url'] = None
        img['height'] = data['imageHeight']
        img['width'] = data['imageWidth']
        img['id'] = img_id
        if img_id <= 430:
            images_train.append(img)
            train_list.append(j.replace('.json','.png'))
        else:
            images_val.append(img)
            val_list.append(j.replace('.json','.png'))
        for c in data['shapes']:
            ann = {}
            ann['segmentation']=[[]]
            x = []
            y = []
            xmax = 0.0
            xmin = 10000000000000.0
            ymax = 0.0
            ymin = 10000000000000.0
            for p in c["points"]:
                xmax = max(xmax,p[0])
                ymax = max(ymax,p[1])
                xmin = min(xmin,p[0])
                ymin = min(ymin,p[1])
                ann['segmentation'][0].append(float(p[0]))
                ann['segmentation'][0].append(float(p[1]))
            ann['area'] = float((xmax-xmin)*(ymax-ymin))
            ann['iscrowd'] = 0
            ann['image_id'] = img_id
            ann['bbox'] = []
            ann['bbox'].append(float(xmin))
            ann['bbox'].append(float(ymin))
            ann['bbox'].append(float(xmax-xmin))
            ann['bbox'].append(float(ymax-ymin))
                            
            ann['category_id'] = 1
            ann['id'] = ann_id
            ann_id += 1
            if img_id <= 430:
                annotations_train.append(ann)
            else:
                annotations_val.append(ann)
        img_id += 1
        
    categories = [{
          "supercategory": "chicken",
          "id": 1,
          "name": "chicken"
        }]
    train = {}
    val = {}
    train['images'] = images_train
    val['images'] = images_val
    train['annotations'] = annotations_train
    val['annotations'] = annotations_val
    train['categories'] = categories
    val['categories'] = categories

    with open(os.path.join('work_dir','output','coco_train.json'),'w') as f:
        json.dump(train,f,indent=2)

    with open(os.path.join('work_dir','output','coco','coco_val.json'),'w') as f:
        json.dump(val,f,indent=2)
        
    print('Files written to:',os.path.join('output','coco'))
    return True


# EOF
