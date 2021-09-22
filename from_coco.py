#!/usr/bin/env python
# coding: utf-8

import json
import cv2
import os
import numpy as np
import shutil
import pycocotools.mask as mask
import matplotlib.pyplot as plt
from tqdm import tqdm


def decode(rleObjs):
    if type(rleObjs) == list:
        return mask.decode(rleObjs)
    else:
        return mask.decode([rleObjs])[:,:,0]
    
def init():

    work_dir = ''

    with open(os.path.join(work_dir,'data','coco','segm.json')) as f:
        data = json.load(f)

    with open(os.path.join(work_dir,'data','coco','segm.json')) as f:
        data_gt = json.load(f)


    img_path = os.path.join(work_dir,'data','coco','train2017')
    save_path = os.path.join(work_dir,'output','coco','results')
    vid_path = os.path.join(work_dir,'output','video')

    id2class = {1:"UNO",2:"DOS",3:"CUATRO",4:"CINCO"}

    max_no_objects = 4
    colors = []
    for r in range(250,50,-25):
        for g in range(250,50,-25):
            for b in range(250,50,-25):
                colors.append([r,g,b])
                
    return True

def coco2img_mask(data_gt):

    for i in data_gt['images']:
        img = cv2.imread(os.path.join(img_path, i['file_name']))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        m0 = np.zeros((480,848),dtype= np.uint8)
        for s in data_gt:
            if s['image_id'] == i['id']:
                rle = s['segmentation']
                m = decode(rle)
                m0 = m0|m
    #             img = cv2.bitwise_or(img,img,mask = m)
        img_new = cv2.bitwise_or(img,img,mask = m0)
        cont_img = np.concatenate((img,img_new),axis = 1)
        plt.imsave(os.path.join(save_path,i['file_name']),cont_img)

def coco2video(data_gt, destination_dir, th = 100):

    name2id = {}
    for i in data_gt['images']:
        name2id[i['file_name']] =  i['id']


    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(vid_path, fourcc, int(30), (int(1696), int(480)))
    n = 0
    for i in tqdm(range(0,300000,30)):
        if os.path.exists(os.path.join(destination_dir,"frame"+str(i)+".png")):
            img = cv2.imread(os.path.join(destination_dir,"frame"+str(i)+".png"))
    #         img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            m0 = np.zeros((480,848),dtype= np.uint8)
            for s in data:
                if s['image_id'] == name2id["frame"+str(i)+".png"]:
                    rle = s['segmentation']
                    m = decode(rle)
                    m0 = m0|m
        #             img = cv2.bitwise_or(img,img,mask = m)
            img_new = cv2.bitwise_or(img,img,mask = m0)
            cont_img = np.concatenate((img,img_new),axis = 1)
    #         plt.imsave(os.path.join(save_path,"frame"+str(i)+".png"),cont_img)
            writer.write(cont_img)
            n += 1
        if n > th:
            break
    writer.release()






