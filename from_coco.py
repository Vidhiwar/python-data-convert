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




for i in data_gt['images']:
    img = cv2.imread(os.path.join(img_path, i['file_name']))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    m0 = np.zeros((480,848),dtype= np.uint8)
    for s in data:
        if s['image_id'] == i['id']:
            rle = s['segmentation']
            m = decode(rle)
            m0 = m0|m
#             img = cv2.bitwise_or(img,img,mask = m)
    img_new = cv2.bitwise_or(img,img,mask = m0)
    cont_img = np.concatenate((img,img_new),axis = 1)
    plt.imsave(os.path.join(save_path,i['file_name']),cont_img)



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
    if n > 5400:
        break
writer.release()
            


# In[46]:


img_path = os.path.join('/home/vidhiwar/Data/chicken/coco/','test2017')
for i in tqdm(data_gt['images']):
    img = cv2.imread(os.path.join(img_path, i['file_name']))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    m0 = np.zeros((480,848,3),dtype= np.uint8)
    c_index = 0
    for s in data:
        if s['image_id'] == i['id']:
            rle = s['segmentation']
            m = decode(rle)
            m = np.array(m,dtype = np.bool)
            if c_index >= 10:
                c_index = 0
            m0[m,:] = colors[c_index]
            c_index += 1
#             img = cv2.bitwise_or(img,img,mask = m)
#     img_new = cv2.bitwise_or(img,img,mask = m0)
#     cont_img = np.concatenate((img,img_new),axis = 1)
    cont_img= cv2.addWeighted(img,0.65,m0,0.35,0)
    plt.imsave(os.path.join(save_path,i['file_name']),cont_img)


# In[14]:


with open('/home/vidhiwar/work_dir/mrcnn_chicken/inference/coco_2017_val/segm.json') as f:
    data = json.load(f)
    
with open('/home/vidhiwar/Data/chicken/coco/annotations/instances_val2017.json') as f:
    data_gt = json.load(f)
# destination_dir = os.path.join('/home/vidhiwar/Data/chicken/coco/','val2017')
name2id = {}
for i in data_gt['images']:
    name2id[i['file_name']] =  i['id']
    
destination_dir = os.path.join('/home/vidhiwar/Data/chicken/coco/','val2017')    
fourcc = cv2.VideoWriter_fourcc(*'XVID')
writer = cv2.VideoWriter(vid_path, fourcc, int(30), (int(848), int(480)))
n = 0
for i in tqdm(range(12960000,17000000,30)):
    if os.path.exists(os.path.join(destination_dir,"frame"+str(i)+".png")) and "frame"+str(i)+".png" in name2id.keys():
        img = cv2.imread(os.path.join(destination_dir,"frame"+str(i)+".png"))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        m0 = np.zeros((480,848,3),dtype= np.uint8)
        c_index = 0
        for s in data:
            if "frame"+str(i)+".png" in name2id.keys():
                if s['image_id'] == name2id["frame"+str(i)+".png"]:
                    rle = s['segmentation']
                    m = decode(rle)
                    m = np.array(m,dtype = np.bool)
                    if c_index >= 10:
                        c_index = 0
                    m0[m,:] = colors[int(s['category_id'])]
    #                 pos = (int((np.max(np.nonzero(m)[0])+np.min(np.nonzero(m)[0]))),int((np.max(np.nonzero(m)[1])+np.min(np.nonzero(m)[1]))/2))
    #                 font = cv2.FONT_HERSHEY_SIMPLEX
    #                 fontScale = 1
    #                 thickness = 2
    #                 m0 = cv2.putText(m0, id2class[int(s['category_id'])], pos, font,  fontScale, colors[c_index], thickness, cv2.LINE_AA) 
    #                 c_index += 1

    #             img = cv2.bitwise_or(img,img,mask = m)
    #     img_new = cv2.bitwise_or(img,img,mask = m0)
    #     cont_img = np.concatenate((img,img_new),axis = 1)
        cont_img= cv2.addWeighted(img,0.65,m0,0.35,0)
        plt.imsave(os.path.join(save_path,"frame"+str(i)+".png"),cont_img)
        cont_img = cv2.cvtColor(cont_img,cv2.COLOR_RGB2BGR)
        writer.write(cont_img)
#         

        
        
        n += 1
    if n >= 4262:
        break
writer.release()


# In[13]:



masks = []
ann_id = 1
for s in data:
    seg = {}
    seg['image_id'] = s['image_id']
    seg['category_id'] = s['category_id']
    seg['segmentation'] = s['segmentation']
    seg['prediction_score'] = s['score']
    seg['id'] = ann_id
    ann_id += 1
    masks.append(seg)
data_gt['annotations'] = masks
with open('/home/vidhiwar/Data/chicken/coco/res/chicken_seg.json','w') as f:
    json.dump(data_gt,f,indent=2)



# In[8]:


123428%30


# In[7]:


source_dir = os.path.join('/media/vidhiwar/Seagate_Desktop_Drive/','chicken_week1/')
destination_dir = os.path.join('/media/vidhiwar/Storage01/','chicken')
n = 0
for i in tqdm(range(0,2592000,123420)):
#     if n >= 21:
#         break
    file = 'frame'+str(i)+'.png'
    shutil.copy(os.path.join(source_dir,'rgb',file),os.path.join(destination_dir,'rgb',file))
    shutil.copy(os.path.join(source_dir,'d',file.replace('.png','.npy')),os.path.join(destination_dir,'d',file.replace('.png','.npy')))
    #shutil.copy(os.path.join(source_dir,'pc',file.replace('.png','.pcd')),os.path.join(destination_dir,'pc',file.replace('.png','.pcd')))
    n += 1
    
    


# In[42]:


name2id["frame26160.png"]


# In[ ]:




