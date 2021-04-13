# -*- coding: utf-8 -*-

# pre_datafolder.py version 2.0
from sklearn.model_selection import train_test_split, StratifiedKFold
import shutil
import pandas as pd
from PIL import Image, ImageOps
import random
import os.path as osp
import os
import numpy as np
import math
import cv2
import glob
import multiprocessing
import re
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
Image.MAX_IMAGE_PIXELS = None
# %matplotlib inline
train_mask_dir = "../Colonoscopy_tissue_segment_dataset/train_mask"  # create  train mask
train_dir = "../Colonoscopy_tissue_segment_dataset/train"  # create train data
val_mask_dir = "../Colonoscopy_tissue_segment_dataset/val_mask"
val_dir = "../Colonoscopy_tissue_segment_dataset/val"
try:
    os.makedirs(train_mask_dir)
    os.makedirs(train_dir)
    os.makedirs(val_dir)
    os.makedirs(val_mask_dir)
except:
    pass

def move_file(sourse_fn, target_fn,is_mask=False):
    if is_mask==False:
        for full_fn in sourse_fn:
            fn = full_fn.split("/")[-1]
            shutil.move(full_fn, os.path.join(target_fn, fn))

    else:
        for full_fn in sourse_fn:
            fn = full_fn.split("/")[-1].split(".")[0]+"_mask.jpg"  # 无前缀纯mask文件名
            source_path = full_fn.rsplit(".",1)[0]+"_mask.jpg"  # 源mask名
            try:
                shutil.move(source_path, os.path.join(target_fn, fn))
            except:
                size = Image.open(full_fn).size
                np_img = np.zeros(size[::-1], dtype=np.uint8)
                mask_img = Image.fromarray(np_img)
                mask_img.save(os.path.join(target_fn, fn))


def train_val_split_by_id(input_path_list,scale_n = 0.1):
    '''args:
    input_path_list:dataset wait to be split
    scale_n: propation of validation
    '''
    prefix = input_path_list[0].rsplit("/",1)[0]+"/"
    idx = [fn.split("/")[-1] for fn in input_path_list]  # remove repeat information
    pattern = r"2\d{3}-\d{2}-\d{2}"
    sp_list=[]
    for id_ in idx:
        sp_list.append(re.split(pattern ,id_))
    scale_n = math.floor(len(sp_list)*scale_n)
    print("length of input set:{} \n lenghth of val set:{}".format(len(idx) , scale_n))

    id_list=[]       #  main_id information of sufferer
    for i,j in enumerate(sp_list):
        if j[0]=="":        # 如果是以日期开头的数据 顺位取第二组key info
            tmp = re.split("-lv1",j[1])[0]
            id_list.append([i,tmp])
        elif re.search(r"^\d{4}\D\d+",j[0]):         # 如果前面为年份信息 取中间的key info
            tmp = re.match(r"^\d{4}\D\d+",j[0])[0]
            id_list.append([i,tmp])
        else:
            id_list.append([i,j[0]])
    id_df= pd.DataFrame(id_list,columns=["index","id"])
    tmp = id_df.groupby('id').groups
    count_num = 0
    train_dataset = []
    val_dataset = []
    for i in range(len(tmp)):        #  抽取val set
        k= random.sample(tmp.keys(), 1)[0]  # 随机一个字典中的key，第二个参数为限制个数
        for j in tmp[k]:
            val_dataset.append(idx[j])
        count_num += len(tmp[k])
        del tmp[k] # 删除已抽取的键值对
        if count_num>=scale_n:
            break
    for i in val_dataset:
        idx.remove(i)
    for i in range(len(idx)):
        idx[i]= prefix+idx[i]
    for i in range(len(val_dataset)):
        val_dataset[i]= prefix+val_dataset[i]

    return idx, val_dataset

if __name__ == "__main__":

    # 非mask  原图的名称
    url2 = "../Colonoscopy_tissue_segment_dataset/tissue-train-pos-v1/*[!_mask].jpg"
    all_pos_fn = glob.glob(url2)
    n_pos = len(all_pos_fn)
    print(n_pos)
    #url1 = "/Users/eggwardhan/Documents/cv/DigestPath2019/Colonoscopy_tissue_segment_dataset/tissue-train-pos-v1/*_mask.jpg"
    url3 = "../Colonoscopy_tissue_segment_dataset/tissue-train-neg/*.jpg"
    all_neg_fn = glob.glob(url3)[:80]
    # all_train_fn.extend(all_neg_fn)
    #  split test and train dataset
    total_samples = len(all_neg_fn)+len(all_pos_fn)
    # idx = np.arange(total_samples)
    pos_train, pos_val = train_val_split_by_id(all_pos_fn)
    neg_train, neg_val = train_val_split_by_id(all_neg_fn)
    train_fn = pos_train.copy()
    train_fn.extend(neg_train)
    # print(train_fn)
    val_fn = pos_val.copy()
    val_fn.extend(neg_val)
    print("No. of train files:", len(train_fn))
    print("No. of val files:", len(val_fn))
    try:
        move_file(train_fn, train_mask_dir, True)
        move_file(train_fn,train_dir)
        move_file(val_fn, val_mask_dir, True)
        move_file(val_fn, val_dir)
    except TypeError:
        print(TypeError)

    '''
    train_fn, val_fn = train_test_split(
        all_train_fn, stratify=mask_df.labels, test_size=0.1, random_state=10)'''

'''    move_file(train_fn,train_dir)
    move_file(train_fn, train_mask_dir, True)
    move_file(val_fn, val_dir)
    move_file(val_fn, val_mask_dir, True)
'''
