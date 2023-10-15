#coding:utf8
import os
from torch.utils import data

import numpy as np
from sklearn.utils import shuffle
import nibabel as nib
import random
from random import gauss
from transformations import rotation_matrix
from scipy.ndimage.interpolation import map_coordinates
import glob

def make_rand_vector(dims):
    vec = [gauss(0, 1) for i in range(dims)]
    mag = sum(x**2 for x in vec) ** .5
    return [x/mag for x in vec]

def coordinateTransformWrapper(X_T1,maxDeg=0,maxShift=7.5,mirror_prob = 0.5):
    randomAngle = np.radians(maxDeg*2*(random.random()-0.5))
    unitVec = tuple(make_rand_vector(3))
    shiftVec = [maxShift*2*(random.random()-0.5),
                maxShift*2*(random.random()-0.5),
                maxShift*2*(random.random()-0.5)]
    X_T1 = coordinateTransform(X_T1,randomAngle,unitVec,shiftVec)
    return X_T1

def coordinateTransform(vol,randomAngle,unitVec,shiftVec,order=1,mode='constant'):
    #from transformations import rotation_matrix
    ax = (list(vol.shape))
    ax = [ ax[i] for i in [1,0,2]]
    coords=np.meshgrid(np.arange(ax[0]),np.arange(ax[1]),np.arange(ax[2]))

    # stack the meshgrid to position vectors, center them around 0 by substracting dim/2
    xyz=np.vstack([coords[0].reshape(-1)-float(ax[0])/2,     # x coordinate, centered
               coords[1].reshape(-1)-float(ax[1])/2,     # y coordinate, centered
               coords[2].reshape(-1)-float(ax[2])/2,     # z coordinate, centered
               np.ones((ax[0],ax[1],ax[2])).reshape(-1)])    # 1 for homogeneous coordinates
    
    # create transformation matrix
    mat=rotation_matrix(randomAngle,unitVec)

    # apply transformation
    transformed_xyz=np.dot(mat, xyz)

    # extract coordinates, don't use transformed_xyz[3,:] that's the homogeneous coordinate, always 1
    x=transformed_xyz[0,:]+float(ax[0])/2+shiftVec[0]
    y=transformed_xyz[1,:]+float(ax[1])/2+shiftVec[1]
    z=transformed_xyz[2,:]+float(ax[2])/2+shiftVec[2]
    x=x.reshape((ax[1],ax[0],ax[2]))
    y=y.reshape((ax[1],ax[0],ax[2]))
    z=z.reshape((ax[1],ax[0],ax[2]))
    new_xyz=[y,x,z]
    new_vol=map_coordinates(vol,new_xyz, order=order,mode=mode)
    return new_vol

def generate_label(label,sigma = 2, bin_step = 1):
    labelset = np.array([i * bin_step + 12 for i in range(int(88 / bin_step))])

    dis = np.exp(-1/2. * np.power((labelset - label)/sigma/sigma, 2))
    dis = dis / dis.sum()
    return dis, labelset

from sklearn.model_selection import StratifiedKFold
def make_train_test(length, fold_idx, seed = 0, ns_splits = 5):
    assert 0 <= fold_idx and fold_idx < 5, "fold_idx must be from 0 to 9."
    skf = StratifiedKFold(n_splits=ns_splits, shuffle=True, random_state=seed)
    labels = np.zeros((length))

    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, test_idx = idx_list[fold_idx]
    return train_idx, test_idx

import pandas as pd


class AllData_DataFrame(data.Dataset):

    def __init__(self, df_csv, config,train=True,out_label = 1):
        self.train = train
        self.df = pd.read_csv(df_csv)
        self.out_label = out_label
        self.fold = config.fold
        self.da = config.da
        if self.da is True:
            print(">>>>>>>>>>>>>>> Big DA for student. <<<<<<<<<<<<<<<<<")

        if train is True:
            term = 1
        elif train is False:
            term = 0
        else:
            raise KeyError()

        self.lists = list(self.df[(self.df[f'fold_{self.fold}']==term) & (self.df['label'] != self.out_label)]['data'])
        self.lbls = list(self.df[(self.df[f'fold_{self.fold}']==term) & (self.df['label'] != self.out_label)]['label'])
        min_label, max_label = min(self.lbls), max(self.lbls)
        self.lbls = [0 if f == min_label else 1 for f in self.lbls]
        #[int(f.split("/")[-1].split("_")[1][:-4]) for f in self.lists]
        #assert self.lbls == list(self.df[(self.df[term]==1) & (self.df['label'] != self.out_label)]['label'])
        if self.train is True:
            print(f"{term} files: {len(self.lists)}, min_label: {min_label}, max_label: {max_label} \
                    label dist: {len([f for f in self.lbls if f ==min_label])}-{len([f for f in self.lbls if f !=min_label])}, ",set(self.lbls))
        
        self.imgs = np.array(self.lists)
        self.lbls = np.array(self.lbls)

    def __getitem__(self,index, generate_age_dist = False):
        if self.imgs[index].endswith(".nii.gz"):
            img = nib.load(self.imgs[index]).get_fdata()
        elif self.imgs[index].endswith(".npy"):
            img = np.load(self.imgs[index])[0]
        else:
            print("failed loading... Please check files.")
            exit()
        lbl = self.lbls[index]
        if generate_age_dist is True:
            lbl_y, lbl_bc = generate_label(lbl, sigma = 2, bin_step= 4)
        else:
            lbl_y = lbl
            lbl_bc = lbl
        
        if self.train:
            if self.da is True:
                img = coordinateTransformWrapper(img,maxDeg=30,maxShift=5, mirror_prob = 0)
            else:
                img = coordinateTransformWrapper(img,maxDeg=10,maxShift=5, mirror_prob = 0)
        else:
            img = img

        img = img[np.newaxis,...]

        return img, lbl, lbl_y, lbl_bc, index
    
    def __len__(self):
        return len(self.imgs)

class AllData_DDGSD(AllData_DataFrame):
    def __getitem__(self,index, generate_age_dist = False):
        if self.imgs[index].endswith(".nii.gz"):
            img = nib.load(self.imgs[index]).get_fdata()
        elif self.imgs[index].endswith(".npy"):
            img = np.load(self.imgs[index])[0]
        else:
            print("failed loading... Please check files.")
            exit()
        lbl = self.lbls[index]
        if generate_age_dist is True:
            lbl_y, lbl_bc = generate_label(lbl, sigma = 2, bin_step= 4)
        else:
            lbl_y = lbl
            lbl_bc = lbl
        
        if self.train:
            img1 = coordinateTransformWrapper(img,maxDeg=10,maxShift=0, mirror_prob = 0)
            img2 = coordinateTransformWrapper(img,maxDeg=0,maxShift=5, mirror_prob = 0)
        else:
            img1 = img
            img2 = img

        img1 = img1[np.newaxis,...]
        img2 = img2[np.newaxis,...]

        return img1,img2, lbl, lbl_y, lbl_bc, index
    
class AllData(data.Dataset):

    def __init__(self, root, train=True,test=False):
        self.train = train
        self.root = root
        all_files = shuffle(sorted(glob.glob(os.path.join(self.root , "*"))), random_state = 1111) 
        all_files = [f for f in all_files if f.endswith(".nii.gz") or f.endswith(".npy")]

        all_files = [f for f in all_files if float(f.split("/")[-1].split("_")[1]) > 18 and float(f.split("/")[-1].split("_")[1]) < 94] 
        print("All filese: ", len(all_files))
        rest_idx, val_idx = make_train_test(len(all_files),0, ns_splits=5)
        train_idx, test_idx = make_train_test(len(rest_idx),0, ns_splits=8)
        #train_idx = [f for f in range(len(all_files))]

        if train:
            self.imgs = np.array(all_files)[rest_idx][train_idx]
            self.lbls = [float(f.split("/")[-1].split("_")[1]) for f in self.imgs]

            assert len(self.imgs) > 0, "No images found"
            print("Train files: ", len(self.imgs), min(self.lbls), max(self.lbls))
        elif test is False:
            self.imgs = np.array(all_files)[val_idx]
            self.lbls = [float(f.split("/")[-1].split("_")[1]) for f in self.imgs]
            print("valid files: ", len(self.imgs), min(self.lbls), max(self.lbls))
        else:
            self.imgs = np.array(all_files)[rest_idx][test_idx]
            self.lbls = [float(f.split("/")[-1].split("_")[1]) for f in self.imgs]           
            print("Test files: ", len(self.imgs), min(self.lbls), max(self.lbls))
    def __getitem__(self,index):
        if self.imgs[index].endswith(".nii.gz"):
            img = nib.load(self.imgs[index]).get_fdata()
        elif self.imgs[index].endswith(".npy"):
            img = np.load(self.imgs[index])
        else:
            print("failed loading... Please check files.")
            exit()

        lbl = self.lbls[index]
        lbl_y3, lbl_bc3 = generate_label(lbl, sigma = 2, bin_step= 4)
        
        if self.train:
            img = coordinateTransformWrapper(img,maxDeg=10,maxShift=5, mirror_prob = 0)
        else:
            img = img

        img = img[np.newaxis,...]

        return img, lbl, lbl_y3, lbl_bc3, index
    
    def __len__(self):
        return len(self.imgs)

import torch
class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target, self.next_y3, self.next_bc3, self.next_indices = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            self.next_y3 = None
            self.next_bc3 = None
            self.next_indices = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True).float()
            self.next_target = self.next_target.cuda(non_blocking=True).long()
            self.next_y3 = self.next_y3.cuda(non_blocking=True).float()
            self.next_bc3 = self.next_bc3.cuda(non_blocking=True).float()
            self.next_indices = self.next_indices.cuda(non_blocking=True).long()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        y3 = self.next_y3
        bc3 = self.next_bc3
        indices = self.next_indices

        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        if y3 is not None:
            y3.record_stream(torch.cuda.current_stream())
        if bc3 is not None:
            bc3.record_stream(torch.cuda.current_stream())
        if indices is not None:
            indices.record_stream(torch.cuda.current_stream())

        self.preload()

        return input, target, y3, bc3,indices

class data_prefetcher_DDGSD():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_input1,self.next_input2, self.next_target, self.next_y3, self.next_bc3, self.next_indices = next(self.loader)
        except StopIteration:
            self.next_input1 = None
            self.next_input2 = None
            self.next_target = None
            self.next_y3 = None
            self.next_bc3 = None
            self.next_indices = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input1 = self.next_input1.cuda(non_blocking=True).float()
            self.next_input2 = self.next_input2.cuda(non_blocking=True).float()
            self.next_target = self.next_target.cuda(non_blocking=True).long()
            self.next_y3 = self.next_y3.cuda(non_blocking=True).float()
            self.next_bc3 = self.next_bc3.cuda(non_blocking=True).float()
            self.next_indices = self.next_indices.cuda(non_blocking=True).long()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input1 = self.next_input1
        input2 = self.next_input2
        target = self.next_target
        y3 = self.next_y3
        bc3 = self.next_bc3
        indices = self.next_indices

        if input1 is not None:
            input1.record_stream(torch.cuda.current_stream())
        if input2 is not None:
            input2.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        if y3 is not None:
            y3.record_stream(torch.cuda.current_stream())
        if bc3 is not None:
            bc3.record_stream(torch.cuda.current_stream())
        if indices is not None:
            indices.record_stream(torch.cuda.current_stream())

        self.preload()

        return input1,input2, target, y3, bc3,indices