# -*- coding: utf-8 -*-

import time
import datetime
import scipy.io as sio
import tifffile
from scipy import signal
import numpy as np
from scipy import misc
from scipy.io import loadmat
import sys
from numpy import *
from torch.nn import functional
from torch import nn
import torch
import os
import torch.utils.data as data
import metrics
import h5py

def fspecial(func_name,kernel_size,sigma):
    if func_name=='gaussian':
        m=n=(kernel_size-1.)/2.
        y,x=ogrid[-m:m+1,-n:n+1]
        h=exp( -(x*x + y*y) / (2.*sigma*sigma) )
        h[ h < finfo(h.dtype).eps*h.max() ] = 0
        sumh=h.sum()
        if sumh!=0:
            h/=sumh
        return h
    if func_name=='mean':
        m=n=int((kernel_size-1.)//2)
        # print(type(m))
        # exit()
        h = np.ones([m,n])
        sumh=h.sum()
        h/=sumh
        return h



def Gaussian_downsample(x,psf,s):
    y=np.zeros((x.shape[0],int(x.shape[1]/s),int(x.shape[2]/s)))
    if x.ndim==2:
        x=np.expand_dims(x,axis=0)
    for i in range(x.shape[0]):
        x1=x[i,:,:]
        # x2=x1
        x2=signal.convolve2d(x1,psf, boundary='symm',mode='same')
        y[i,:,:]=x2[0::s,0::s]
    return y




