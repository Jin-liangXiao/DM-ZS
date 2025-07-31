"""
Logger copied from OpenAI baselines to avoid extra RL-based dependencies:
https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/logger.py
"""

import os
import os.path as osp
import json
import datetime
from collections import OrderedDict

import torch
import torch.nn.functional as F
import numpy as np
import math

def mkdirs(paths):
    if isinstance(paths, str):
        os.makedirs(paths, exist_ok=True)
    else:
        for path in paths:
            os.makedirs(path, exist_ok=True)

def get_timestamp():
    return datetime.now().strftime('%y%m%d_%H%M%S')

def parse(args):
    args = vars(args)
    
    opt_path = args['baseconfig']
   # opt_path = args.config
    gpu_ids = args['gpu_ids']


    # remove comments starting with '//'
    json_str = ''
    with open(opt_path, 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line
    opt = json.loads(json_str, object_pairs_hook=OrderedDict)

    for key in args:
        if args[key] is not None:
            opt[key] = args[key]

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
    print('export CUDA_VISIBLE_DEVICES=' + gpu_ids)
    
    return opt

class NoneDict(dict):
    def __missing__(self, key):
        return None

def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt


def diff_3d(img, keepdim=True):
    # _, _, H, W = img.shape
    if keepdim:
        diff_x, diff_y, diff_z = torch.zeros_like(img).to(img.device), torch.zeros_like(img).to(img.device), torch.zeros_like(img).to(img.device)
        diff_x[..., :-1, :] = torch.diff(img, dim=-2)
        diff_x[..., -1, :] = img[..., 0, :] - img[..., -1, :]
        diff_y[..., :-1] = torch.diff(img, dim=-1)
        diff_y[..., -1] = img[..., 0] - img[..., -1]
        diff_z[:, :-1, ...] = torch.diff(img, dim=-3)
        diff_z[:, -1, ...] = img[:, 0, ...] - img[:, -1, ...]
    else:
        diff_x = torch.diff(img, dim=-2)
        diff_y = torch.diff(img, dim=-1)
        diff_z = torch.diff(img, dim=-3)
    return diff_x, diff_y, diff_z
