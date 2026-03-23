import os
import re
import json
import numpy as np
import subprocess
import torch
import math
import cv2
import skvideo.io
import torchvision
from einops import reduce, repeat, rearrange
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from skimage import feature as skif

def get_lbp_data(img,lbp_radius=1,lbp_point=8) :
    lbp_img = skif.local_binary_pattern(img,lbp_point,lbp_radius,'default')
    return lbp_img

def get_resize_function(size_h, size_w, target_ratio=1, random_crop=False):
    if random_crop:
        return torchvision.transforms.RandomResizedCrop(
            (size_h, size_w), scale=(0.40, 1.0)
        )
    if target_ratio > 1:
        size_h = int(target_ratio * size_w)
        assert size_h > size_w
    elif target_ratio < 1:
        size_w = int(size_h / target_ratio)
        assert size_w > size_h
    return torchvision.transforms.Resize((size_h, size_w))


def get_resized_video(
    video, size_h=224, size_w=224, random_crop=False, arp=False, **kwargs,
):
    video = video.permute(1, 0, 2, 3)
    resize_opt = get_resize_function(
        size_h, size_w, video.shape[-2] / video.shape[-1] if arp else 1, random_crop
    )
    video = resize_opt(video)
    return video.permute(1, 0, 2, 3)

class CropSegment(object):
    r"""
    Crop a clip along the spatial axes, i.e. h, w
    DO NOT crop along the temporal axis

    args:
        size_x: horizontal dimension of a segment
        size_y: vertical dimension of a segment
        stride_x: horizontal stride between segments
        stride_y: vertical stride between segments
    return:
        clip (tensor): dim = (N, C, D, H=size_y, W=size_x). N are segments number by applying sliding window with given window size and stride
    """

    def __init__(self, size_x, size_y, stride_x, stride_y):

        self.size_x = size_x
        self.size_y = size_y
        self.stride_x = stride_x
        self.stride_y = stride_y

    def __call__(self, clip):
        # input dimension [C, D, H, W]
        channel = clip.shape[0]
        depth = clip.shape[1]

        clip = clip.unfold(2, self.size_x, self.stride_x)
        clip = clip.unfold(3, self.size_y, self.stride_y)
        clip = clip.permute(2, 3, 0, 1, 4, 5)
        clip = clip.contiguous().view(-1, channel, depth, self.size_x, self.size_y)

        return clip


class True_DisTypeVideoDataset(Dataset):
    r"""
    A Dataset for a folder of videos

    args:
        subj_score_file (str): path to the subjective score file. It contains train/test split, ref list, dis list, fps list and mos list
        directory (str): the path to the directory containing all videos
        mode (str, optional): determines whether to read train/test data
        channel (int, optional): number of channels of a sample
        size_x: horizontal dimension of a segment
        size_y: vertical dimension of a segment
        stride_x: horizontal stride between segments
        stride_y: vertical stride between segments
    """

    def __init__(self, subj_score_file, directory, mode='train', channel=1, size_x=112, size_y=112, stride_x=80, stride_y=80, frameWant=32, transform=None):

        with open(subj_score_file, "r") as f:
            data = json.load(f)
        self.mode = mode
        self.video_dir = directory
        data = data[mode]
        # self.ref = data['ref']
        self.dis = data['dis']
        self.label = data['mos']
        self.disType = 0
        self.channel = channel
        self.size_x = size_x
        self.size_y = size_y
        self.stride_x = stride_x
        self.stride_y = stride_y
        self.frameWant = frameWant
        self.transform = transform
        self.mean = 0.458971
        self.std = 0.225609

    def __getitem__(self, index):

        dis = os.path.join(self.video_dir, self.dis[index])
        distype = 0
        label = float(self.label[index])
        framerate = 0
        frame_height = 0
        frame_width = 0

        if framerate <= 30:
            stride_t = 2
        elif framerate <= 60:
            stride_t = 4
        else:
            raise ValueError('Unsupported fps')


        if dis.endswith(('.mp4')):
            dis, dis_lbp, dis_dw = self.load_mp4(dis, frame_height, frame_width, stride_t, frameWant=self.frameWant, transform=self.transform)
        else:
            raise ValueError('Unsupported video format')
        
        label = torch.from_numpy(np.asarray(label))

        return dis, dis_lbp, dis_dw, label


    def load_mp4(self, file_path, frame_height, frame_width, stride_t, frameWant=32, start=0, transform=None):

        video = skvideo.io.vread(file_path)

        ret = []
        ret_edge = []
        ret_lbp = []
        ret_dw = []
        get = 1
        frameNum = video.shape[0]
        
        l = 5  #canny low
        u = 140   #canny high

        if frameWant != 0:
            stride_t = math.ceil(frameNum / frameWant) - 1
        else:
            stride_t = 1

        for i in range(frameNum):
            if i % stride_t == 0 and (frameWant == 0 or get <= frameWant):
                get += 1
                frame = video[i]
                frame = rearrange(frame, 'h w c -> c h w')
                if transform is not None:
                    img = torch.from_numpy(frame)
                    img = transform(img)
                    frame = img.numpy()
                frame = rearrange(frame, 'c h w -> h w c')
                (B, G, R) = cv2.split(frame)
                B_lbp = get_lbp_data(B)
                G_lbp = get_lbp_data(G)
                R_lbp = get_lbp_data(R)
                RGB_lbp = np.dstack((B_lbp, G_lbp, R_lbp))
                RGB_lbp = RGB_lbp.astype('float32') / 255.
                RGB_lbp = rearrange(RGB_lbp, 'h w c -> c 1 h w')
                RGB_lbp = (RGB_lbp - self.mean) / self.std
                
                frame = frame.astype('float32') / 255.
                frame = rearrange(frame, 'h w c -> c 1 h w')
                frame = (frame - self.mean) / self.std
                
                frame1 = video[i].astype('float32') / 255.
                frame1 = rearrange(frame1, 'h w c -> c 1 h w')
                frame1 = (frame1 - self.mean) / self.std
                ret_sized = torch.from_numpy(frame1)
                ret_sized = get_resized_video(ret_sized)
                
                ret.append(frame)
                ret_lbp.append(RGB_lbp)
                ret_dw.append(ret_sized)
        ret = np.concatenate(ret, axis=1)
        ret_lbp = np.concatenate(ret_lbp, axis=1)
        ret_dw = np.concatenate(ret_dw, axis=1)
        ret = torch.from_numpy(np.asarray(ret))
        ret_lbp = torch.from_numpy(np.asarray(ret_lbp))
        ret_dw = torch.from_numpy(np.asarray(ret_dw))
        
        frame_height = ret.shape[-2]
        frame_width = ret.shape[-1]
        
        if self.stride_x and self.stride_y:
            offset_v = (frame_height - self.size_y) % self.stride_y
            offset_t = int(offset_v / 4 * 2)
            offset_b = offset_v - offset_t
            offset_h = (frame_width - self.size_x) % self.stride_x
            offset_l = int(offset_h / 4 * 2)
            offset_r = offset_h - offset_l
            ret = ret[:, :, offset_t:frame_height -
                      offset_b, offset_l:frame_width-offset_r]
            ret_lbp = ret_lbp[:, :, offset_t:frame_height -
                      offset_b, offset_l:frame_width-offset_r]
            spatial_crop = CropSegment(
                self.size_x, self.size_y, self.stride_x, self.stride_y)
            ret = spatial_crop(ret)
            ret_lbp = spatial_crop(ret_lbp)
        
        
       # return ret
        return ret, ret_lbp, ret_dw

    def __len__(self):
        return len(self.dis)


if __name__ == '__main__':

    frame_height, frame_width = 272, 480
    size_x, size_y = 224, 224
    stride_x, stride_y = 224, 224
    
    if min(frame_height, frame_width) <= 400:
        stride_x, stride_y = 224, 224
    
    offset_v = (frame_height - size_y) % stride_y
    offset_t = int(offset_v / 4 * 2)
    offset_b = offset_v - offset_t
    offset_h = (frame_width - size_x) % stride_x
    offset_l = int(offset_h / 4 * 2)
    offset_r = offset_h - offset_l
    dis = torch.rand(1, 32, 480, 272)
    
    dis = dis[:, :, offset_t:frame_height-offset_b, offset_l:frame_width-offset_r]
    spatial_crop = CropSegment(size_x, size_y, stride_x, stride_y)
    # ref = spatial_crop(ref)
    dis = spatial_crop(dis)
    
    print(f"dis shape : {dis.shape}")
