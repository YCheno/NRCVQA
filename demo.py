import argparse
import json
import math
import os
import re
import subprocess

import cv2
import numpy as np
import skvideo.io
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import open_clip
import torchvision
from config import get_config
from torchvision import transforms
from einops import rearrange, reduce, repeat
from PIL import Image
from scipy.stats import kendalltau, pearsonr, spearmanr
from timm.utils import AverageMeter
from torch.utils.data import DataLoader, Dataset
from skimage import feature as skif


from config import get_config
from models.build import build_model
from models.vb_cnn_transformer import TextEncoder



def encode_text_prompts(prompts,model,device="cuda"):
        tokenizer = open_clip.get_tokenizer("RN50")
        text_tokens = tokenizer(prompts).to(device)
        with torch.no_grad():
            embedding = model.token_embedding(text_tokens)
            text_features = model.encode_text(text_tokens).float()
        return text_tokens, embedding, text_features

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




def load_mp4(file_path, frame_height, frame_width, stride_t=0, frameWant=32, start=0, transform=None):

    mean = 0.458971
    std = 0.225609

    # just load the luminance channel of the input video
    video = skvideo.io.vread(file_path)

    ret = []
    ret_lbp = []
    ret_dw = []
    get = 1
    frameNum = video.shape[0]
    transform = transforms.CenterCrop((720, 1280))

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
            RGB_lbp = (RGB_lbp - mean) / std
            
            frame = frame.astype('float32') / 255.
            frame = rearrange(frame, 'h w c -> c 1 h w')
            frame = (frame - mean) / std
            
            frame_copy = video[i].astype('float32') / 255.
            frame_copy = rearrange(frame_copy, 'h w c -> c 1 h w')
            frame_copy = (frame_copy - mean) / std
            ret_sized = torch.from_numpy(frame_copy)
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
    return ret, ret_lbp, ret_dw



def load_video(config, transform=None):

   
    video, video_lbp, video_dw = load_mp4(config.video_path, config.frame_height, config.frame_width,
                         frameWant=config.frameWant, transform=transform)

    if min(config.frame_height, config.frame_width) <= 400:
        stride_x, stride_y = 224, 224

    if config.stride_x and config.stride_y:
        spatial_crop = CropSegment(config.crop_size_x, config.crop_size_y, config.stride_x, config.stride_y)
        video = spatial_crop(video)
        video_lbp = spatial_crop(video_lbp)

    return video, video_lbp, video_dw




def load_pre_trained(config, path, prompts_texture, prompts_aesthetic, prompts_content, ctx_number):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    clip_model, _, _ = open_clip.create_model_and_transforms("RN50",pretrained="openai")
    clip_model = clip_model.to(device)
    
    text_encoder = TextEncoder(clip_model).to(device)
    
    text_tokens_T, embedding_T, text_T = encode_text_prompts(prompts_texture , clip_model, device=device)
    text_tokens_A, embedding_A, text_A = encode_text_prompts(prompts_aesthetic , clip_model, device=device)
    text_tokens_C, embedding_C, text_C = encode_text_prompts(prompts_content , clip_model, device=device)
    clip_visual = clip_model.visual.to(device)
    
    model = build_model(config.model,text_tokens_T,text_tokens_A,text_tokens_C,embedding_T,embedding_A,embedding_C,text_encoder,ctx_number)

    model_dict = model.state_dict()
    if config.model != 'pre_train':
        pre_dict = torch.load(path, device)

        
        pretrained_dict = {k: v for k,
                           v in pre_dict.items() if k in model_dict}
        
        model_dict.update(pretrained_dict)
    else:
        pretrained_model = torch.load(path, device)#['model']
        print(name1)
        if 'head.weight' in pretrained_model:
            pretrained_model.pop('head.weight')
        if 'head.bias' in pretrained_model:
            pretrained_model.pop('head.bias')
        pretrained_dict = {
            k: v for k, v in pretrained_model.items() if 'backbone.' + k in model_dict}
        model_dict.update(pretrained_dict)
        
        print(f"length of pretrained dict : {len(pretrained_dict)}")

    model.load_state_dict(model_dict, strict=False)
    model.to(device)
    return model, text_T, text_A, text_C, clip_visual, device


def predict(config):


    prompts_texture = [ 
    f"a photo with simple texture",
    f"a photo with slightly complex texture",
    f"a photo with generally complex texture",
    f"a photo with complex texture",
    f"a photo with highly complex texture" ]
    prompts_content = [ 
    f"a photo with excellent content",
    f"a photo with good content",
    f"a photo with fair content",
    f"a photo with poor content",
    f"a photo with bad content" ]
    prompts_aesthetic = [ 
   # f"a excellent aesthetic photo" ,
    f"a excellent aesthetic photo",
    f"a good aesthetic photo",
    f"a fair aesthetic photo",
    f"a poor aesthetic photo",
    f"a bad aesthetic photo" ]
    

    prefix = os.path.abspath('.')
    VQAModel, text_T, text_A, text_C, clip_visual, device = load_pre_trained(config, config.pre_trained_path, prompts_texture, prompts_aesthetic, prompts_content, ctx_number)


    video, video_lbp, video_dw = load_video(config)
    
    A_feats = []
    T_feats = []

    N, C, D, H, W = video.size()
    clip = torch.randint(10000000, (1,)) % N
    video = video[ clip, :, :, :, :]
    video_lbp = video_lbp[clip, :, :, :, :]
    video = video.to(device)
    video_lbp = video_lbp.to(device)
    video_dw = video_dw.to(device).unsqueeze(0)
    
    video_lbp = video_lbp.permute(0, 2, 1, 3, 4)
    video_lbp = video_lbp.contiguous().view(D, C, H, W)
    video_dw = video_dw.permute(0, 2, 1, 3, 4)
    video_dw = video_dw.contiguous().view( D, C, H, W)
    
    with torch.no_grad():
        img_t_feats =  clip_visual(video_lbp)
        img_feats =  clip_visual(video_dw)
        logits_T = img_t_feats @ text_T.t()
        logits_C = img_feats @ text_C.t()
        logits_A = logits_A @ text_A.t()
        logits_T = logits_T.softmax(dim=-1)
        logits_C = logits_C.softmax(dim=-1)
        ilogits_A = logits_A.softmax(dim=-1)
        
        T_feats = logits_T.contiguous().view(1, D * 5)
        logits_C = logits_C.contiguous().view(1, D * 5)
        logits_A = logits_A.contiguous().view(1, D * 5)
        AC_feats = torch.cat((logits_A,logits_C), dim=-1)
               
    
    pred = VQAModel(video, AC_feats, T_feats)
    pred = torch.sigmoid(pred)
    print(f"Predicted Score = {pred}")


def parse_option():
    parser = argparse.ArgumentParser(
        'Demo for Self-Supervised Representation Learning for Video Quality Assessment', add_help=False)

    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--video_path', type=str, default='/mnt/wwn-0x5000cca0c3e1998a/cartoon/test.mp4',
                        help="path for test video")
    parser.add_argument('--frame_width', type=int, default=1280,
                        help="frame width for test video")
    parser.add_argument('--frame_height', type=int, default=720,
                        help="frame height for test video")
    parser.add_argument('--stride_x', type=int, default=224,
                        help='stride size_x for test video')
    parser.add_argument('--stride_y', type=int, default=224,
                        help='stride size_y for test video')
    parser.add_argument('--crop_size_x', type=int, default=224,
                        help='crop size_x for test video')
    parser.add_argument('--crop_size_y', type=int, default=224,
                        help='crop size_y for test video')
    parser.add_argument('--model', type=str,
                        default='NRCVQA', help='Model Type')
    parser.add_argument('--frameWant', type=int, default=12,
                        help='test frame for test video')
    parser.add_argument('--pre_trained_path', type=str,
                        default='./checkpoints/NRCVQA.pth', help='pretrained weight path')
    parser.add_argument('--fine_tune', default=True, type=bool, help='Fine Tune Or Not')

    config = parser.parse_args()

    return config

if __name__ == '__main__':


    config = parse_option()
    predict(config)

