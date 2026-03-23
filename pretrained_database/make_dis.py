import json
import logging
import os
import random
import subprocess
import time
from collections import OrderedDict
from datetime import datetime

import cv2
import numpy as np
import skvideo.io
from imgaug import augmenters as iaa
from tqdm import tqdm

DAY = "{0:%Y-%m-%d}".format(datetime.now())

# rootdir: 310 videos path
rootdir = ' '

# result dir: mp4
resultdir = ' '

def getlogger(path='logs', level=logging.INFO):
    logger = logging.getLogger()
    logger.setLevel(level)
    rq = time.strftime('%Y-%m-%d-%H-%M', time.localtime(time.time()))
    log_path = os.path.join(resultdir, path, DAY)
    if not os.path.exists(log_path):
        print("log_path don't existed...")
        os.makedirs(log_path)
    log_name = os.path.join(log_path, rq + '.log')
    logfile = log_name

    print("log file: {}".format(logfile))

    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger



def getJsonString(strFileName):
    strCmd = 'ffprobe -v quiet -print_format json -show_format -show_streams -i "' + \
        strFileName + '"'
    mystring = os.popen(strCmd).read()
    return mystring

def h264(ref_video, file, ref_path):
    mp = {1 : 33, 2 : 38, 3 : 42}
    #mp = {1 : 28, 2 : 40}
    for i in range(1, 4):
        dis_path = os.path.join(resultdir, 'h264', os.path.splitext(file)[0] +  '_h264_' + str(i) + '.mp4')
        checkDir(os.path.join(resultdir, 'h264'))
            
        strCmd='/home/amd7302/YXX/ffmpeg/ffmpeg -i '+ ref_path + ' ' + '-vcodec libx264' + ' -crf ' + str(mp[i]) + ' ' + dis_path
        os.system(strCmd)
        
def h265(ref_video, file, ref_path):
    mp = {1 : 35, 2 : 40, 3 : 45}
    for i in range(1, 4):
        dis_path = os.path.join(resultdir, 'h265', os.path.splitext(file)[0] +  '_h265_' + str(i) + '.mp4')

        checkDir(os.path.join(resultdir, 'h265'))
            
        strCmd='/home/amd7302/YXX/ffmpeg/ffmpeg -i '+ ref_path + ' ' + '-vcodec libx265' + ' -crf ' + str(mp[i]) + ' ' + dis_path
        os.system(strCmd)


def GaussNoise(ref_video, ref_file):
    for i in range(1, 4):    
        dis_path = os.path.join(resultdir, 'GaussNoise', os.path.splitext(ref_file)[0] +  '_GaussNois_' + str(i) + '.mp4')

        checkDir(os.path.join(resultdir, 'GaussNoise'))

        writer = skvideo.io.FFmpegWriter(dis_path)
        aug = iaa.AdditiveGaussianNoise(loc=0, scale=0.02*255*i)
        for idx, frame in enumerate(ref_video):
            writer.writeFrame(aug.augment_image(frame))
        writer.close()


def GaussBlur(ref_video, ref_file):
    for i in range(1, 4):    
        dis_path = os.path.join(resultdir, 'GaussBlur', os.path.splitext(ref_file)[0] +  '_GaussBlur_' + str(i) + '.mp4')

        checkDir(os.path.join(resultdir, 'GaussBlur'))

        writer = skvideo.io.FFmpegWriter(dis_path)
        # Gauss Blur
        ks = 2 * i + 1
        for idx, frame in enumerate(ref_video):
            writer.writeFrame(cv2.GaussianBlur(frame, (ks, ks), i))
        writer.close()


def Contrast(ref_video, ref_file):
    mp = {1 : 0.8, 2 : 1.2, 3 : 1.6}
    for i in range(1, 4):    
        dis_path = os.path.join(resultdir, 'Contrast', os.path.splitext(ref_file)[0] +  '_Contrast_' + str(i) + '.mp4')

        checkDir(os.path.join(resultdir, 'Contrast'))

        writer = skvideo.io.FFmpegWriter(dis_path)
        aug = iaa.GammaContrast((mp[i], mp[i]))
        for idx, frame in enumerate(ref_video):
            writer.writeFrame(aug.augment_image(frame))
        writer.close()
       
def Color(ref_video, ref_file):
    mp = {1 : 1.2, 2 : 1.6, 3 : 2.0}
    for i in range(1, 4):    
        dis_path = os.path.join(resultdir, 'Color', os.path.splitext(ref_file)[0] +  '_Color_' + str(i) + '.mp4')

        checkDir(os.path.join(resultdir, 'Color'))

        writer = skvideo.io.FFmpegWriter(dis_path)
        for idx, frame in enumerate(ref_video):
            writer.writeFrame(color(frame,mp[i]))
        writer.close()
        
def color(frame,factor):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv[:,:,1] = np.clip(hsv[:,:,1] * factor, 0, 255)
    new_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return new_frame


def fps(ref_video, file, ref_path):
    mp = {1 : 20, 2 : 15, 3 : 12}
    for i in range(1, 4):
        dis_path = os.path.join(resultdir, 'fps', os.path.splitext(file)[0] +  '_fps_' + str(i) + '.mp4')

        checkDir(os.path.join(resultdir, 'fps'))
            
        strCmd='/home/amd7302/YXX/ffmpeg/ffmpeg -i '+ ref_path + ' -r ' + str(mp[i] )+ ' ' + dis_path
        os.system(strCmd)
        
    


def checkDir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        print(f"make dir : {dir} success! ")

if __name__ == "__main__":

    for dir in [resultdir]:
        checkDir(dir)

    logger = getlogger()
    
    files = os.listdir(rootdir)

    for file in tqdm(files):
        if file.endswith('.mp4'):
            ref_path = os.path.join(rootdir, file)

            video = skvideo.io.vread(ref_path)
            h264(video, file, ref_path)
            h265(video, file, ref_path)
            GaussBlur(video, file)
            GaussNoise(video, file)
            Contrast(video, file)
            Color(video, file)
            fps(video, file, ref_path)

            
            
