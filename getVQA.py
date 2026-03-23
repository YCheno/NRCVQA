import logging
import os
import time
import torch

from datetime import datetime, timedelta
from torch.utils.tensorboard import SummaryWriter
from database.VQA.dataset_yuv import DisTypeVideoDataset
from database.VQA.true_dataset_yuv import True_DisTypeVideoDataset
from torch.utils.data import DataLoader
from torchvision import transforms


DAY = "{0:%Y-%m-%d}".format(datetime.now())
TIMESTAMP = "{0:%H-%M}".format(datetime.now())

writer = SummaryWriter()

def getlogger(path=None, level=logging.DEBUG):
    logger = logging.getLogger()
    logger.setLevel(level) 
    rq = time.strftime(TIMESTAMP)
    log_path = './Logs'
    log_path = os.path.join(log_path, path, DAY)
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


def getSynCVQA(cfg, transform_train, transofrm_test, batch_train=1, batch_test=1, train_percent=0.8, idx=-1):
    summary_dir = 'runs/Syn_CVQA/' + DAY
    writer = SummaryWriter(summary_dir)
    check_path = './checkpoints/' + 'Syn_CVQA/'
    logger = getlogger(path='Syn_CVQA')


    channel = 3
    size_x = 224
    size_y = 224
    stride_x = 224
    stride_y = 224

    subj_dataset = './database/VQA/Syn_CVQA/'
    video_path = '/mnt/wwn-0x5000cca0c3e1998a/Syn_CVQA/'
    batch = {'train' : batch_train, 'test' : batch_test}

    if cfg.IDX != -1:
        subj_dataset = './database/VQA/Syn_CVQA/Cartoon_LEAVE2_' + str(cfg.IDX) + '.json'
        if cfg.IDX == 100:
            subj_dataset = './database/VQA/Syn_CVQA/Cartoon_subj_score_TEST.json'
            video_dataset = {x:DisTypeVideoDataset(subj_dataset, video_path, x, channel, size_x, size_y, stride_x, stride_y, frameWant=cfg.VQA.FRAMEWANT, transform=transform_train) for x in ['test']}
            dataloaders = {x: DataLoader(video_dataset[x], batch_size=batch[x], shuffle=True, num_workers=cfg.DATA.NUM_WORKERS , drop_last=False) for x in ['test']}
            return writer, check_path, None, dataloaders['test'], logger

    logger.info(f"subj_dataset = {subj_dataset}")

    video_dataset = {x: DisTypeVideoDataset(subj_dataset, video_path, x, channel, size_x, size_y, stride_x, stride_y, frameWant=cfg.VQA.FRAMEWANT, transform=transform_train) for x in ['train', 'test']}
    dataloaders = {x: DataLoader(video_dataset[x], batch_size=batch[x], shuffle=True, num_workers=cfg.DATA.NUM_WORKERS , drop_last=False) for x in ['train', 'test']}
    return writer, check_path, dataloaders['train'], dataloaders['test'], logger


def getTrueCVQA(cfg, transform_train, transofrm_test, batch_train=1, batch_test=1, train_percent=0.75, idx=-1):
    summary_dir = 'runs/True_CVQA/' + DAY
    writer = SummaryWriter(summary_dir)
    check_path = './checkpoints/' + 'True_CVQA/'
    logger = getlogger(path='True_CVQA')


    channel = 3
    size_x = 224
    size_y = 224
    stride_x = 224
    stride_y = 224

    subj_dataset = './database/VQA/True_CVQA/'
    video_path = ' '
    #video_path = '/mnt/wwn-0x5000cca0c3e1998a/true_cartoon_150/20s'
    batch = {'train' : batch_train, 'test' : batch_test}

    if cfg.IDX != -1:
        subj_dataset = './database/VQA/True_CVQA/Cartoon_subj_score_' + str(cfg.IDX) + '.json'
        if cfg.IDX == 100:
            subj_dataset = './database/VQA/True_CVQA/True_Cartoon_subj_score_TEST.json'
            video_dataset = {x: True_DisTypeVideoDataset(subj_dataset, video_path, x, channel, size_x, size_y, stride_x, stride_y, frameWant=cfg.VQA.FRAMEWANT, transform=transform_train) for x in ['test']}
            dataloaders = {x: DataLoader(video_dataset[x], batch_size=batch[x], shuffle=False, num_workers=cfg.DATA.NUM_WORKERS , drop_last=False) for x in ['test']}
            return writer, check_path, None, dataloaders['test'], logger

    logger.info(f"subj_dataset = {subj_dataset}")

    video_dataset = {x: True_DisTypeVideoDataset(subj_dataset, video_path, x, channel, size_x, size_y, stride_x, stride_y, frameWant=cfg.VQA.FRAMEWANT, transform=transform_train) for x in ['train', 'test']}
    dataloaders = {x: DataLoader(video_dataset[x], batch_size=batch[x], shuffle=False, num_workers=cfg.DATA.NUM_WORKERS , drop_last=False) for x in ['train', 'test']}
    return writer, check_path, dataloaders['train'], dataloaders['test'], logger

