import argparse
import logging
import os
import time
from datetime import datetime, timedelta
import gc
from typing import Collection
from tqdm import tqdm

import scipy.io as sio

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import open_clip
from scipy.stats import pearsonr, spearmanr, kendalltau
from timm.utils import AverageMeter

from config import get_config
from lr_scheduler import build_scheduler
from optimizer import build_optimizer

from models.build import build_model
from getVQA import *
from utils import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor
from models.NRCVQA import TextEncoder


bestSROCC = 0
bestPLCC = 0
bestRMSE = 1e9
bestKROCC = 0

finetune = False

totalTime = 0.0


def encode_text_prompts(prompts,model,device="cuda"):
        tokenizer = open_clip.get_tokenizer("RN50")
        text_tokens = tokenizer(prompts).to(device)
        with torch.no_grad():
            embedding = model.token_embedding(text_tokens)
            text_features = model.encode_text(text_tokens).float()
        return text_tokens, embedding, text_features


def check_nan(epoch_pred):
    for idx, val in enumerate(epoch_pred):
        if np.isnan(val):
            epoch_pred[idx] = 0


def dump_data(idx, pred, dmos, dataset):
    import scipy.io as sio
    path = f'./Data/True_CVQA/{str(finetune)}'
    if not os.path.exists(path):
        os.makedirs(path)
    sio.savemat(os.path.join(
        path, f'{dataset}-{idx}-{TIMESTAMP}.mat'), {'pred': pred, 'dmos': dmos})


def trainLoop(config, text_encoder, clip_visual, train_set, model, device, criterion, optimizer, logger, writer, epoch, lr_scheduler):
    model.train()
    optimizer.zero_grad()
    num_steps = len(train_set)

    epoch_pred, epoch_label = [], []
    totalLoss = 0

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    start = time.time()
    end = time.time()

    for idx, (video, video_lbp, video_dw, label) in enumerate(tqdm(train_set)):

        A_feats = []
        T_feats = []
        
        B, N, C, D, H, W = video.size()

        clip = torch.randint(10000000, (1,)) % N
        video = video[:, clip, :, :, :, :].squeeze(1)#[B, C, D, H, W]
        video_lbp = video_lbp[:, clip, :, :, :, :].squeeze(1)
        

        
        video = video.to(device)
        video_lbp = video_lbp.to(device)
        video_dw = video_dw.to(device)

        label = label.view(-1, 1).type(torch.float32).to(device)
         
        for clip in range(D):
            video_clip = video_dw[:, :, clip, :, :].squeeze(2)
            video_clip_t = video_lbp[:, :, clip, :, :].squeeze(2)
            with torch.no_grad():
                   img_t_feats =  clip_visual(video_clip_t)
                   img_feats =  clip_visual(video_clip)
                   
                   logits_T = img_t_feats @ text_T.t()
                   logits_T = logits_T.softmax(dim=-1)
                   logits_A = img_feats @ text_A.t()
                   logits_A = logits_A.softmax(dim=-1)
                   logits_C = img_feats @ text_C.t()
                   logits_C = logits_C.softmax(dim=-1)
                   logits_AC = torch.cat((logits_A,logits_C), dim=-1)
                   feats_A = logits_AC.cpu().numpy()
                   feats_T = logits_T.cpu().numpy()
                   A_feats.append(feats_A)
                   T_feats.append(feats_T)
        A_feats = np.concatenate(A_feats, axis=1) 
        T_feats = np.concatenate(T_feats, axis=1) 
        A_feats = torch.from_numpy(np.asarray(A_feats)).to(device)
        T_feats = torch.from_numpy(np.asarray(T_feats)).to(device)

        pred = model(video, A_feats, T_feats)

        if config.TRAIN.LOSS != 'mix':
            loss = criterion(pred, label)
        else:
            loss = config.TRAIN.lambda1 * \
                criterion[0](pred, label) + config.TRAIN.lambda2 * \
                criterion[1](pred, label)

        if config.TRAIN.ACCUMULATION_STEPS > 1:
            loss = loss / config.TRAIN.ACCUMULATION_STEPS
            loss.backward()
            if config.TRAIN.CLIP_GRAD:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.TRAIN.CLIP_GRAD)
            else:
                grad_norm = get_grad_norm(model.parameters())
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            if config.TRAIN.CLIP_GRAD:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.TRAIN.CLIP_GRAD)
            else:
                grad_norm = get_grad_norm(model.parameters())
            optimizer.step()
            lr_scheduler.step_update(epoch * num_steps + idx)

        totalLoss += loss.item()
        epoch_pred.extend(pred.detach().cpu().numpy().reshape(-1))
        epoch_label.extend(label.detach().cpu().numpy().reshape(-1))

        loss_meter.update(loss.item(), label.size(0))
        batch_time.update(time.time() - end)
        norm_meter.update(grad_norm)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')

    check_nan(epoch_pred)

    SROCC = spearmanr(epoch_pred, epoch_label)[0]
    PLCC = pearsonr(epoch_pred, epoch_label)[0]
    epoch_time = time.time() - start

    logger.info(
        f"epoch : {epoch}\n \t\ttrain SROCC : {SROCC:.4f}\n \t\tPLCC : {PLCC:.4f}\n \t\tepoch Loss : {totalLoss:.4f}\n \t\tepoch_time {epoch_time:.4f}\n")

    writer.add_scalar('Loss/Train', totalLoss, epoch)
    writer.add_scalar('SROCC/Train', SROCC, epoch)
    writer.add_scalar('PLCC/Train', PLCC, epoch)


def testLoop(config, text_encoder, clip_visual, test_set, model, device, criterion, logger, writer, epoch, checkpath, dataset, disNum=4):

    model.eval()

    epoch_pred, epoch_label = [], []

    testLoss = 0
    beginTime = time.time()
    with torch.no_grad():
        for (img, img_lbp, img_dw, label) in tqdm(test_set):

            A_feats = []
            T_feats = []
            
            B, N, C, D, H, W = img.size()
            clip = torch.randint(10000000, (1,)) % N
            img = img[:, clip, :, :, :, :].squeeze(1)#[B, C, D, H, W]
            img_lbp = img_lbp[:, clip, :, :, :, :].squeeze(1)
            
            
            img = img.to(device)
            img_lbp = img_lbp.to(device)
            img_dw = img_dw.to(device)

            label = label.view(-1, 1).to(device)
            
            
            for clip in range(D):
                video_clip_t = img_lbp[:, :, clip, :, :].squeeze(2)
                video_clip = img_dw[:, :, clip, :, :].squeeze(2)
                img_t_feats =  clip_visual(video_clip_t)
                img_feats =  clip_visual(video_clip)
                
                logits_T = img_t_feats @ text_T.t()
                logits_T = logits_T.softmax(dim=-1)
                logits_A = img_feats @ text_A.t()
                logits_A = logits_A.softmax(dim=-1)
                logits_C = img_feats @ text_C.t()
                logits_C = logits_C.softmax(dim=-1)
                logits_AC = torch.cat((logits_A,logits_C), dim=-1)
                feats_A = logits_AC.cpu().numpy()
                feats_T = logits_T.cpu().numpy()
                A_feats.append(feats_A)
                T_feats.append(feats_T)     
            A_feats = np.concatenate(A_feats, axis=1) 
            T_feats = np.concatenate(T_feats, axis=1) 
            A_feats = torch.from_numpy(np.asarray(A_feats)).to(device)
            T_feats = torch.from_numpy(np.asarray(T_feats)).to(device)
            
            pred = model(img, A_feats, T_feats)


            if config.TRAIN.LOSS != 'mix':
                testLoss += criterion(pred, label)
            else:
                testLoss += config.TRAIN.lambda1 * \
                    criterion[0](pred, label) + config.TRAIN.lambda2 * \
                    criterion[1](pred, label)


            epoch_label.extend(label.cpu().numpy().reshape(-1))
            epoch_pred.extend(pred.cpu().numpy().reshape(-1))

        global totalTime
        totalTime += time.time() - beginTime

        check_nan(epoch_pred)

        SROCC = spearmanr(epoch_label, epoch_pred)[0]
        PLCC = pearsonr(epoch_pred, epoch_label)[0]
        dump_data(epoch, epoch_pred, epoch_label, dataset)
        RMSE = np.sqrt(
            ((np.array(epoch_pred) - np.array(epoch_label)) ** 2).mean())
        KROCC = kendalltau(epoch_pred, epoch_label)[0]

        global bestSROCC, bestPLCC, bestRMSE, bestKROCC
        bestSROCC = max(bestSROCC, SROCC)
        bestPLCC = max(bestPLCC, PLCC)
        bestKROCC = max(bestKROCC, KROCC)
        bestRMSE = min(bestRMSE, RMSE)

        print(f"epoch : {epoch}\t Best SROCC : {bestSROCC:.4f} Best PLCC :{bestPLCC:.4f} SROCC : {SROCC:.4f}, PLCC : {PLCC:.4f}, Test Loss : {testLoss:.4f}")
        logger.info(f"epoch: {epoch}\t Best SROCC :{bestSROCC:.4f} \tBest PLCC :{bestPLCC:.4f}\tBest KROCC :{bestKROCC:.4f}\tBest RMSE :{bestRMSE:.4f} \n \t\ttest : SROCC: {SROCC:.4f}\n \t\tPLCC: {PLCC:.4f}\n \t\tKROCC: {KROCC:.4f}\n \t\tRMSE: {RMSE:.4f}\n \t\ttestLoss : {testLoss:.4f}\n")


        writer.add_scalar('Loss/Test', testLoss, epoch)
        writer.add_scalar('SROCC/Test', SROCC, epoch)
        writer.add_scalar('PLCC/Test', PLCC, epoch)
        writer.add_scalar('RMSE/Test', RMSE, epoch)

        if SROCC >= bestSROCC and SROCC >= config.TEST.BESTSROCC:
            bestSROCC = SROCC
            if not os.path.exists(os.path.join(checkpath, DAY)):
                print(f"checkpoints path : {os.path.join(checkpath, DAY)}")
                os.makedirs(os.path.join(checkpath, DAY))
            torch.save(model.state_dict(), os.path.join(
                checkpath, DAY, TIMESTAMP + '_' + str(bestSROCC) + 'model.pth'))  
            print(f"save checkpoints, Best SROCC is : {bestSROCC}")


def load_pre_trained(logger, config, path, prompts_texture, prompts_aesthetic, prompts_content, ctx_number):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    clip_model, _, _ = open_clip.create_model_and_transforms("RN50",pretrained="openai")
    clip_model = clip_model.to(device)
    
    text_encoder = TextEncoder(clip_model).to(device)
    
    text_tokens_T, embedding_T, text_T = encode_text_prompts(prompts_texture , clip_model, device=device)
    text_tokens_A, embedding_A, text_A = encode_text_prompts(prompts_aesthetic , clip_model, device=device)
    text_tokens_C, embedding_C, text_C = encode_text_prompts(prompts_content , clip_model, device=device)
    clip_visual = clip_model.visual.to(device)
    
    model = build_model(config.MODEL.TYPE,text_tokens_T,text_tokens_A,text_tokens_C,embedding_T,embedding_A,embedding_C,text_encoder,ctx_number)

    # new model dict
    model_dict = model.state_dict()
   # for k in model_dict.keys():
       # print(k)
    # load pre trained model
    if config.MODEL.TYPE != 'pre_train':
        pre_dict = torch.load(path, device)
      #  for k,v in pre_dict.items():
         #   if k in model_dict:
           #    print(k)
          #  else:
              # print("null:",k)
        
        pretrained_dict = {k: v for k,
                           v in pre_dict.items() if k in model_dict}
        logger.info(
            f"Model Type : {config.MODEL.TYPE}\t Length Of Pre trained model : {len(pretrained_dict)}")
        model_dict.update(pretrained_dict)
    else:
        pretrained_model = torch.load(path, device)#['model']
        #pretrained_name = pretrained_model.state_dict()
        #name1 = [name for name in pretrained_name if 'lbphead' in name]
        print(name1)
        if 'head.weight' in pretrained_model:
            pretrained_model.pop('head.weight')
        if 'head.bias' in pretrained_model:
            pretrained_model.pop('head.bias')
        # get the same weight
        # pretrained_dict = {k: v for k, v in pretrained_model.items() if k in model_dict}
        # TODO VB_SWIN
        pretrained_dict = {
            k: v for k, v in pretrained_model.items() if 'backbone.' + k in model_dict}
        # overwrite the new model dict
        model_dict.update(pretrained_dict)
        # update the dict to new model
        logger.info(
            f"Model Type : {config.MODEL.TYPE}\t Length Of Pre trained model : {len(pretrained_dict)}")
        print(f"length of pretrained dict : {len(pretrained_dict)}")

    model.load_state_dict(model_dict, strict=False)
    model.to(device)
    return model, text_encoder, clip_visual, device
    #return model, device


def load_model(logger, prompts_texture, prompts_aesthetic, prompts_content, ctx_number):
    logger.info("Just load model, WithOut Pre-Trained Weight")
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    clip_model, _, _ = open_clip.create_model_and_transforms("RN50",pretrained="openai")
    clip_model = clip_model.to(device)
    
    text_encoder = TextEncoder(clip_model).to(device)
    
    #tokenizer = open_clip.get_tokenizer("RN50")
    text_tokens_T, embedding_T, text_T = encode_text_prompts(prompts_texture , clip_model, device=device)
    text_tokens_A, embedding_A, text_A = encode_text_prompts(prompts_aesthetic , clip_model, device=device)
    text_tokens_C, embedding_C, text_C = encode_text_prompts(prompts_content , clip_model, device=device)
    clip_visual = clip_model.visual.to(device)
    
    model = build_model(config.MODEL.TYPE,text_tokens_T,text_tokens_A,text_tokens_C,embedding_T,embedding_A,embedding_C,text_encoder,ctx_number)
    model.to(device)
    return model, text_encoder, clip_visual, device



def main(config, idx):

    dataset = config.DATA.DATASET
    
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
    f"a excellent aesthetic photo",
    f"a good aesthetic photo",
    f"a fair aesthetic photo",
    f"a poor aesthetic photo",
    f"a bad aesthetic photo" ]

    
    
    ctx_number = 0

    epoch = config.TRAIN.EPOCHS
    batch_train = config.DATA.BATCH_SIZE
    batch_test = config.DATA.BATCH_TEST

    # loss function
    if config.TRAIN.LOSS == 'mse':
        criterion = nn.MSELoss()
    elif config.TRAIN.LOSS == 'plcc':
        criterion = PlccLoss()
    elif config.TRAIN.LOSS == 'mix':
        criterion = [nn.MSELoss(), PlccLoss()]

    img_size = config.DATA.IMG_SIZE

    transform_train, transform_test = None, None

    if dataset == 'Syn_CVQA':
        writer, check_path, train_set, test_set, logger = \
            getCSIQVQA(config, transform_train, transform_test, batch_train=batch_train,
                       batch_test=batch_test, train_percent=config.TRAIN.PERCENT, idx=idx)
    elif dataset == 'True_CVQA':
        transform_train, transform_test = transforms.CenterCrop((360, 640)), transforms.CenterCrop((360, 640))
        writer, check_path, train_set, test_set, logger = \
            getCartoonVQA(config, transform_train, transform_test, batch_train=batch_train,
            batch_test=batch_test, train_percent=config.TRAIN.PERCENT,idx=idx)

    prefix = os.path.abspath('.')
    pre_trained_path = os.path.join(prefix, config.TRAIN.PRE_TRAINED)

    if config.FINE_TUNE:
        model, text_encoder, clip_visual, device = load_pre_trained(logger, config, pre_trained_path, prompts_texture, prompts_aesthetic, prompts_content, ctx_number)
        global finetune
        finetune = True
    elif config.TRAIN.USEPRETRAINED:
       model, device = load_pre_trained(logger, config, pre_trained_path)
    else:
        model, text_encoder, clip_visual, device = load_model(logger, prompts_texture, prompts_aesthetic, prompts_content, ctx_number)

    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad) 

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    
    logger.info(f"number of params: {n_parameters}")

    optimizer = build_optimizer(config, model)

    lr_scheduler = build_scheduler(config, optimizer, len(train_set))

    logger.info(f"PRE TRAINED PATH : {config.TRAIN.PRE_TRAINED}")
    # print config
    logger.info(model)

    logger.info(config.dump())

    global bestSROCC
    global bestPLCC
    bestSROCC = 0
    bestPLCC = 0



    for i in range(epoch):
        trainLoop(config, text_encoder, clip_visual, train_set, model, device, criterion, optimizer,
                  logger, writer, epoch=i, lr_scheduler=lr_scheduler)
        testLoop(config, text_encoder, clip_visual, test_set, model, device, criterion, logger, writer,
                 epoch=i, checkpath=check_path, dataset=dataset, disNum=disNum)
        gc.collect()

    logger.info(
        f"Transformer. Total Cost Time = {totalTime}, epoch = {epoch}, epoch cost time = {totalTime / epoch} seconds")

    writer.close()
    print("Done!")


def plcc(pred, target):
    n = len(pred)
    if n != len(target):
        raise ValueError('input and target must have the same length')
    if n < 2:
        raise ValueError('input length must greater than 2')

    xmean = torch.mean(pred)
    ymean = torch.mean(target)
    xm = pred - xmean
    ym = target - ymean
    bias = 1e-8
    normxm = torch.norm(xm, p=2) + bias
    normym = torch.norm(ym, p=2) + bias

    r = torch.dot(xm/normxm, ym/normym)
    return r


class PlccLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # print(pred.shape, target.shape)
        val = 1.0 - plcc(pred.view(-1).float(), target.view(-1).float())
        return torch.log(val)
        # return 1.0 - plcc(pred.view(-1).float(), target.view(-1).float())


def parse_option():
    parser = argparse.ArgumentParser(
        'NRCVQA for fine-tuned True_Cartoon_Video training and evaluation script', add_help=False)

    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int,
                        help="batch size for single GPU")
    parser.add_argument('--batch-test', type=int, default=24,
                        help="batch test size for single GPU")
    parser.add_argument('--accumulation-steps', type=int,
                        help="gradient accumulation steps")
    parser.add_argument('--dataset', type=str,
                        default='True_CVQA', help='dataset')
    parser.add_argument('--model', type=str,
                        default='NRCVQA', help='Model Type')
    parser.add_argument('--frame', type=int, default='100',
                        help='Frame Per Video')
    parser.add_argument('--base_lr', type=float,
                        default='5e-5', help='Base Learning Rate')
    parser.add_argument('--loss', type=str, default='plcc',
                        help='Loss Function')
    parser.add_argument('--best', type=float, default='0.8',
                        help='Best SROCC For Save Checkpoints')
    parser.add_argument('--epoch', type=int,
                        default='100', help='Epoch Number')
    parser.add_argument('--warm_up_epochs', type=int,
                        default='5', help='Warm Up Epoch Number')
    parser.add_argument('--pre_trained_path', type=str, default='./checkpoints/Pre_Train_VQA/, help='pretrained weight path')

    parser.add_argument('--fine_tune', type=bool, help='Fine Tune Or Not')
    parser.add_argument('--five', type=bool,
                        help='FIVE Test Calc Mean-Std Result.')
    parser.add_argument('--idx', type=int, default=-1, help='Index')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


if __name__ == '__main__':

    _, config = parse_option()

    seed = config.SEED
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)

    # linear scale the learning rate according to total batch size, may not be optimal
    # linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE
    # linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE
    # linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE
    linear_scaled_lr = config.TRAIN.BASE_LR
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR
    linear_scaled_min_lr = config.TRAIN.MIN_LR

    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        print("USING ACCUMULATION_STEPS")
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    print("base lr :", linear_scaled_lr)
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    print("warmup lr :", linear_scaled_warmup_lr)
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    print("min lr :", linear_scaled_min_lr)
    config.freeze()

    if config.FIVE:
        for idx in range(5):
            main(config, idx)
    else:
        main(config, config.IDX)
