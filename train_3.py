#!usr/bin/python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import os
import sys
import pandas as pd
from models.net import choose_net
from sklearn.model_selection import KFold
import torch.distributed as dist
from dataset import MyData, FGVC7Data
import argparse
from loss import *
from utils.utils import get_transform, AverageMeter, TopKAccuracyMetric
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
from torch.utils.data.sampler import SubsetRandomSampler
from config import cfg
import logging
import numpy as np
import random



if not os.path.exists(cfg.MODEL.MODEL_PATH):
    os.makedirs(cfg.MODEL.MODEL_PATH)

def setup_logger(name, save_dir):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fh = logging.FileHandler(os.path.join(
            save_dir, "train_log_{}.txt".format(name)), mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# device = torch.device("cuda")
torch.backends.cudnn.benchmark = True

# df = pd.read_csv(os.path.join(Data_path, 'train.csv'))
loss_container = AverageMeter(name='loss')
raw_metric = TopKAccuracyMetric(topk=(1,2))
# General loss functions
loss_way = cfg.MODEL.LOSS_WAY
ce_weight = 1.0
arc_weight = 0
if loss_way == 'all':
    ce_weight = arc_weight = 0.5
elif loss_way == 'ce':
    ce_weight = 1.0
    arc_weight = 0
elif loss_way == 'arc' :
    ce_weight = 0
    arc_weight = 1.0
criterion = Criterion(weight_arcface=arc_weight, weight_ce=ce_weight)


def main(file_name, log):
    set_seed(cfg.SOLVER.SEED)
    config_file = './configs/' + file_name
    cfg.merge_from_file(config_file)
    # os.environ["CUDA_VISIBLE_DEVICES"] = cfg.MODEL.DEVICE_ID
    # device = torch.device("cuda")
    weight_path = cfg.MODEL.MODEL_PATH + cfg.MODEL.NAME + '.pth'
    model = choose_net(name=cfg.MODEL.NAME, num_classes=cfg.MODEL.CLASSES, weight_path=cfg.MODEL.WEIGHT_FROM)
    best_acc = 0.0
    log.info('Train : {}'.format(cfg.MODEL.NAME))
    if os.path.exists(weight_path):
        checkpoint = torch.load(weight_path)
        state_dict = checkpoint['state_dict']
        best_acc = checkpoint['best_acc']
        model.load_state_dict(state_dict)
        log.info('Network loaded from {}'.format(weight_path))

    # model.to(device)
    # model.cuda()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.SOLVER.BASE_LR, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.SOLVER.MAX_EPOCHS, eta_min=1e-6)
    train_dataset = FGVC7Data(root=cfg.DATASETS.ROOT_DIR, phase='train', transform=get_transform(cfg.INPUT.SIZE_TRAIN, 'train'))
    indices = range(len(train_dataset))
    split = int(cfg.DATASETS.SPLIT * len(train_dataset))
    train_indices = indices[split:]
    test_indices = indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(test_indices)
    train_loader = DataLoader(train_dataset, batch_size=cfg.DATASETS.BATCH_SIZE, sampler=train_sampler, num_workers=cfg.DATASETS.WORKERS,
                              pin_memory=True)
    val_loader = DataLoader(train_dataset, batch_size=cfg.DATASETS.BATCH_SIZE, sampler=valid_sampler, num_workers=cfg.DATASETS.WORKERS,
                                pin_memory=True)

    for epoch in range(cfg.SOLVER.MAX_EPOCHS):
        # pbar = tqdm(total=len(train_loader), unit='batches', ncols=150)  # unit 表示迭代速度的单位
        # pbar.set_description('Epoch {}/{}'.format(epoch + 1, cfg.SOLVER.MAX_EPOCHS))
        train(model, optimizer, epoch, train_loader, log)
        scheduler.step()
        if (epoch+1) % 5 == 0:
            acc = validate(model, val_loader, epoch, log)
            if acc > best_acc:
                if torch.cuda.device_count()>1:
                    torch.save({'best_acc':best_acc, 'state_dict':model.module.state_dict()}, weight_path)
                else:
                    torch.save({'best_acc':best_acc, 'state_dict':model.state_dict()}, weight_path)
        # pbar.close()

def train(model, optimizer, epoch, train_loader, log):
    loss_container.reset()
    raw_metric.reset()
    pbar = tqdm(enumerate(train_loader), total=int(len(train_loader.dataset)*(1-cfg.DATASETS.SPLIT)/cfg.DATASETS.BATCH_SIZE))
    pbar.set_description('Train Epoch {}/{}'.format(epoch + 1, cfg.SOLVER.MAX_EPOCHS))
    model.train()
    # for batch_idx, (img, label, _) in enumerate(train_loader):
    for batch_idx, (img, label, _) in pbar:
        img, label = img.cuda(), label.cuda()
        out = model(img)
        batch_loss = criterion(out, label, img)
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        with torch.no_grad():
            y_pred_raw = out[0]
            epoch_loss = loss_container(batch_loss.item())
            epoch_raw_acc = raw_metric(y_pred_raw, label)
        # pbar.update()
        pbar.set_postfix_str('Loss: {:.2f}  Train acc@1: {:.2f}'.format(epoch_loss, epoch_raw_acc[0]))
    # pbar.close()
    log.info('Train: {} \t Loss : {} \t Train Acc : {} '.format(epoch+1, epoch_loss, epoch_raw_acc[0]))

def validate(model,test_loader, epoch, log):
    loss_container.reset()
    raw_metric.reset()
    pbar = tqdm(enumerate(test_loader), total=int(len(test_loader.dataset)*cfg.DATASETS.SPLIT/cfg.DATASETS.BATCH_SIZE))
    pbar.set_description('Validation Epoch {}/{}'.format(epoch + 1, cfg.SOLVER.MAX_EPOCHS))
    model.eval()
    with torch.no_grad():
        for batch_idx, (img, label, _) in pbar:
            img, label = img.cuda(), label.cuda()
            y_pred, y_arc = model(img)
            batch_loss = criterion.ce_forward(y_pred, label)
            epoch_loss = loss_container(batch_loss.item())
            epoch_acc = raw_metric(y_pred, label)
            pbar.set_postfix_str('Loss: {:.2f}  Val acc@1: {:.2f}'.format(epoch_loss, epoch_acc[0]))
    # pbar.close()
    log.info('Validation: {} \t Loss : {} \t Validation Acc : {} '.format(epoch+1, epoch_loss, epoch_acc[0]))
    return epoch_acc[0]


if __name__ == '__main__':
    congfig_files = {'Resnest200.yaml', 'efficientnetb7.yaml', 'efficientnetb5.yaml'}
    for file_name in congfig_files:
        log = setup_logger(file_name, './log')
        main(file_name, log)



























