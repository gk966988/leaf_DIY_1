#!usr/bin/python
# -*- coding: utf-8 -*-
import torch
import os
import pandas as pd
from models.net import choose_net
from sklearn.model_selection import KFold
import torch.distributed as dist
from dataset import MyData
import argparse
from loss import *
from utils.utils import get_transform, AverageMeter, TopKAccuracyMetric
from torch.utils.data import DataLoader
import time
from tqdm import tqdm

image_size = [256, 256]
batch_size = 16
workers = 4
Epoches = 100
num_class = 5
learning_rate = 0.003

Data_path = '../data/cassava-leaf-disease-classification'
Model_path = './weights/'
if not os.path.exists(Model_path):
    os.makedirs(Model_path)

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
device = torch.device("cuda")


df = pd.read_csv(os.path.join(Data_path, 'train.csv'))

# General loss functions
loss_way = 'all'
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

loss_container = AverageMeter(name='loss')
raw_metric = TopKAccuracyMetric(topk=(1,2))

def train(model, train_loader, optimizer):
    loss_container.reset()
    raw_metric.reset()
    start_time = time.time()
    model.train()
    for i, (img, ladels, _) in enumerate(train_loader):
        img, ladels = img.to(device), ladels.to(device)
        out = model(img)
        batch_loss = criterion(out, ladels, img)
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        with torch.no_grad():
            y_pred_raw = out[0]
            epoch_loss = loss_container(batch_loss.item())
            epoch_raw_acc = raw_metric(y_pred_raw, ladels)
        end_time = time.time()
        print('Time {:.3f}, Loss {:.4f}, Raw Acc ({:.2f} {:.2f})'.format(end_time - start_time, epoch_loss, epoch_raw_acc[0], epoch_raw_acc[1]))

        return epoch_loss

def validate(model, val_loader):
    loss_container.reset()
    raw_metric.reset()
    start_time = time.time()
    a, b, c = 0.0, 0.0, 0.0
    model.eval()
    with torch.no_grad():
        for i, (img, labels, _) in enumerate(val_loader):
            img, labels = img.to(device), labels.to(device)
            y_pred, y_arc = model(img)
            batch_loss = criterion.ce_forward(y_pred, labels)
            epoch_loss = loss_container(batch_loss.item())
            a = epoch_loss
            epoch_acc = raw_metric(y_pred, labels)
            b = epoch_acc[0]
    end_time = time.time()
    print('Time {:.3f}, Val Loss {:.4f}, Val Acc ({:.2f})'.format(end_time - start_time, a, b))
    return b

def main(model_name):

    model = choose_net(name=model_name, num_classes=num_class, weight_path='github')
    model.to(device)
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, Epoches, eta_min=1e-6)
    # cheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=2)
    kf = KFold(n_splits=5, shuffle=True)
    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        print(f'fold:{fold+1}...','train_size: %d, val_size: %d' % (len(train_idx), len(val_idx)))
        df_train = df.values[train_idx]
        df_val = df.values[val_idx]
        train_dataset = MyData(root=Data_path, df=df_train, phase='train', transform=get_transform(image_size, 'train'))
        val_dataset = MyData(root=Data_path, df=df_val, phase='test', transform=get_transform(image_size, 'test'))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        best_acc = 0.0
        for epoch in range(Epoches):
            print('Train {} / {}'.format(epoch+1, Epoches))
            train_loss = train(model, train_loader, optimizer)
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(train_loss)
            else:
                scheduler.step(epoch)
            if epoch % 5 == 0:
                acc = validate(model, val_loader)
                if acc > best_acc:
                    if torch.cuda.device_count() > 1:
                        torch.save(model.module.state_dict(), Model_path + '/' + f"{model_name}_best_fold{fold + 1}.pth")
                    else:
                        torch.save(model.state_dict(),
                                   Model_path + '/' + f"{model_name}_best_fold{fold + 1}.pth")

if __name__ == '__main__':
    MODELS = ['efficientnet-b6', 'efficientnet-b0']
    for model_name in MODELS:
        main(model_name)







