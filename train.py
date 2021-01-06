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

image_size = [386, 386]
batch_size = 32
workers = 4
Epoches = 100
num_class = 5
learning_rate = 0.003

Data_path = '../data/cassava-leaf-disease-classification'
Model_path = './weights/'
if not os.path.exists(Model_path):
    os.makedirs(Model_path)

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', default=0, type=int,
                    help='node rank for distributed training')
args = parser.parse_args()

dist.init_process_group(backend='nccl')
device=torch.device('cuda:{}'.format(args.local_rank))
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

def train(model, train_loader, optimizer, pbar):
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
        batch_info = 'Time {:.3f}, Loss {:.4f}, Raw Acc ({:.2f} {:.2f})'.format(end_time - start_time, epoch_loss, epoch_raw_acc[0], epoch_raw_acc[1])
        pbar.update()
        pbar.set_postfix_str(batch_info)
        return epoch_loss

def validate(model, val_loader, pbar):
    loss_container.reset()
    raw_metric.reset()
    start_time = time.time()

    model.eval()
    with torch.no_grad():
        for i, (img, labels, _) in enumerate(val_loader):
            img, labels = img.to(device), labels.to(device)
            y_pred, y_arc = model(img)
            batch_loss = criterion.ce_forward(y_pred, labels)
            epoch_loss = loss_container(batch_loss.item())
            epoch_acc = raw_metric(y_pred, labels)
    end_time = time.time()
    batch_info = 'Time {:.3f}, Val Loss {:.4f}, Val Acc ({:.2f}, {:.2f})'.format(end_time-start_time, epoch_loss, epoch_acc[0], epoch_acc[1])
    pbar.set_postfix_str(batch_info)
    return epoch_acc[0]

def main(model_name):

    model = choose_net(name=model_name, num_classes=num_class, weight_path='github')
    model.to(device)
    torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[args.local_rank],output_device=args.local_rank,find_unused_parameters=True)
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

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                                  num_workers=workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, shuffle=False, drop_last=True)
        best_acc = 0.0
        for epoch in range(Epoches):
            pbar = tqdm(total=len(train_loader), unit='batches')
            pbar.set_description('Epoch {}/{}'.format(epoch + 1, Epoches))
            train_loss = train(model, train_loader, optimizer, pbar)
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(train_loss)
            else:
                scheduler.step(epoch)
            if epoch % 5 == 0:
                acc = validate(model, val_loader, pbar)
                if acc > best_acc:
                    torch.save(model.state_dict(), Model_path + '/' + f"{model_name}_best_fold{fold + 1}.pth")
            pbar.close()

if __name__ == '__main__':
    MODELS = ['efficientnet-b6', 'efficientnet-b0']
    for model_name in MODELS:
        main(model_name)







