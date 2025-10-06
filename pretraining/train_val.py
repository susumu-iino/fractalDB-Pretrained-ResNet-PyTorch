# -*- coding: utf-8 -*-
"""
Created on Sat Aug 05 23:55:12 2018
@author: Kazushige Okayasu, Hirokatsu Kataoka
"""

import time
import torch
import tqdm

# Training
def train(args, model, device, train_loader, optimizer, criterion, epoch, mixup_fn=None):
    iteration = (epoch-1)*len(train_loader)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    
    # switch to train mode
    model.train()
    end = time.time()
    str_epoch = 'Epoch:{:03d}'.format(epoch)
    #for i, (data, target) in enumerate(train_loader):
    for i, (data, target) in enumerate(tqdm.tqdm(train_loader, desc=str_epoch, total=len(train_loader))):
        # measure data loading time
        data_time.update(time.time() - end)
        data, target = data.to(device), target.to(device)

        if mixup_fn is not None:
           target, target = mixup_fn(data, target)

        # compute output
        output = model(data)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1 = accuracy_top1(output, target)

        #acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), data.size(0))
        top1.update(acc1, data.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return losses.avg, top1.avg

# Validation
def validate(args, model, device, val_loader, criterion, iteration):

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for i, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            
            # compute output
            output = model(data)
            loss = criterion(output, target)
            
            # measure accuracy and record loss
            #acc1, acc5 = accuracy(output, target, topk=(1, 5))
            acc1 = accuracy_top1(output, target)
            losses.update(loss.item(), data.size(0))
            top1.update(acc1[0], data.size(0))
            #top5.update(acc5[0], data.size(0))
        '''
        print('Test: Loss ({loss.avg:.4f})\t'
              'Acc@1 ({top1.avg:.3f})\t'
              'Acc@5 ({top5.avg:.3f})'.format(
               loss=losses, top1=top1, top5=top5))
        '''
    return losses.avg, top1.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy_top1(output, target):
    with torch.no_grad():
        batch_size = target.size(0)
        _, pred = output.topk(1, 1, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred)).float()
        res = correct.mul_(100.0/batch_size).sum()
        return res

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res