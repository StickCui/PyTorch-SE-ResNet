# encoding:utf-8
# Modify from https://github.com/pytorch/examples/blob/master/imagenet/main.py
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

import model as modelZoo
from utils import ImageData
import utils

import os
import shutil
# import numpy as np
import argparse

model_names = ['se_resnet50', 'se_resnet101', 'se_resnet152', 'se_resnext50', 'se_resnext101', 'se_resnext152']

parser = argparse.ArgumentParser(description='PyTorch SE-ResNet Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='se_resnet101',
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: se_resnet101)')
parser.add_argument('--trainroot', required=True, help='path to train dataset (images list file)')
parser.add_argument('--valroot', required=True, help='path to val dataset (images list file)')
parser.add_argument('--lr', default=0.01, type=float,
                    help='learning rate for training')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', type=int,
                    default=64, help='input batch size')
parser.add_argument('--optim', type=str, default='SGD',
                    help='optim for training, Adam / SGD (default)')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='momentum for SGD')
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    help='weight_decay for SGD / Adam')
parser.add_argument('--workers', type=int,
                    default=4, help='workers for reading datasets')
parser.add_argument('--gpu', type=str, default='0',
                    help='ID of GPUs to use, eg. 1,3')
parser.add_argument('--model_path', type=str, default='weights',
                    help='model file to save')
parser.add_argument('--resume_path', type=str, default=None,
                    help='model file to resume to train')
parser.add_argument('--num_classes', type=int, default=365,
                    help='model file to resume to train')
parser.add_argument('--displayInterval', type=int,
                    default=200, help='Interval to be displayed')
args = parser.parse_args()
print(args)

if args.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    gpus = list(range(len(args.gpu.split(','))))
else:
    gpus = [0]  # [1,2]

lr_opt = lambda lr, epoch: lr * (0.1 ** (float(epoch) / 20)) # lr changes with epoch

if args.arch not in model_names:
    raise NotImplementedError('Other optimizer is not implemented')
else:
    Net = getattr(modelZoo, args.arch)
    model = Net(num_classes=args.num_classes)

model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

cudnn.benchmark = True

if args.resume_path is not None:
    pretrained_model = torch.load(args.resume_path)
    model.load_state_dict(pretrained_model['state_dict'])
    best_prec1 = pretrained_model['best_prec1']
    print('Load resume model done.')
else:
    best_prec1 = 0
print('Best top-1: {:.4f}'.format(best_prec1))

# Data
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_loader = torch.utils.data.DataLoader(
    ImageData(args.trainroot, transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=args.batch_size, shuffle=True,
    num_workers=args.workers, pin_memory=True)

val_loader = torch.utils.data.DataLoader(
    ImageData(args.valroot, transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True)

# define loss function (criterion) and pptimizer
criterion = nn.CrossEntropyLoss().cuda()

if args.optim == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)
elif args.optim == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
else:
    raise NotImplementedError('Other optimizer is not implemented')

def train(train_loader, model, criterion, optimizer, epoch, lr_cur):
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top3 = utils.AverageMeter()

    # switch to train mode
    model.train()

    for i, (input, target) in enumerate(train_loader):

        target = target.cuda(gpus[0], async=True)
        # print target
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure utils.accuracy and record loss
        prec1, prec3 = utils.accuracy(output.data, target, topk=(1, 3))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top3.update(prec3[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % args.displayInterval == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Lr: {lr:.4e}\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@3 {top3.val:.3f} ({top3.avg:.3f})'.format(
                      epoch, i, len(train_loader), lr=lr_cur, loss=losses, top1=top1, top3=top3))


def validate(val_loader, model, criterion):
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top3 = utils.AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(gpus[0], async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure utils.accuracy and record loss
        prec1, prec3 = utils.accuracy(output.data, target, topk=(1, 3))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top3.update(prec3[0], input.size(0))

        if (i+1) % args.displayInterval == 0:
            print('Test: [{0}/{1}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@3 {top3.val:.3f} ({top3.avg:.3f})'.format(
                      i, len(val_loader), loss=losses, top1=top1, top3=top3))

    print(' * Prec@1 {top1.avg:.3f} Prec@3 {top3.avg:.3f}'
          .format(top1=top1, top3=top3))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint'):
    torch.save(state, os.path.join(args.model_path, filename + '_latest.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(args.model_path, filename + '_latest.pth.tar'),
                        os.path.join(args.model_path, filename + '_best.pth.tar'))

for epoch in range(args.start_epoch, args.epochs):
    lr_cur = lr_opt(args.lr, epoch)  # speed change
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_cur

    # train for one epoch
    train(train_loader, model, criterion, optimizer, epoch, lr_cur)

    # evaluate on validation set
    prec1 = validate(val_loader, model, criterion)

    # remember best prec1 and save checkpoint
    is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)
    save_checkpoint({
        'epoch': epoch + 1,
        'arch': args.arch,
        'state_dict': model.state_dict(),
        'best_prec1': best_prec1,
    }, is_best, args.arch)
