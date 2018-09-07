from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms

from utils.split_validation import split_validation
from utils.misc import *

import matplotlib
matplotlib.use('Agg') # before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt

import numpy as np
import argparse
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--nepoch', default=350, type=int)
parser.add_argument('--seed', default=1124, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--depth', default=26, type=int)
parser.add_argument('--width', default=1, type=int)
parser.add_argument('--parallel', action='store_true')
parser.add_argument('--outf', default='demo')
parser.add_argument('--resume', default=None, help='path to checkpoint')
args = parser.parse_args()
my_makedir(args.outf)

print('==> Preparing data..')
NORM = ((0.4914, 0.4822, 0.4465), (0.2470, 0.2430, 0.2610))
trset = torchvision.datasets.CIFAR10(root='/home/yu/datasets/', 
    train=True, download=True, transform=transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(*NORM)]))
# TODO: We don't want to do data augmentation on the validation set!
trset, vaset = split_validation(trset, 5000, args.seed)
trloader = torch.utils.data.DataLoader(trset, batch_size=args.batch_size, shuffle=True, num_workers=2)
valoader = torch.utils.data.DataLoader(vaset, batch_size=args.batch_size, shuffle=False, num_workers=2)

teset = torchvision.datasets.CIFAR10(root='/home/yu/datasets/',
    train=False, download=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(*NORM)]))
teloader = torch.utils.data.DataLoader(teset, batch_size=args.batch_size, shuffle=False, num_workers=2)

print('==> Building model..')
from models.preact_resnet import ResNetCifar
net = ResNetCifar(args.depth, args.width)
if args.parallel:
    net = torch.nn.DataParallel(net)
net = net.cuda()
nparams = print_nparams(net)
cudnn.benchmark = True

best_err = 1        # best test error
start_epoch = 0     # start from epoch 0 or last checkpoint epoch
all_loss = []       # list of tuples (tr_err, tr_loss, va_err, va_loss, te_err, te_loss)

if args.resume is not None:
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(args.resume)
    net.load_state_dict(checkpoint['net'])
    best_err = checkpoint['va_err']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

def train():
    net.train()
    loss_sum = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        loss_sum += loss.data
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().data[0]
    return 1-correct/total, loss_sum

def test(dataloader):
    global best_err
    net.eval()
    loss_sum = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets, volatile=True)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        loss_sum += loss.data
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().data[0]
    return 1-correct/total, loss_sum

print('==> Running..')
for epoch in range(start_epoch, start_epoch+args.nepoch):
    if args.nepoch == 350:
        # for the default setting, use a fixed learning rate schedule
        # must restart the optimizer for each learning rate change
        if epoch == 150:
            optimizer = optim.SGD(net.parameters(), lr=args.lr / 10, momentum=0.9, weight_decay=5e-4)
        elif epoch == 250:
            optimizer = optim.SGD(net.parameters(), lr=args.lr / 100, momentum=0.9, weight_decay=5e-4)

    tr_err, tr_loss = train()
    va_err, va_loss = test(valoader)
    te_err, te_loss = test(teloader)
    all_loss.append((tr_err, tr_loss, va_err, va_loss, te_err, te_loss))

    print('%d/%d:\t' % (epoch, args.nepoch) + 
          '%.2f\t%.3f\t' % (tr_err*100, tr_loss) +'\t'+
          '%.2f\t%.3f\t' % (va_err*100, va_loss) +'\t'+
          '%.2f\t%.3f\t' % (te_err*100, te_loss))

    y_tr, _, y_va, _, y_te, _ = zip(*all_loss)
    plt.plot(np.asarray(y_te)*100, label='test error (%)')
    plt.plot(np.asarray(y_tr)*100, label='training error (%)')
    plt.legend()
    plt.savefig(         '%s/loss.pdf'          % (args.outf))
    plt.close()
    torch.save(all_loss, '%s/loss.pth'          % (args.outf))

    if epoch > 300:
        # if done for all epochs, I/O takes too much time
        if va_err < best_err:
            print('Saving..')
            state = {'tr_err': tr_err, 'va_err': va_err, 'te_err': te_err,
                    'tr_loss': tr_loss, 'va_loss': va_loss, 'te_loss': te_loss,
                    'epoch': epoch, 'nparams': nparams}
            torch.save(state, args.outf + '/ckpt_info.pth')
            state['net'] = net.state_dict()
            torch.save(state, args.outf + '/ckpt.pth')
            best_err = va_err
