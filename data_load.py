# -*- coding: utf-8 -*-

### basic modules
import numpy as np
import time, pickle, os, sys, json, PIL, tempfile, warnings, importlib, math, copy, shutil, setproctitle

### torch modules
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import torch.nn.functional as F
from torch import autograd

from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR, MultiStepLR

# data loader that returns indices of images
class MyDataset(Dataset):
    def __init__(self, trainset):
        self.set = trainset
        
    def __getitem__(self, index):
        data, target = self.set[index]        
        return data, target, index

    def __len__(self):
        return len(self.set)

def data_loaders(data, batch_size, test_batch_size, augmentation=False, normalization=False, drop_last=False, shuffle=True): 

    if not os.path.exists('data/'):
        os.makedirs('data/')

    if data == 'cifar10': 
        if augmentation and normalization:
            trainset_transforms = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.225, 0.225, 0.225))])
            trainset = datasets.CIFAR10(root='data/', train=True, download=True, transform=trainset_transforms)
            testset = datasets.CIFAR10(root='data/', train=False, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.225, 0.225, 0.225))]))
            
        elif augmentation and not(normalization):
            trainset_transforms = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()])
            trainset = datasets.CIFAR10(root='data/', train=True, download=True, transform=trainset_transforms)
            testset = datasets.CIFAR10(root='data/', train=False, download=True, transform=transforms.Compose([
                transforms.ToTensor()]))
            
        else:
            trainset = datasets.CIFAR10(root='data/', train=True, download=True, transform=transforms.ToTensor())
            testset = datasets.CIFAR10(root='data/', train=False, download=True, transform=transforms.ToTensor())


        
    if data == 'mnist':
        trainset = datasets.MNIST(root='data/', train=True, download=True, transform=transforms.ToTensor())
        testset = datasets.MNIST(root='data/', train=False, download=True, transform=transforms.ToTensor())
    
    if data == 'tinyimagenet':
        if augmentation and normalization:
            trainset_transforms = transforms.Compose([
                    transforms.RandomCrop(64, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
            trainset = datasets.ImageFolder(root='data/tiny-imagenet-200/train', transform=trainset_transforms)
            testset = datasets.ImageFolder(root='data/tiny-imagenet-200/val', transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]))
            
        elif not augmentation and normalization:
            trainset_transforms = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
            testset_transforms = transforms.Compose([
                    transforms.ToTensor()])
            trainset = datasets.ImageFolder(root='data/tiny-imagenet-200/train', transform=trainset_transforms)
            testset = datasets.ImageFolder(root='data/tiny-imagenet-200/val', transform=testset_transforms)
            
        elif augmentation and not normalization:
            trainset_transforms = transforms.Compose([
                    transforms.RandomCrop(64, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor()])
            testset_transforms = transforms.Compose([
                    transforms.ToTensor()])
            trainset = datasets.ImageFolder(root='data/tiny-imagenet-200/train', transform=trainset_transforms)
            testset = datasets.ImageFolder(root='data/tiny-imagenet-200/val', transform=testset_transforms)
            
        elif not augmentation and not normalization:
            trainset_transforms = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor()])
            testset_transforms = transforms.Compose([
                    transforms.ToTensor()])
            trainset = datasets.ImageFolder(root='data/tiny-imagenet-200/train', transform=trainset_transforms)
            testset = datasets.ImageFolder(root='data/tiny-imagenet-200/val', transform=testset_transforms)


    data_loader  = torch.utils.data.DataLoader(dataset=MyDataset(trainset),
                                              batch_size=batch_size,
                                              shuffle=shuffle, 
                                              drop_last = drop_last,
                                              num_workers=1, pin_memory=True)
    test_data_loader  = torch.utils.data.DataLoader(dataset=MyDataset(testset),
                                              batch_size=test_batch_size,
                                              shuffle=False,
                                              num_workers=1, pin_memory=True)
    
    return data_loader, test_data_loader
