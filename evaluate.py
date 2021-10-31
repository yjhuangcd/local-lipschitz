### basic modules
import numpy as np
import time, pickle, os, sys, json, PIL, tempfile, warnings, importlib, math, copy, shutil, setproctitle
from datetime import datetime

### torch modules
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import torch.nn.functional as F

import data_load
import utils
import Local_bound as Local

if __name__ == "__main__":
    args = utils.argparser()
    print(datetime.now())
    print(args)
    _, test_loader = data_load.data_loaders(args.data, args.batch_size, args.test_batch_size, augmentation=args.augmentation, normalization=args.normalization, drop_last=args.drop_last, shuffle=args.shuffle)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    test_log = open("test.log", "w")
    
    model = utils.select_model(args.data, args.model, args.init)
    
    args.print = True
    u_list = None
        
    aa = torch.load(args.saved_model + "_best.pth")['state_dict']
    model = utils.select_model(args.data, args.model, args.init) 
    model.load_state_dict(aa)
    
    print('std testing ...')
    std_err = utils.evaluate(test_loader, model, args.epochs, test_log, args.verbose)
    print('verification testing ...')
    _, last_err, robust_losses_test, losses_test, errors_test = Local.evaluate(test_loader, model, args.epsilon, args.epochs, test_log, args.verbose, args, u_list, u_test=None, save_u=False)  
    print('pgd testing ...')
    pgd_err = utils.evaluate_pgd(test_loader, model, args)
    print('Best model evaluation:', std_err.item(), pgd_err.item(), last_err.item())
    