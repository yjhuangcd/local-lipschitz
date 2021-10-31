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

import data_load_storeu_multi as data_load
import utils_multi as utils
import Local_tiny_multi as Local

# distributed training
import apex
from apex import amp
from apex.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.utils.data.distributed

if __name__ == "__main__":
    args = utils.argparser()
    print(datetime.now())
    print(args)
    test_log = open("test.log", "w")
    
    ##### Distributed Training #####
    device_count = torch.cuda.device_count()
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend = 'nccl', init_method = 'env://')

    if 'NGC_ARRAY_INDEX' in os.environ: ## NGC cluster 
        node_rank  = int(os.environ['NGC_ARRAY_INDEX'])
        print(' -- init NGC: ', 
            'node_rank', node_rank, 'local_rank', args.local_rank)
    else: 
        node_rank = args.node_rank
        print('-- No multinode')

    global_rank = node_rank * device_count + args.local_rank
    args.world_size  = torch.distributed.get_world_size() #os.environ['world_size']
    print('node_rank', node_rank, 
        'device_count', device_count,
         'local_rank', args.local_rank,
        'global_rank', global_rank, 
        'world_size', args.world_size)    
    
    utils.seed_torch(args.seed)
    _, test_loader = data_load.data_loaders(args.data, args.batch_size, args.test_batch_size, augmentation=args.augmentation, normalization=args.normalization, drop_last=args.drop_last, shuffle=args.shuffle)
    
    model = utils.select_model(args.data, args.model, args.init)
    
    u_list = None
    u_test = None 
    aa = torch.load(args.saved_model + "_best.pth", map_location="cpu")
    model.load_state_dict(aa['state_dict'])
    
    model = DistributedDataParallel(model)
    print('std testing ...')
    # std_err = utils.evaluate(test_loader, model, args.epochs, test_log, args.verbose, args)
    print('pgd testing ...')
    # pgd_err = utils.evaluate_pgd(test_loader, model, args)
    print('verification testing ...')
    _, last_err, robust_losses_test, losses_test, errors_test, sparse = Local.evaluate(test_loader, model, args.epsilon, args.epochs, test_log, args.verbose, args, u_list, u_test, global_rank, save_u=False)  
    print('Best model evaluation:', std_err, pgd_err, last_err)
    