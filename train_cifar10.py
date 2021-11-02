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
    args = utils.argparser(epochs=800,warmup=20,rampup=400,batch_size=256,epsilon_train=0.1551)
    print(datetime.now())
    print(args)
    print('saving file to {}'.format(args.prefix))
    setproctitle.setproctitle(args.prefix)
    dir, _ = os.path.split(args.prefix + '_train.log')
    if not os.path.exists(dir):
        os.makedirs(dir)
    train_log = open(args.prefix + "_train.log", "w")
    test_log = open(args.prefix + "_test.log", "w")
    train_loader, test_loader = data_load.data_loaders(args.data, args.batch_size, args.test_batch_size, augmentation=args.augmentation, normalization=args.normalization, drop_last=args.drop_last, shuffle=args.shuffle)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
        
    best_err = 1
    err = 1
    model = utils.select_model(args.data, args.model, args.init)
    
    # compute the feature size at each layer
    input_size = []
    depth = len(model)
    x = torch.randn(1,3,32,32).cuda()
    for i, layer in enumerate(model.children()):
        if i < depth-1:
            input_size.append(x.size()[1:])
            x = layer(x)
        
    # create u on cpu to store singular vector for every input at every layer
    u_train = []
    u_test = []
    
    for i in range(len(input_size)):
        print(i)
        if not model[i].__class__.__name__=='ReLU_x' and not model[i].__class__.__name__=='ClampGroupSort' and not model[i].__class__.__name__=='Flatten':
            u_train.append(torch.randn((len(train_loader.dataset), *(input_size[i])), pin_memory=True))
            u_test.append(torch.randn((len(test_loader.dataset), *(input_size[i])), pin_memory=True))
        else:
            u_train.append(None)
            u_test.append(None)
        
    if args.opt == 'adam': 
        opt = optim.Adam(model.parameters(), lr=args.lr)
    elif args.opt == 'sgd': 
        opt = optim.SGD(model.parameters(), lr=args.lr, 
                        momentum=args.momentum,
                        weight_decay=args.weight_decay) 
    print(opt)
    if args.lr_scheduler == 'step':
        lr_scheduler = optim.lr_scheduler.StepLR(opt, step_size=args.step_size, gamma=args.gamma)
    elif args.lr_scheduler =='multistep':
        lr_scheduler = MultiStepLR(opt, milestones=args.wd_list, gamma=args.gamma)
    elif (args.lr_scheduler == 'exp'):
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
          opt, lr_lambda=lambda step: utils.lr_exp(args.lr, args.end_lr, step, args.epochs, args.more))
    print(lr_scheduler)
    eps_schedule = np.linspace(args.starting_epsilon,
                               args.epsilon_train,                                
                               args.schedule_length)
    
    kappa_schedule = np.linspace(args.starting_kappa, 
                               args.kappa,                                
                               args.kappa_schedule_length)
    u_list = None
    for t in range(args.epochs): 
        # set up epsilon and kappa scheduling
        if t < args.warmup:
            epsilon = 0
            epsilon_next = 0
        elif args.warmup <= t < args.warmup+len(eps_schedule) and args.starting_epsilon is not None: 
            epsilon = float(eps_schedule[t-args.warmup])
            epsilon_next = float(eps_schedule[np.min((t+1-args.warmup, len(eps_schedule)-1))])
        else:
            epsilon = args.epsilon_train
            epsilon_next = args.epsilon_train
            
        if t < args.warmup:
            kappa = 1
            kappa_next = 1
        elif args.warmup <= t < args.warmup+len(kappa_schedule):
            kappa = float(kappa_schedule[t-args.warmup])
            kappa_next = float(kappa_schedule[np.min((t+1-args.warmup, len(kappa_schedule)-1))])
        else:
            kappa = args.kappa
            kappa_next = args.kappa
        print('%.f th epoch: epsilon: %.7f - %.7f, kappa: %.4f - %.4f, lr: %.7f'%(t,epsilon,epsilon_next,kappa,kappa_next,opt.state_dict()['param_groups'][0]['lr']))
            
        # begin training
        if t < args.warmup:
            utils.train(train_loader, model, opt, t, train_log, args.verbose)
            _ = utils.evaluate(test_loader, model, t, test_log, args.verbose)
        elif args.warmup <= t:
            st = time.time()
            u_list, u_train, robust_losses_train, robust_errors_train, losses_train, errors_train = Local.train(train_loader, model, opt, epsilon, kappa, t, train_log, args.verbose, args, u_list, u_train)
            print('Taken', time.time()-st, 's/epoch')
            
            u_test, err, robust_losses_test, losses_test, errors_test = Local.evaluate(test_loader, model, epsilon_next, t, test_log, args.verbose, args, u_list, u_test)
                       
        if args.lr_scheduler == 'step': 
            if max(t - (args.rampup + args.warmup - 1) + 1, 0):
                print("LR DECAY STEP")
            lr_scheduler.step(epoch=max(t - (args.rampup + args.warmup - 1) + 1, 0))
        elif args.lr_scheduler =='multistep' or args.lr_scheduler =='exp':
            print("LR DECAY STEP")
            lr_scheduler.step()      
        else:
            raise ValueError("Wrong LR scheduler")
            
        # Save the best model after epsilon has been the largest
        if t>=args.warmup+len(eps_schedule):    
            if err < best_err and args.save: 
                print('Best Error Found! %.3f'%err)
                best_err = err
                torch.save({
                    'state_dict' : model.state_dict(), 
                    'err' : best_err,
                    'epoch' : t
                    }, args.prefix + "_best.pth")

            torch.save({ 
                'state_dict': model.state_dict(),
                'err' : err,
                'epoch' : t
                }, args.prefix + "_checkpoint.pth")  

    args.print = True
    
    trained = torch.load(args.prefix + "_best.pth")['state_dict']
    model_eval = utils.select_model(args.data, args.model, args.init)
    model_eval.load_state_dict(trained)
    print('std testing ...')
    std_err = utils.evaluate(test_loader, model_eval, t, test_log, args.verbose)
    print('pgd testing ...')
    pgd_err = utils.evaluate_pgd(test_loader, model_eval, args)
    print('verification testing ...')
    u_test, last_err, robust_losses_test, losses_test, errors_test = Local.evaluate(test_loader, model_eval, args.epsilon, t, test_log, args.verbose, args, u_list, u_test)  
    print('Best model evaluation:', std_err.item(), pgd_err.item(), last_err.item())