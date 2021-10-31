### basic modules
import numpy as np
import time, pickle, os, sys, json, PIL, tempfile, warnings, importlib, math, copy, shutil, setproctitle

### torch modules
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import torch.nn.functional as F

from utils import *

DEBUG = False

def train(loader, model, opt, epsilon, kappa, epoch, log, verbose, args, u_list, u_train):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    errors = AverageMeter()
    robust_losses = AverageMeter()
    robust_errors = AverageMeter()
    
    model.train()

    end = time.time()
    net_local = net_Local_Lip(model, args)
        
    for i, (X,y,idx) in enumerate(loader):
        epsilon1 = epsilon+i/len(loader)*(args.epsilon-args.starting_epsilon)/args.schedule_length
        kappa1 = kappa+i/len(loader)*(args.kappa-args.starting_kappa)/args.schedule_length
        X,y = X.cuda(), y.cuda()
        data_time.update(time.time() - end)
 
        # clean loss
        out = model(X)
        ce = nn.CrossEntropyLoss()(out, y)
        err = (out.max(1)[1] != y).float().sum() / X.size(0)
        
        # extract singular vector for each data point
        u_train_data = []
        for ll in range(len(u_train)):
            if u_train[ll] is not None:
                u_train_data.append(u_train[ll][idx,:].cuda())
            else:
                u_train_data.append(None)

        # robust loss
        local_loss, local_err, u_list, u_train_idx = robust_loss(net_local, epsilon1, X, y, u_list, u_train_data, args.sniter, args.opt_iter, gloro=args.gloro)
        
        for ll in range(len(u_train)):
            if u_train_idx[ll] is not None:
                u_train[ll][idx,:] = u_train_idx[ll].clone().detach().cpu()
        
        if args.gloro:
            loss = local_loss
        else:
            loss = kappa1*ce + (1-kappa1)*local_loss
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        if 'relux' in args.model:
            # project all threshold in relu_x to be positive
            for ll in range(len(model)):
                if model[ll].__class__.__name__=='ReLU_x':
                    model[ll].threshold.data = torch.clamp(model[ll].threshold.data, 0.)

        # measure accuracy and record loss
        losses.update(ce.item(), X.size(0))
        errors.update(err.item(), X.size(0))
        robust_losses.update(local_loss.detach().item(), X.size(0))
        robust_errors.update(local_err, X.size(0))
        
        # measure elapsed time
        batch_time.update(time.time()-end)
        end = time.time()

        print(epoch, i, local_loss.detach().item(), 
                local_err.item(), ce.item(), err.item(), file=log)
              
        if verbose and (i==0 or i==len(loader)-1 or (i+1) % verbose == 0): 
            endline = '\n' if (i==0 or i==len(loader)-1 or (i+1) % verbose == 0) else '\r'
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Eps {eps:.3f}\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'L {loss.val:.4f} ({loss.avg:.4f})//'
                  'RL {rloss.val:.4f} ({rloss.avg:.4f})\t'
                  'E {errors.val:.3f} ({errors.avg:.3f})//'
                  'RE {rerrors.val:.3f} ({rerrors.avg:.3f})'.format(
                   epoch, i+1, len(loader), batch_time=batch_time, eps=epsilon1,
                   data_time=data_time, loss=losses, errors=errors, 
                   rloss = robust_losses, rerrors = robust_errors), end=endline)
        log.flush()

        if DEBUG and i ==10: 
            break
            
    print('')
    torch.cuda.empty_cache()
    return u_list, u_train, robust_losses.avg, robust_errors.avg, losses.avg, errors.avg


def evaluate(loader, model, epsilon, epoch, log, verbose, args, u_list, u_test, save_u=True):
    # save_u: save singular vector for each sample in the test set

    batch_time = AverageMeter()
    losses = AverageMeter()
    errors = AverageMeter()
    robust_losses = AverageMeter()
    robust_errors = AverageMeter()

    model.eval()

    end = time.time()
    torch.set_grad_enabled(False)
    net_local = net_Local_Lip(model, args)
    for i, (X,y,idx) in enumerate(loader):
        X,y = X.cuda(), y.cuda().long()
        if y.dim() == 2: 
            y = y.squeeze(1)

        # extract u_test
        if save_u:
            u_test_data = []
            for ll in range(len(u_test)):
                if u_test[ll] is not None:
                    u_test_data.append(u_test[ll][idx,:].cuda())
                else:
                    u_test_data.append(None)
        else: # do not need to use saved u in the real testing time
            u_test_data = []
                        
        local_loss, local_err, u_list, u_test_idx = robust_loss(net_local, epsilon, X, y, u_list, u_test_data, args.test_sniter, args.test_opt_iter, show=args.print, gloro=args.gloro)
            
        if save_u:
            for ll in range(len(u_test)):
                if u_test_idx[ll] is not None:
                    u_test[ll][idx,:] = u_test_idx[ll].clone().detach().cpu()
        
        out = model(X)
        ce = nn.CrossEntropyLoss()(out, y)
        err = (out.max(1)[1] != y).float().sum()  / X.size(0)

        # measure accuracy and record loss
        losses.update(ce.item(), X.size(0))
        errors.update(err, X.size(0))
        robust_losses.update(local_loss.detach().item(), X.size(0))
        robust_errors.update(local_err, X.size(0))

        # measure elapsed time
        batch_time.update(time.time()-end)
        end = time.time()

        print(epoch, i, local_loss.item(), local_err.item(), ce.item(), err.item(),
           file=log)
              
        if verbose and (i==0 or i==len(loader)-1 or (i+1) % verbose == 0) and not args.print: 
            endline = '\n' if (i==0 or i==len(loader)-1 or (i+1) % verbose == 0) else '\r'
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'L {loss.val:.4f} ({loss.avg:.4f})//'
                  'RL {rloss.val:.3f} ({rloss.avg:.3f})\t'
                  'E {error.val:.3f} ({error.avg:.3f})//'
                  'RE {rerrors.val:.3f} ({rerrors.avg:.3f})'.format(
                      i+1, len(loader), batch_time=batch_time, 
                      loss=losses, error=errors, rloss = robust_losses, 
                      rerrors = robust_errors), end=endline)
            
        log.flush()

        if DEBUG and i ==10: 
            break
    torch.set_grad_enabled(True)
    torch.cuda.empty_cache()
    print('')
    print(' * Robust error {rerror.avg:.3f}\t'
          'Error {error.avg:.3f}'
          .format(rerror=robust_errors, error=errors))
    
    return u_test, robust_errors.avg, robust_losses.avg, losses.avg, errors.avg

def robust_loss(net_local, epsilon, X, y, u_list=None, u_data=None, sniter=1, opt_iter=1, show=False, gloro=False):
    mu, mu_prev, r_prev, ibp_mu, ibp_mu_prev, ibp_r_prev, W, u_list, u_data = net_local(X, epsilon, u_list, u_data, sniter)
    onehot_y = one_hot(y)
    bcp_translation = BCP_translation(mu_prev, r_prev, ibp_mu_prev, ibp_r_prev, W, opt_iter, show)
    bcp_translation[bcp_translation!=bcp_translation]=0
    if gloro:                        
        worst_logit = gloro_translation(mu, bcp_translation, onehot_y, y)
    else:
        worst_logit = translation(mu, bcp_translation, onehot_y)
    robust_loss = nn.CrossEntropyLoss()(worst_logit, y)
    robust_err = (worst_logit.max(1)[1].data != y.cuda()).float().sum()/ X.size(0)
    
    return robust_loss, robust_err, u_list, u_data

class net_Local_Lip(nn.Module):
    def __init__(self, model, args):
        super(net_Local_Lip, self).__init__()
        self.model = model
        self.args = args
        
    def forward(self, x, epsilon=None, u_list=None, u_data=None, sniter=1):
        args = self.args
        eps = epsilon
        self.sniter = sniter
        
        mu = x.clone() 
        self.b =  x.size()[0] # batch_size
        r = eps * torch.ones(self.b, device=x.device) 
        ibp_r = eps * torch.ones(x.size(), device=x.device) 
            
        depth = len(list(self.model.children())) # number of main layers
        ibp_mu = mu
            
        # singular vectors for global Lipschitz bound
        if u_list is None:
            u_list = [None]*depth
        self.u_list = u_list
        # singular vectors for local Lipschitz bound at each data point
        self.u_data = u_data
        
        # create indicator matrices for all layers
        # Dn_mask: indicator for upper bound smaller than zero
        self.Dn_mask = [None]*depth
        # save input size at each layer
        self.in_size = [None]*depth
        self.in_size[0] = mu.size()
        
        for i, layer in enumerate(self.model.children()):
            if (i+1)==depth:
                # compute local Lipschitz bound until the second last layer
                specnorms = self.local_linear()
                r_tight = 1.0
                for col in range(len(specnorms)):
                    r_tight *= specnorms[col]
                r_tight *= eps
                mu_prev = mu
                r_prev = r_tight
                ibp_mu_prev = ibp_mu
                ibp_r_prev = ibp_r
                # Last layer of neural network
                mu = self._linear(mu, layer.weight, layer.bias)
                ibp_mu = self._linear(ibp_mu_prev, layer.weight.abs(), None)
                return mu, mu_prev, r_prev, ibp_mu, ibp_mu_prev, ibp_r_prev, layer.weight, u_list, self.u_data
            
            # Box constrained propagation
            if isinstance(layer, nn.Conv2d):
                mu, r, ibp_mu, ibp_r = self.bcp_conv2d(layer, mu, r, ibp_mu, ibp_r, i)
                if 'relux' in args.model:
                    self.bcp_mask(ibp_mu, ibp_r, i, self.model[i+1])                
            elif isinstance(layer, nn.Linear):
                mu, r, ibp_mu, ibp_r = self.bcp_linear(layer, mu, r, ibp_mu, ibp_r, i)
                if 'relux' in args.model:
                    self.bcp_mask(ibp_mu, ibp_r, i, self.model[i+1])
            elif layer.__class__.__name__=='ReLU_x':
                # save the mask for conv or linear layer before relu
                self.Dn_mask[i] = self.Dn_mask[i-1]
                mu, r, ibp_mu, ibp_r = self.bcp_relu(layer, mu, r, ibp_mu, ibp_r, i)
            elif layer.__class__.__name__=='ClampGroupSort':
                mu, r, ibp_mu, ibp_r = self.bcp_maxmin(layer, mu, r, ibp_mu, ibp_r, i)
                self.bcp_mask_maxmin(ibp_mu, ibp_r, i, layer)
            elif layer.__class__.__name__=='Flatten':
                # save the mask for conv or linear layer before flatten
                # need to reshape for the linear layer
                self.Dn_mask[i] = self.Dn_mask[i-1].view(mu.size()[0], -1)
                mu, r, ibp_mu, ibp_r = self.bcp_flatten(layer, mu, r, ibp_mu, ibp_r, i)
            # save input size at each layer
            self.in_size[i+1] = mu.size()
                
    def local_linear(self):
        EPS = 1e-12
        tol = 1e-3
        depth = len(list(self.model.children())) # number of main layers
        # save local Lipschitz bound at each layer, initialized as -1.0
        specnorms = [-1.0] * (depth-2)
        
        # compute local Lipschitz bound until the penultimate layer
        for i in range(depth-1):
            # pass for relu or flatten layer
            if not self.model[i].__class__.__name__=='ReLU_x' and not self.model[i].__class__.__name__=='ClampGroupSort' and not self.model[i].__class__.__name__=='Flatten':
                layer = self.model[i]
                W = layer.weight
                if len(self.u_data) > 0:
                    # singular vectors are different for different inputs in our local bound
                    u = self.u_data[i]
                else:
                    # random intialization
                    u = torch.randn((*self.in_size[i]), device=W.device)                    
                # power method to compute spectral norm of reduced matrices
                for itr in range(self.sniter):
                    u_tmp = u
                    # apply mask to the input u for both relu and maxmin
                    if i>0:
                        u = u * self.Dn_mask[i-1]
                    in_vector = u
                    if isinstance(layer, nn.Conv2d):
                        out = _conv2d(in_vector, W, None, stride=layer.stride, padding=layer.padding)
                    elif isinstance(layer, nn.Linear):
                        out = F.linear(in_vector, W, None)
                    # apply mask to the output for relu activations
                    if 'relu' in self.args.model:
                        out = out * self.Dn_mask[i]

                    if len(out.size()) > 2:
                        out_norm = out.view(out.size()[0], -1).norm(dim=1).view(out.size()[0],1,1,1)
                    else:
                        out_norm = out.norm(dim=1, keepdim=True)
                    v = out / (out_norm + EPS)
                    
                    in_vector = v
                    if isinstance(layer, nn.Conv2d):
                        out = _conv_trans2d(in_vector, W, stride=layer.stride, padding=layer.padding, output_padding = 0)
                        #  When the output size of conv_trans differs from the expected one.
                        if out.shape != u.shape:
                            out = _conv_trans2d(in_vector, W, stride=layer.stride, padding=layer.padding, output_padding = 1)
                    elif isinstance(layer, nn.Linear):
                        out = F.linear(in_vector, W.t(), None)
                    # apply mask for both relu and maxmin
                    if i>0:
                        out = out * self.Dn_mask[i-1]

                    if len(out.size()) > 2:
                        out_norm = out.view(out.size()[0], -1).norm(dim=1).view(out.size()[0],1,1,1)
                    else:   
                        out_norm = out.norm(dim=1, keepdim=True)                        
                    u = out / (out_norm + EPS)

                    if len(u.size())>2:
                        diffnorm = (u-u_tmp).view(u.size()[0], -1).norm(dim=1)
                    else:
                        diffnorm = (u-u_tmp).norm(dim=1)

                    if diffnorm.max()<tol or (itr+1)==self.sniter:
                        # save singular vector u
                        if len(self.u_data) > 0:
                            self.u_data[i] = u
                        break
                        
                # compute spectral norm using singular vector u       
                # apply mask to the input u
                if i>0:
                    u = u * self.Dn_mask[i-1]
                in_vector = u
                if isinstance(layer, nn.Conv2d):
                    out = _conv2d(in_vector, W, None, stride=layer.stride, padding=layer.padding)
                elif isinstance(layer, nn.Linear):
                    out = F.linear(in_vector, W, None)
                # apply mask to the output for relu activations
                if 'relu' in self.args.model:
                    out = out * self.Dn_mask[i]

                # sigma of size equals to the batch size, because for each sample we have different sigma
                sigma = (v*out).view(v.size()[0],-1).sum(1)
                specnorms[i] = sigma
        
        # remove -1.0 elements in specnorms (correspond to relu or flatten layer)        
        specnorms = [x for x in specnorms if (torch.is_tensor(x)==True or (torch.is_tensor(x)==False and x!=-1.0))]       
                
        return specnorms
                
                
    def bcp_mask(self, ibp_mu, ibp_r, i, relux):
        D1_mask = torch.zeros_like(ibp_mu) 
        D0_mask = torch.zeros_like(ibp_mu)
        Dn_mask = torch.ones_like(ibp_mu)
        ibp_lb = ibp_mu - ibp_r
        ibp_ub = ibp_mu + ibp_r
        
        # D1_msak: indicator for lower bound larger than relu_x threshold
        D1_mask[ibp_lb > relux.threshold] = 1.0
        # D0_msak: indicator for upper bound smaller than zero
        D0_mask[ibp_ub <= 0] = 1.0
        # Dn_msak: indicator for varying outputs
        Dn_mask -= (D1_mask + D0_mask)
        self.Dn_mask[i] = Dn_mask
        return
        
    def bcp_mask_maxmin(self, ibp_mu, ibp_r, i, layer):
        ibp_lb = ibp_mu - ibp_r
        ibp_ub = ibp_mu + ibp_r
        Dn_mask = torch.ones_like(ibp_mu)
        
        # first half max channels
        maxlb, minlb = ibp_lb.split(ibp_lb.size(1) // 2, 1)
        D1_mask = torch.zeros_like(maxlb) 
        D1_mask[maxlb >= layer.max] = 1.0
        
        # second half min channels
        maxub, minub = ibp_ub.split(ibp_ub.size(1) // 2, 1)
        D0_mask = torch.zeros_like(minlb)
        D0_mask[minub <= layer.min] = 1.0
        
        DD_mask = torch.cat([D1_mask, D0_mask], dim=1)
        Dn_mask -= DD_mask
        self.Dn_mask[i] = Dn_mask
        
        return

    def bcp_conv2d(self, layer, mu, r, ibp_mu, ibp_r, i):
        ibp_mu = self._conv2d(ibp_mu, layer.weight, layer.bias, layer.stride, padding=layer.padding)
        ibp_r = self._conv2d(ibp_r, torch.abs(layer.weight), None, layer.stride, padding=layer.padding)
        ibp_ub = (ibp_mu+ibp_r)
        ibp_lb = (ibp_mu-ibp_r) 
        
        mu_after = self._conv2d(mu, layer.weight, layer.bias, layer.stride, layer.padding)

        u = self.u_list[i]
        p, u = power_iteration_conv_evl(mu, layer, self.sniter, u) 
        self.u_list[i] = u.data                

        W = layer.weight

        ibp_mu1 = mu_after  
        ibp_p1 = W.view(W.size()[0],-1).norm(2,dim=-1)
        ibp_r1 = r.view(-1,1,1,1)*torch.ones_like(mu_after)
        ibp_r1 = ibp_r1*ibp_p1.view(1,-1,1,1)
        ibp_ub1 = ibp_mu1+ibp_r1
        ibp_lb1 = ibp_mu1-ibp_r1
        ibp_ub = torch.min(ibp_ub, ibp_ub1)
        ibp_lb = torch.max(ibp_lb, ibp_lb1)
        ibp_mu = (ibp_ub+ibp_lb)/2
        ibp_r = (ibp_ub-ibp_lb)/2  

        r = r*p 
        mu = mu_after
                
        return mu, r, ibp_mu, ibp_r
    
    def bcp_linear(self, layer, mu, r, ibp_mu, ibp_r, i):
        
        ibp_mu = self._linear(ibp_mu, layer.weight, layer.bias)
        ibp_r = self._linear(ibp_r, torch.abs(layer.weight),None)
        ibp_ub = (ibp_mu+ibp_r)
        ibp_lb = (ibp_mu-ibp_r)
        
        mu_after = self._linear(mu, layer.weight, layer.bias)
        W = layer.weight ### h(-1),h(-2)

        u = self.u_list[i]
        p, u = power_iteration_evl(W, self.sniter, u)
        self.u_list[i] = u.data

        ibp_mu1 = mu_after ### b, h(-1)
        ibp_r1 = r.view(self.b,1)*W.norm(2,dim=-1).view(1,-1).repeat(self.b,1)

        ibp_ub1 = ibp_mu1+ibp_r1
        ibp_lb1 = ibp_mu1-ibp_r1

        ibp_ub = torch.min(ibp_ub, ibp_ub1)
        ibp_lb = torch.max(ibp_lb, ibp_lb1)
        ibp_mu = (ibp_ub+ibp_lb)/2
        ibp_r = (ibp_ub-ibp_lb)/2

        r = r*p 
        mu = mu_after
                
        return mu, r, ibp_mu, ibp_r
    
    def bcp_relu(self, layer, mu, r, ibp_mu, ibp_r, i):  
        
        threshold = layer.threshold
        
        ibp_ub = ibp_mu+ibp_r
        ibp_lb = ibp_mu-ibp_r        
        ibp_ub = self._relu_x(ibp_ub, threshold)
        ibp_lb = self._relu_x(ibp_lb, threshold)
        ibp_mu = (ibp_ub+ibp_lb)/2
        ibp_r = (ibp_ub-ibp_lb)/2

        mu_size = []
        for j in range(len(mu.size())):
            if j<1:
                continue
            mu_size.append(1)        

        ibp_ub1 = self._relu_x(mu+r.view(-1,*mu_size), threshold)
        ibp_lb1 = self._relu_x(mu-r.view(-1,*mu_size), threshold)
        ibp_ub = torch.min(ibp_ub, ibp_ub1)
        ibp_lb = torch.max(ibp_lb, ibp_lb1)
        ibp_mu = (ibp_ub+ibp_lb)/2
        ibp_r = (ibp_ub-ibp_lb)/2
        
        mu = self._relu_x(mu, threshold)        
        r = r
        self.u_list[i] = 1
        return mu, r, ibp_mu, ibp_r

    def bcp_maxmin(self, layer, mu, r, ibp_mu, ibp_r, i):  
        
        maxthres = layer.max
        minthres = layer.min
                
        ibp_ub_in = ibp_mu+ibp_r
        ibp_lb_in = ibp_mu-ibp_r      
        
        mu_size = []
        for j in range(len(mu.size())):
            if j<1:
                continue
            mu_size.append(1)        
            
        ball_ub_in = mu+r.view(-1,*mu_size)
        ball_lb_in = mu-r.view(-1,*mu_size)
        
        ibp_lb_out = self._maxmin(ibp_lb_in, maxthres, minthres)
        ibp_ub_out = self._maxmin(ibp_ub_in, maxthres, minthres)
        
        ball_lb_out = self._maxmin(ball_lb_in, maxthres, minthres)
        ball_ub_out = self._maxmin(ball_ub_in, maxthres, minthres)

        ub = torch.min(ibp_ub_out, ball_ub_out)
        lb = torch.max(ibp_lb_out, ball_lb_out)
        ibp_mu = (ub+lb)/2
        ibp_r = (ub-lb)/2
            
        mu = self._maxmin(mu, maxthres, minthres)        
        r = r
        self.u_list[i] = 1
        return mu, r, ibp_mu, ibp_r
    
    def bcp_flatten(self, layer, mu, r, ibp_mu, ibp_r, i):
        ibp_mu = self._flatten(ibp_mu)
        ibp_r = self._flatten(ibp_r)
        mu = self._flatten(mu)
        r = r
        self.u_list[i] = 1
        return mu, r, ibp_mu, ibp_r
    
    def _conv2d(self,x,w,b,stride=1,padding=0):
        return F.conv2d(x, w,bias=b, stride=stride, padding=padding)
    
    def _linear(self,x,w,b):
        return F.linear(x,w,b)
    
    def _relu(self,x):
        return F.relu(x)
    
    def _relu_x(self,x, threshold):
        return torch.max(torch.zeros_like(x), torch.min(x, threshold))

    def _maxmin(self, x, maxthres, minthres):
        a, b = x.split(x.size(1) // 2, 1)
        a, b = torch.min(torch.max(a, b), maxthres), torch.max(torch.min(a, b), minthres)
        return torch.cat([a, b], dim=1)
    
    def _flatten(self,x):
        return x.view(x.size()[0],-1)


# compuate the worst-case logit using BCP method
def BCP_translation(mu_prev, r_prev, ibp_mu_prev, ibp_r_prev, W, opt_iter, show):
    EPS = 1e-12
    
    ## W: c, h
    c, h = W.size()
    b = mu_prev.size()[0]

    all_one = torch.ones((1,c,c,1), device=W.device)
    eye = torch.eye(c,c, device=W.device).view(1,c,c,1)
    diag_zero = ((all_one-eye)==1).type(torch.cuda.FloatTensor).repeat(1,1,1,1) # b,c,c,1
    
    wi_wj = (W.view(c,1,-1).repeat(1,c,1) -W.repeat(c,1,1)).unsqueeze(0) # 1,10(i),10(j),h
    wi_wj_rep = wi_wj.repeat(b,1,1,1)

    # Use EPS to avoid an error in gradient computation.
    w_norm = (wi_wj+EPS).norm(2,dim=-1,keepdim=True) # 1,10,10,1

    r_rep = r_prev.view(b,1,1,1) # 1,1,1,1

    wi_wj1 =  r_rep*F.normalize(wi_wj, p=2, dim=-1) # b,10,10,1

    norm = diag_zero*(r_rep+EPS)+EPS

    mu_rep = mu_prev.view(b,1,1,h) # b,1,1,h                    
    i_mu_rep = ibp_mu_prev.view(b,1,1,h) # b,1,1,h
    i_r_rep = ibp_r_prev.view(b,1,1,h) # b,1,1,h

    ### mu as a zero point
    ub = i_mu_rep + i_r_rep - mu_rep # b,1,1,h
    lb = i_mu_rep - i_r_rep - mu_rep # b,1,1,h
    
    if ub.min()<0 or lb.max()>0:
        ub[ub<0]=0        
        lb[lb>0]=0

    iteration = 0

    while (iteration == 0 or ((diag_zero*wi_wj1<lb)+(diag_zero*wi_wj1>ub)).sum()>0 and iteration < opt_iter):
        iteration +=1
        before_clipped = diag_zero*wi_wj1*(wi_wj1<=lb).type(torch.float) + diag_zero*wi_wj1*(wi_wj1>=ub).type(torch.float)
        after_clipped = diag_zero*lb*(wi_wj1<=lb).type(torch.float) + diag_zero*ub*(wi_wj1>=ub).type(torch.float)

        original_norm = (diag_zero*(before_clipped+EPS)).norm(2,dim=-1,keepdim=True)
        clipped_norm = (diag_zero*(after_clipped+EPS)).norm(2,dim=-1,keepdim=True)

        r_before2 = torch.clamp(norm.abs()**2-original_norm.abs()**2, min=0)
        r_after2 = torch.clamp(norm.abs()**2-clipped_norm.abs()**2, min=0)

        r_before = (r_before2+EPS+eye).sqrt()
        r_after = (r_after2+EPS+eye).sqrt()

        ratio = r_after/(r_before+EPS)

        in_sample = ((wi_wj1>lb)*(wi_wj1<ub)).type(torch.float) 
        enlarged_part = ratio*diag_zero*wi_wj1*in_sample 

        new_wi_wj = (after_clipped + enlarged_part)
        inner_prod  = (wi_wj_rep*new_wi_wj).sum(-1)
        wi_wj1 = new_wi_wj

    return inner_prod

def translation(mu, translation, onehot_y):
    b, c = onehot_y.size()
    # mu : b, 10
    # translation : b, 10, 10
    # onehot_y: b, 10
    
    delta = translation.bmm(onehot_y.unsqueeze(2)).view(b,c) 
    
    # b,10,10 x b,10,1 -> b,10,1 -> b,10
    # delta: b,10
    translated_logit = mu+((delta)*(1-onehot_y))
    return translated_logit

def gloro_translation(mu, translation, onehot_y, y):
    b, c = onehot_y.size()
    # mu : b, 10
    # translation : b, 10, 10
    # onehot_y: b, 10
    
    delta = translation.bmm(onehot_y.unsqueeze(2)).view(b,c) 
    
    # b,10,10 x b,10,1 -> b,10,1 -> b,10
    # delta: b,10
    translated_logit = mu+((delta)*(1-onehot_y))
    j = torch.arange(translated_logit.size(0)).long()
    translated_logit[j,y] -= torch.ones_like(y)*float('inf')
    
    # gloro
    maxo, _ = torch.max(translated_logit, dim=1, keepdim=True)   
    logits_verify = torch.cat((mu, maxo), dim=1)
    
    return logits_verify

def _conv2d(x,w,b,stride=1,padding=0):
    return F.conv2d(x, w, bias=b, stride=stride, padding=padding)


def _conv_trans2d(x,w,stride=1, padding=0, output_padding = 0):
    return F.conv_transpose2d(x, w,stride=stride, padding=padding, output_padding=output_padding)


def power_iteration_evl(A, num_simulations, u=None):
    if u is None:
        u = torch.randn((A.size()[0],1), device=A.device)
        
    B = A.t()
    for i in range(num_simulations):
        u1 = B.mm(u)
        u1_norm = u1.norm(2)
        v = u1 / u1_norm
        u_tmp = u

        v1 = A.mm(v)
        v1_norm = v1.norm(2)
        u = v1 / v1_norm

        if (u-u_tmp).norm(2)<1e-4 or (i+1)==num_simulations:
            break
        
    out = u.t().mm(A).mm(v)[0][0]

    return out, u


def power_iteration_conv_evl(mu, layer, num_simulations, u=None):
    EPS = 1e-12
    output_padding = 0
    if u is None:
        u = torch.randn((1,*mu.size()[1:]), device=mu.device)
        
    W = layer.weight
    if layer.bias is not None:
        b=torch.zeros_like(layer.bias)
    else:
        b = None
        
    for i in range(num_simulations):
        u1 = _conv2d(u, W, b, stride=layer.stride, padding=layer.padding)
        u1_norm = u1.norm(2)
        v = u1 / (u1_norm+EPS)
        u_tmp = u

        v1 = _conv_trans2d(v, W, stride=layer.stride, padding=layer.padding, output_padding = output_padding)
        #  When the output size of conv_trans differs from the expected one.
        if v1.shape != u.shape:
            output_padding = 1
            v1 = _conv_trans2d(v, W, stride=layer.stride, padding=layer.padding, output_padding = output_padding)            
        v1_norm = v1.norm(2)
        u = v1 / (v1_norm+EPS)

        if (u-u_tmp).norm(2)<1e-4 or (i+1)==num_simulations:
            break
        
    out = (v*(_conv2d(u, W, b, stride=layer.stride, padding=layer.padding))).view(v.size()[0],-1).sum(1)[0]
    return out, u
