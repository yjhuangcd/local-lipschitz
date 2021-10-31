#!/bin/bash

###### Training ######

# train Tinyimagenet-ReLU on 4 GPUs, Local bound + BCP loss
python -m torch.distributed.launch --nproc_per_node=4 train_tiny_multi.py --warmup 0 --model tinyimagenet_relux --batch_size 32 --rampup 125 --sniter 1 --init 1.0 --epochs 250 --sparse 0.01 --indices 0,6,12,14,17 --data tinyimagenet --test_batch_size 32 --epsilon_train 0.16 --lr 2.5e-4 --lr_scheduler exp

# train Tinyimagenet-MaxMin on 4 GPUs, Local bound + BCP loss
python -m torch.distributed.launch --nproc_per_node=4 train_tiny_multi.py --warmup 0 --model tiny_clmaxmin --batch_size 32 --rampup 125 --sniter 3 --epochs 250 --sparse 0.01 --indices 0,6,12,14,17 --data tinyimagenet --test_batch_size 32 --epsilon_train 0.16 --lr 1e-4 --lr_scheduler exp

###### Evaluating ######

# Evaluating the Relu model
python -m torch.distributed.launch --nproc_per_node=4 evaluate_tiny.py --data 'tinyimagenet' --epsilon 0.141 --model 'tinyimagenet_relux' --test_batch_size 32 --test_sniter 2000 --saved_model 'pretrained/relu/tiny_8C2F'

# Evaluating the MaxMin model
python -m torch.distributed.launch --nproc_per_node=2 evaluate_tiny.py --data 'tinyimagenet' --epsilon 0.141 --model 'tiny_clmaxmin' --test_batch_size 32 --test_sniter 2000 --saved_model 'pretrained/maxmin/tiny_8C2F_maxmin'

wait 
echo "All done"
