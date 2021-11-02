#!/bin/bash

###### Training ######
# train MNIST (4C3F), Local bound + Gloro loss
python train_mnist.py --model relux --sniter 5 --init 2.0 --epsilon_train 1.75 --gloro --no_augmentation --opt_iter 10 --prefix 'relu/mnist_4C3F'

# train CIFAR-10 (4C3F), Local bound + BCP loss
python train_cifar10.py --model relux --sniter 2 --init 2.0 --end_lr 1e-6

# train CIFAR-10 (6C2F), Local bound + BCP loss
python train_cifar10.py --model c6f2_relux --sniter 2 --init 2.0 --end_lr 1e-6 --prefix 'relu/cifar_6C2F'

# train CIFAR-10 (6C2F), Local bound + Gloro loss
python train_cifar10.py --model c6f2_relux --sniter 2 --init 2.0 --end_lr 1e-6 --gloro

# train CIFAR-10 (6C2F-MaxMin), Local bound + BCP loss
python train_cifar10.py --model c6f2_clmaxmin --sniter 2 --end_lr 1e-6 --prefix 'maxmin/cifar_6C2F_maxmin'

###### Evaluating ######
python evaluate.py --data 'mnist' --model 'relux' --epsilon 1.58 --test_batch_size 256 --test_sniter 2000 --saved_model 'pretrained/relu/mnist_4C3F'

python evaluate.py --data 'cifar10' --model 'relux' --epsilon 0.141 --test_batch_size 256 --test_sniter 2000 --saved_model 'pretrained/relu/cifar_4C3F'

python evaluate.py --data 'cifar10' --model 'c6f2_relux' --epsilon 0.141 --test_batch_size 256 --test_sniter 2000 --saved_model 'pretrained/relu/cifar_6C2F'

python evaluate.py --data 'cifar10' --model 'c6f2_clmaxmin' --epsilon 0.141 --test_batch_size 256 --test_sniter 2000 --saved_model 'pretrained/maxmin/cifar_6C2F_maxmin'

wait 
echo "All done"
