# SVRG for neural networks (PyTorch)
Implementation of **stochastic variance reduction gradient descent (SVRG)** for optimizing non-convex neural network functions in PyTorch, according to [1]. 

This is a joint work with [Yusheng Hu](https://github.com/vcaptainv) and Bryant Wang during the [Optimization for Machine Learning (Fall 2019)](http://satyenkale.com/satyenkale/optml-f19/) course at Columbia University. 

[1] Zeyuan Allen-Zhu and Elad Hazan, Variance Reduction for Faster Non-Convex Optimization, ICML, 2016

## Code
```
git clone https://github.com/yueqiw/OptML-SVRG-NonConvex.git
# python 3.6
```

### Train neural networks with SVRG
```
python run_svrg.py --optimizer SVRG --nn_model CIFAR10_convnet --dataset CIFAR10 --lr 0.01
python run_svrg.py --optimizer SVRG --nn_model MNIST_one_layer --dataset MNIST --lr 0.01
python run_svrg.py --optimizer SGD --nn_model MNIST_one_layer --dataset MNIST --lr 0.01
```

### Run experiments to compare SVRG vs. SGD
```
python run.py --CIFAR10_SVRG_lr_search
python run.py --CIFAR10_SGD_lr_search
```
