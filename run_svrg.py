import torch
from torch import nn 
from torch.utils.data import DataLoader
import numpy as np
import argparse
import os 
import json
from datetime import datetime
import time
from sgd import SGD_Simple
from svrg import SVRG_k, SVRG_Snapshot
from utils import MNIST_dataset, CIFAR10_dataset, MNIST_two_layers, MNIST_one_layer, CIFAR10_ConvNet, AverageCalculator, accuracy, plot_train_stats

parser = argparse.ArgumentParser(description="Train SVRG/SGD on MNIST data.")
parser.add_argument('--optimizer', type=str, default="SVRG",
                    help="optimizer.")
parser.add_argument('--nn_model', type=str, default="MNIST_one_layer",
                    help="neural network model.")
parser.add_argument('--dataset', type=str, default="MNIST",
                    help="neural network model.")
parser.add_argument('--n_epoch', type=int, default=100,
                    help="number of training iterations.")
parser.add_argument('--lr', type=float, default=0.001,
                    help="learning rate.")
parser.add_argument('--batch_size', type=int, default=64,
                    help="batch size.")
parser.add_argument('--weight_decay', type=float, default=0.0001,
                    help="regularization strength.")
parser.add_argument('--exp_name', type=str, default="",
                    help="name of the experiment.")
parser.add_argument('--print_every', type=int, default=1,
                    help="how often to print the loss.")


OUTPUT_DIR = "outputs"
BATCH_SIZE_LARGE = 256  # for testing and the full-batch outer train loop

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
print("Using device: {}".format(device))

def train_epoch_SGD(model, optimizer, train_loader, loss_fn, flatten_img=True):
    model.train()
    loss = AverageCalculator()
    acc = AverageCalculator()
    
    for images, labels in train_loader:
        images = images.to(device)
        if flatten_img:
            images = images.view(images.shape[0], -1)
        yhat = model(images)
        labels = labels.to(device)
        loss_iter = loss_fn(yhat, labels)

        # optimization 
        optimizer.zero_grad()
        loss_iter.backward()    
        optimizer.step()

        # logging 
        acc_iter = accuracy(yhat, labels)
        loss.update(loss_iter.data.item())
        acc.update(acc_iter)
    
    return loss.avg, acc.avg

def train_epoch_SVRG(model_k, model_snapshot, optimizer_k, optimizer_snapshot, train_loader, loss_fn, flatten_img=True):
    model_k.train()
    model_snapshot.train()
    loss = AverageCalculator()
    acc = AverageCalculator()

    # calculate the mean gradient
    optimizer_snapshot.zero_grad()  # zero_grad outside for loop, accumulate gradient inside
    for images, labels in train_loader:
        images = images.to(device)
        if flatten_img:
            images = images.view(images.shape[0], -1)
        yhat = model_snapshot(images)
        labels = labels.to(device)
        snapshot_loss = loss_fn(yhat, labels) / len(train_loader)
        snapshot_loss.backward()

    # pass the current paramesters of optimizer_0 to optimizer_k 
    u = optimizer_snapshot.get_param_groups()
    optimizer_k.set_u(u)
    
    for images, labels in train_loader:
        images = images.to(device)
        if flatten_img:
            images = images.view(images.shape[0], -1)
        yhat = model_k(images)
        labels = labels.to(device)
        loss_iter = loss_fn(yhat, labels)

        # optimization 
        optimizer_k.zero_grad()
        loss_iter.backward()    

        yhat2 = model_snapshot(images)
        loss2 = loss_fn(yhat2, labels)

        optimizer_snapshot.zero_grad()
        loss2.backward()

        optimizer_k.step(optimizer_snapshot.get_param_groups())

        # logging 
        acc_iter = accuracy(yhat, labels)
        loss.update(loss_iter.data.item())
        acc.update(acc_iter)
    
    # update the snapshot 
    optimizer_snapshot.set_param_groups(optimizer_k.get_param_groups())
    
    return loss.avg, acc.avg


def validate_epoch(model, val_loader, loss_fn):
    """One epoch of validation
    """
    model.eval()
    loss = AverageCalculator()
    acc = AverageCalculator()

    for images, labels in val_loader:
        images = images.view(images.shape[0], -1).to(device)
        yhat = model(images)
        labels = labels.to(device)

        # logging 
        loss_iter = loss_fn(yhat, labels)
        acc_iter = accuracy(yhat, labels)
        loss.update(loss_iter.data.item())
        acc.update(acc_iter)
    
    return loss.avg, acc.avg

if __name__ == "__main__":
    args = parser.parse_args()
    args_dict = vars(args)

    if not args.optimizer in ['SGD', 'SVRG']:
        raise ValueError("--optimizer must be 'SGD' or 'SVRG'.")
    print(args_dict)

    # load the data
    if args.dataset == "MNIST":
        train_set, val_set = MNIST_dataset()
        flatten_img = True
    elif args.dataset == "CIFAR10":
        train_set, val_set = CIFAR10_dataset() 
        flatten_img = False
    else:
        raise ValueError("Unknown dataset")
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE_LARGE, shuffle=True)

    if args.nn_model == "MNIST_one_layer":
        NN_model = MNIST_one_layer  # function name 
    elif args.nn_model == "MNIST_two_layers":
        NN_model = MNIST_two_layers
    elif args.nn_model == "CIFAR10_convnet":
        NN_model = CIFAR10_ConvNet
    else:
        raise ValueError("Unknown nn_model.")

    model = NN_model().to(device)
    if args.optimizer == 'SVRG':
        model_snapshot = NN_model().to(device)

    lr = args.lr  # learning rate
    n_epoch = args.n_epoch  # the number of epochs
    loss_fn = nn.NLLLoss()  # The loss function 

    # the optimizer 
    if args.optimizer == "SGD":
        optimizer = SGD_Simple(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    elif args.optimizer == "SVRG":
        optimizer = SVRG_k(model.parameters(), lr=lr, weight_decay=args.weight_decay)
        optimizer_snapshot = SVRG_Snapshot(model_snapshot.parameters())


    # output folder 
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_name = timestamp + "_" + args.optimizer + "_" + args.nn_model
    if args.exp_name != "":
        model_name = args.exp_name + '_' + model_name
    log_dir = os.path.join(OUTPUT_DIR, model_name)
    if not os.path.isdir(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    with open(os.path.join(log_dir, "args.json"), "w") as f:
        json.dump(args_dict, f)

    # store training stats
    train_loss_all, val_loss_all = [], []
    train_acc_all, val_acc_all = [], []

    for epoch in range(n_epoch):
        t0 = time.time()

        # training 
        if args.optimizer == "SGD":
            train_loss, train_acc = train_epoch_SGD(model, optimizer, train_loader, loss_fn, flatten_img=flatten_img)
        elif args.optimizer == "SVRG":
            train_loss, train_acc = train_epoch_SVRG(model, model_snapshot, optimizer, optimizer_snapshot, train_loader, loss_fn, flatten_img=flatten_img)
        
        # validation 
        val_loss, val_acc = validate_epoch(model, val_loader, loss_fn)
        
        train_loss_all.append(train_loss)  # averaged loss for the current epoch 
        train_acc_all.append(train_acc)
        val_loss_all.append(val_loss)  
        val_acc_all.append(val_acc)
        
        fmt_str = "epoch: {}, train loss: {:.4f}, train acc: {:.4f}, val loss: {:.4f}, val acc: {:.4f}, time: {:.2f}"

        if epoch % args.print_every == 0:
            print(fmt_str.format(epoch, train_loss, train_acc, val_loss, val_acc, time.time() - t0))

        # save data and plot 
        if (epoch + 1) % 5 == 0:
            np.savez(os.path.join(log_dir, 'train_stats.npz'), 
                train_loss=np.array(train_loss_all), train_acc=np.array(train_acc_all),
                val_loss=np.array(val_loss_all), val_acc=np.array(val_acc_all))
            plot_train_stats(train_loss_all, val_loss_all, train_acc_all, val_acc_all, log_dir, acc_low=0.9)
    # done
    open(os.path.join(log_dir, 'done'), 'a').close()
            