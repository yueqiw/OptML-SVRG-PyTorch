import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import json

experiments = ["CIFAR10_SGD_lr_search", "CIFAR10_SVRG_lr_search", 
               "CIFAR10_SGD_small_batch_lr_search", "CIFAR10_SVRG_small_batch_lr_search"]

data_dir = "outputs"

# load data
all_stats = []
for exp in experiments:
    folders = [x for x in os.listdir(data_dir) if x.startswith(exp)]
    
    for folder in folders:
        #print(folder)
        stats = {}
        args_file = os.path.join(data_dir, folder, "args.json")
        with open(args_file, "r") as f:
            args = json.load(f)
        #print(args)
        npz_file = os.path.join(data_dir, folder, "train_stats.npz")
        npz = np.load(npz_file)
        stats['epoch'] = pd.Series(np.arange(len(npz['train_loss'])))
        stats['train_loss'] = pd.Series(npz['train_loss'])
        stats['train_acc'] = pd.Series(npz['train_acc'])
        stats['val_loss'] = pd.Series(npz['val_loss'])
        stats['val_acc'] = pd.Series(npz['val_acc'])
        stats = pd.DataFrame(stats)
        stats['optimizer'] = args['optimizer']
        stats['learning_rate'] = args['lr']
        stats['batch_size'] = args['batch_size']
        all_stats.append(stats)
stats_df = pd.concat(all_stats)
stats_df.head()

stats_df = stats_df[stats_df['learning_rate'] >= 0.001]

# plot results
sns.set(palette="muted", font_scale=1.1)
sns.set_style("ticks")

# training loss
g = sns.FacetGrid(stats_df, col="optimizer", row='batch_size', hue='learning_rate', height=3, aspect=1.25)
def plot_loss(x, y, **kwargs):
    plt.plot(x, y, **kwargs)
    #plt.ylim(0, 0.5)
g = g.map(plot_loss, "epoch", "train_loss").add_legend()
plt.subplots_adjust(top=0.9)
g.fig.suptitle("Training Loss (CIFAR10)", fontsize=15)
g.savefig("figures/cifar10_lr_search_train_loss.png")

# training accuracy
g = sns.FacetGrid(stats_df, col="optimizer", row='batch_size', hue='learning_rate', height=3, aspect=1.25)
def plot_loss(x, y, **kwargs):
    plt.plot(x, y, **kwargs)
    plt.ylim(0.5, 1)
g = g.map(plot_loss, "epoch", "train_acc").add_legend()
plt.subplots_adjust(top=0.9)
g.fig.suptitle("Training Accuracy (CIFAR10)", fontsize=15)
g.savefig("figures/cifar10_lr_search_train_acc.png")

# validation accuracy
g = sns.FacetGrid(stats_df, col="optimizer", row='batch_size', hue='learning_rate', height=3, aspect=1.25)
def plot_loss(x, y, **kwargs):
    plt.plot(x, y, **kwargs)
    plt.ylim(0.4, 0.7)
g = g.map(plot_loss, "epoch", "val_acc").add_legend()
plt.subplots_adjust(top=0.9)
g.fig.suptitle("Validation Accuracy (CIFAR10)", fontsize=15)
g.savefig("figures/cifar10_lr_search_val_acc.png")