import torch
from torch import nn 
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from svrg import SVRG_k, SVRG_0 

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
# Download MNIST dataset and set the valset as the test test
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,)),])
valset = torchvision.datasets.MNIST('MNIST_data', download = True, train = False, transform=transform)
train_set = torchvision.datasets.MNIST("MNIST_data", train = True, transform=transform, download=True)
# Use dataloader to load the data
train_set = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
valset = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

#create data with different branch_size
train_set_bb = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)
train_set_sb = torch.utils.data.DataLoader(train_set, batch_size = 16, shuffle= True)


#Create the nn model
input_size = 784
hidden_sizes = [128, 64]
output_size = 10
#The input non-convex function(model) is Sequential function fron torch. Notice that we have implemented an activation function to make it non-convex

#create two models for svgd 
#model_k is for calculating the gradient of the one random selected sample.
model_k = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1)).to(device)
#model_0 is for calculating the mean gradient of all the sample.
model_0 = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1)).to(device)
model_k.train()
model_0.train()
#set up the optimizer
losses_svrg = []
loss_fn = nn.NLLLoss()

lr = 1e-4
optimizer_k = SVRG_k(model_k.parameters(), lr=lr)
optimizer_0 = SVRG_0(model_0.parameters(), lr=lr)
n_epochs = 100
#set up the starting value
S = n_epochs
accuracies_svgd = []
#Start the for loop
for s in range(1, S+1):
    #calculate the mean gradient
    for images, labels in train_set:
        #reshape the images
        images = images.view(images.shape[0], -1).to(device)
        yhat = model_0(images)
        labels = labels.to(device)
        loss = loss_fn(yhat, labels)/len(train_set)
        loss.backward()
    u = optimizer_0.get_param_groups()
    #pass the current paramesters of optimizer_0 to optimizer_k 
    optimizer_k.set_u(u)
    optimizer_0.zero_grad()
    
    for images, labels in train_set:
        #transform the torch type to cuda to run this faster 
        images = images.view(images.shape[0], -1).to(device)
        yhat = model_k(images)
        labels = labels.to(device)
        loss = loss_fn(yhat, labels)
        loss.backward()
        losses_svrg.append(loss.data)
        yhat2 = model_0(images)
        loss2 = loss_fn(yhat2, labels)
        loss2.backward()
        optimizer_k.step(optimizer_0.get_param_groups())
        optimizer_k.zero_grad()
        optimizer_0.zero_grad()
        

    optimizer_0.set_param_groups(optimizer_k.get_param_groups())
    if s % 1 == 0:
        print("epoch: {}, loss: {}".format(s, loss.data))
    accurate = 0
    total = 0
    for images, labels in valset:
        images = images.view(images.shape[0], -1).to(device)
        yhat = model_k(images)
        labels = labels.to(device)
        values, indices = yhat.max(1)
        #print(indices == labels)
        accurate += int((indices == labels).sum().cpu().detach())
        total += len(indices == labels)
        #print(accurate, total, accurate/float(total))
    
    accuracy = accurate/float(total)
    print("accuacy : {}".format(accuracy))
    accuracies_svgd.append(accuracy)

#plot the loss of sgd
losses_svrg = np.array(losses_svrg)
np.savetxt('loss_svrg.txt', losses_svrg)
plt.plot(losses_svrg)

accuracies_svgd = np.array(accuracies_svgd)
np.savetxt('acc_svrg.txt', accuracies_svgd)
plt.plot(accuracies_svgd)