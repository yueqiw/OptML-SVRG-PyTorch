import torch
from torch import nn 
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from sgd import SGD_Simple

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
model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1)).to(device)
#Assign the learning rate, the number of epochs, 
lr = 1e-2
n_epochs = 100
#The loss function we ussed
loss_fn = nn.NLLLoss()
optimizer = SGD_Simple(model.parameters(), lr=lr)

model.train()
#create the list of losses for sgd
losses_sgd = []
accuracies_sgd = []
for epoch in range(n_epochs):
    for images, labels in train_set:
        images = images.view(images.shape[0], -1).to(device)
        yhat = model(images)
        #print(yhat.shape)
        labels = labels.to(device)
        #print(labels.shape)
        # TODO: random sample 
        loss = loss_fn(yhat, labels)
        loss.backward()    
        losses_sgd.append(loss.data)
        optimizer.step()
        optimizer.zero_grad()

    if epoch % 1 == 0:
        print("epoch: {}, loss: {}".format(epoch, loss.data))
    
    accurate = 0
    total = 0
    for images, labels in valset:
      images = images.view(images.shape[0], -1).to(device)
      yhat = model(images)
      labels = labels.to(device)
      values, indices = yhat.max(1)
      #print(indices == labels)
      accurate += int((indices == labels).sum().cpu().detach())
      total += len(indices == labels)
      #print(accurate, total, accurate/float(total))
    
    accuracy = accurate/float(total)
    print("accuacy : {}".format(accuracy))
    accuracies_sgd.append(accuracy)

#plot the loss of sgd
losses_sgd = np.array(losse_sgd)
np.savetxt('loss_sgd.txt', losses_sgd)
plt.plot(losses_svg)

accuracies_sgd = np.array(accuracies_sgd)
np.savetxt('acc_sgd.txt', accuracies_sgd)
plt.plot(accuracies_sgd)