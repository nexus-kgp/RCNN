import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
import pudb

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.47, 0.47, 0.47), (0.5, 0.5, 0.5))])

learning_rate = 0.001
K = 96
epochs = 10000
droput_p = 0.5
batch_size = 2048

best_loss = 100000

# When you load the model back again via state_dict method,\
# \remember to do net.eval(), otherwise the results will differ

use_gpu = torch.cuda.is_available()

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

dataiter = iter(trainloader)
images, labels = dataiter.next()

class RCNN(nn.Module):

    def __init__(self):
        super(RCNN, self).__init__()

        self.max_pool = nn.MaxPool2d(3,2)
        self.lrn = nn.LocalResponseNorm(13)
        self.droput = nn.Dropout(droput_p)
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(3, K, 5,1)

        self.rcl_1_feed_fwd = nn.Conv2d(K,K,3,1,1)
        self.rcl_1_rec = nn.Conv2d(K,K,3,1,1)

        self.conv2 = nn.Conv2d(K,K,3,1,1)
        
        self.rcl_2_feed_fwd = nn.Conv2d(K,K,3,1,1)
        self.rcl_2_rec = nn.Conv2d(K,K,3,1,1)

        self.conv2 = nn.Conv2d(K,K,3,1,1)
        
        self.rcl_3_feed_fwd = nn.Conv2d(K,K,3,1,1)
        self.rcl_3_rec = nn.Conv2d(K,K,3,1,1)

        self.conv3 = nn.Conv2d(K,K,3,1,1)
        
        self.rcl_4_feed_fwd = nn.Conv2d(K,K,3,1,1)
        self.rcl_4_rec = nn.Conv2d(K,K,3,1,1)

        self.linear = nn.Linear(K,10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        out = self.conv1(x)
        out = self.max_pool(out)

        # First RCL
        out_f = out_r = self.rcl_1_feed_fwd(out)
        for i in range(3):
            out_r = self.rcl_1_rec(out_r) + out_f
        out = out_r
        out = self.lrn(self.relu(out))
        out = self.droput(out)

        # Second RCL
        out_f = out_r = self.rcl_2_feed_fwd(out)
        for i in range(3):
            out_r = self.rcl_2_rec(out_r) + out_f

        out = out_r
        out = self.lrn(self.relu(out))
        out = self.droput(out)
        out = self.max_pool(out)

        # Third RCL 
        out_f = out_r = self.rcl_3_feed_fwd(out)
        for i in range(3):
            out_r = self.rcl_3_rec(out_r) + out_f
        out = out_r
        out = self.lrn(self.relu(out))
        out = self.droput(out)

        # Fourth RCL
        out_f = out_r = self.rcl_4_feed_fwd(out)
        for i in range(3):
            out_r = self.rcl_4_rec(out_r) + out_f

        out = out_r
        out = self.lrn(self.relu(out))
        out = self.droput(out)
        out = nn.MaxPool2d(out.shape[-1])(out)

        out = out.view(-1,K)
        out = self.linear(out)
        out = self.softmax(out)

        return out



net = RCNN()

if use_gpu:
    net.cuda()


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)


#make a checkpoints directory
os.system("mkdir -p checkpoints")


# Train the network
for epoch in range(epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        if use_gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # scheduler.step(loss)

        # print statistics
        running_loss += loss.item()
        if i % 5 == 4:    # print every 5 mini-batches
            
            print('[%d, %5d] loss: %.4f' %
                  (epoch + 1, i + 1, loss))

            if loss.detach().cpu().numpy() < best_loss:
                np.save("best_loss.npy",loss.detach())
                ckpt_path = "checkpoints/{0}-{1}-{2:0.4f}.pyt".\
                format(epoch,i,(loss.detach().cpu().numpy()))

                print("Better loss (= {0:0.4f}) found, \
                    saving model at {1}".format(loss,ckpt_path))

                best_loss = loss.detach().cpu().numpy()
                torch.save(net.state_dict(),ckpt_path)

print('Finished Training')