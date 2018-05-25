import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

learning_rate = 0.001
K = 96
epochs = 10
droput_p = 0.5
batch_size = 4
best_loss = 1000000


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

        self.linear = nn.Linear(K*2*2,10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        out = self.conv1(x)
        out = self.max_pool(out)

        # First RCL
        out_r = self.rcl_1_feed_fwd(out)
        for i in range(3):
            out_r = self.rcl_1_rec(out_r) + self.rcl_1_feed_fwd(out)
        out = out_r
        out = self.droput(out)

        # Second RCL
        out_r = self.rcl_2_feed_fwd(out)
        for i in range(3):
            out_r = self.rcl_2_rec(out_r) + self.rcl_2_feed_fwd(out)

        out = out_r
        out = self.droput(out)
        out = self.max_pool(out)

        # Third RCL 
        out_r = self.rcl_3_feed_fwd(out)
        for i in range(3):
            out_r = self.rcl_3_rec(out_r) + self.rcl_3_feed_fwd(out)
        out = out_r
        out = self.droput(out)

        # Fourth RCL
        out_r = self.rcl_4_feed_fwd(out)
        for i in range(3):
            out_r = self.rcl_4_rec(out_r) + self.rcl_4_feed_fwd(out)

        out = out_r
        out = self.droput(out)
        out = self.max_pool(out)

        out = out.view(-1,K*2*2)
        out = self.linear(out)
        out = self.softmax(out)

        return out

net = RCNN()

if use_gpu:
    net.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

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
        print(loss)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 1000 == 999:    # print every 1000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            if loss < best_loss:
                ckpt_path = "checkpoints/{}-{}.pyt".format(epoch,i)
                print("Better loss found, saving model at {}".format(ckpt_path))
                best_loss = loss
                torch.save(net.state_dict(),ckpt_path)

            running_loss = 0.0

print('Finished Training')