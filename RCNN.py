import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

dataiter = iter(trainloader)
images, labels = dataiter.next()


learning_rate = 0.001
K = 96
epochs = 10
droput_p = 0.5
alpha = 0.001


class RCNN(nn.Module):

    def __init__(self):
        super(RCNN, self).__init__()

        self.max_pool = nn.MaxPool2d(3,2)
        self.lrn = nn.LocalResponseNorm(13,alpha)
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

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

# Train the network
for epoch in range(epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        print(loss)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')