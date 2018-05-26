import torch
import torchvision
# import pudb;pu.db
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import pudb


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])



learning_rate = 0.001
K = 96
epochs = 10
droput_p = 0.5
batch_size = 4

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform = transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=4096,
                                         shuffle=False, num_workers=2)

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
        out_r = self.rcl_1_feed_fwd(out)
        for i in range(3):
            out_r = self.rcl_1_rec(out_r) + self.rcl_1_feed_fwd(out)
        out = out_r
        out = self.lrn(self.relu(out))
        out = self.droput(out)

        # Second RCL
        out_r = self.rcl_2_feed_fwd(out)
        for i in range(3):
            out_r = self.rcl_2_rec(out_r) + self.rcl_2_feed_fwd(out)

        out = out_r
        out = self.lrn(self.relu(out))
        out = self.droput(out)
        out = self.max_pool(out)

        # Third RCL 
        out_r = self.rcl_3_feed_fwd(out)
        for i in range(3):
            out_r = self.rcl_3_rec(out_r) + self.rcl_3_feed_fwd(out)
        out = out_r
        out = self.lrn(self.relu(out))
        out = self.droput(out)

        # Fourth RCL
        out_r = self.rcl_4_feed_fwd(out)
        for i in range(3):
            out_r = self.rcl_4_rec(out_r) + self.rcl_4_feed_fwd(out)

        out = out_r
        out = self.lrn(self.relu(out))
        out = self.droput(out)
        out = nn.MaxPool2d(out.shape[-1])(out)

        out = out.view(-1,K)
        out = self.linear(out)
        out = self.softmax(out)

        return out



use_gpu = torch.cuda.is_available()

net = RCNN()

if use_gpu:
	net.cuda()

correct = 0
total = 0

net.load_state_dict(torch.load('./checkpoints/19-24-2.2642.pyt'))
net.eval()

with torch.no_grad():
    total_iterations = len(list(testloader))
    k = 0
    for data in testloader:
        k = k+1
        images, labels = data
        # pu.db
        if use_gpu:
        	images = images.cuda()
        	labels = labels.cuda()
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        print("{} of {} iterations done".format(k,total_iterations))

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))