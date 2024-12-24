import torch
import torch.nn as nn
import torchvision
import torch.utils.data as Data
import torchvision.transforms as transforms
import sys
from torch.utils.data import Subset
import os
import random
mnist_train = torchvision.datasets.MNIST(root='./data/LeNet_data', train=True, download=True,
                                                transform=transforms.ToTensor())
mnist_test = torchvision.datasets.MNIST(root='./data/LeNet_data', train=False, download=True,
                                               transform=transforms.ToTensor())
x = [i for i in range(0, 10000)]

mnist_train = Subset(mnist_train, indices=x)

batch_size = 128
if sys.platform.startswith('win'):
    num_workers = 0
else:
    num_workers = 4

train_iter = Data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_iter = Data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

class ForgetLayer(nn.Module):
    def __init__(self, channel: int):
        super().__init__()
        self.layer1 = nn.Conv2d(in_channels=channel, out_channels=channel //2 , kernel_size=1)
        self.relu = nn.ReLU()
        self.sigmod = nn.Sigmoid()
        self.layer2 = nn.Conv2d(in_channels=channel // 2, out_channels=channel, kernel_size=1)


    def forward(self, x):
        identity = x
        x = self.layer1(x)
        x = self.relu(x)
        # x = self.sigmod(x)
        x = self.layer2(x)
        x += identity
        x = self.relu(x)
        return x

class LeNet_adpter(nn.Module):
    def __init__(self):
        super(LeNet_adpter,self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(1,6,5),
            nn.Sigmoid(),
            nn.MaxPool2d(2,2),

            # ForgetLayer(channel=6),
            nn.Conv2d(6,16,5),


            nn.Sigmoid(),
            nn.MaxPool2d(2,2),
            ForgetLayer(channel=16),

        )
        self.classifer = nn.Sequential(
            nn.Linear(4*4*16,120),
            nn.Sigmoid(),
            nn.Linear(120,84),
            nn.Sigmoid(),
            nn.Linear(84,10),
        )
    def forward(self,x):
        feature = self.feature(x)
        classifer = self.classifer(feature.view(feature.shape[0],-1))
        return classifer
    def resume_from_disk(self, weights):
        net_dict = self.state_dict()

        # 寻找网络中公共层，并保留预训练参数
        state_dict = {k: v for k, v in weights.items() if k in net_dict.keys()}
        # state_dict = self.get_vgg13_dict(weights, net_dict)
        # 将预训练参数更新到新的网络层
        net_dict.update(state_dict)
        # 加载预训练参数
        self.load_state_dict(net_dict)
        #print(self.state_dict())

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(1,6,5),
            nn.Sigmoid(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(6,16,5),
            nn.Sigmoid(),
            nn.MaxPool2d(2,2)
        )
        self.classifer = nn.Sequential(
            nn.Linear(4*4*16,120),
            nn.Sigmoid(),
            nn.Linear(120,84),
            nn.Sigmoid(),
            nn.Linear(84,10),
        )
    def forward(self,x):
        feature = self.feature(x)
        classifer = self.classifer(feature.view(feature.shape[0],-1))
        return classifer

# net = LeNet()
#
# print(net)
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Traing on ", device)
# num_epochs = 20
# loss = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(net.parameters(), lr=0.005)
# net = net.to(device)
#
# step = 0
# for epoch in range(1, num_epochs + 1):
#     for x, y in train_iter:
#         net.train()
#         x, y = x.to(device), y.to(device)
#         out = net(x)
#         l = loss(out, y)
#         optimizer.zero_grad()
#         l.backward()
#         optimizer.step()
#         step += 1
#         if step % 100 == 0:
#             with torch.no_grad():
#                 net.eval()
#                 acc_sum = (net(x).argmax(dim=1) == y).float().sum().item()
#                 print("Epoch:{},Step:{},Loss:{},train acc:{}".format(epoch, step, l.item(), acc_sum / x.shape[0]))
#
# torch.save({'state_dict': net.state_dict()},
#                            os.path.join("./weights/", 'fashionMnist.pth'))
#
# def evaluate_accuracy(data_iter,net):
#     acc_sum,n = 0.,0
#
#     with torch.no_grad():
#         net.eval()
#         for x,y in data_iter:
#             x, y = x.to(device), y.to(device)
#             acc_sum += (net(x).argmax(dim=1)==y).float().sum().item()
#             n += y.shape[0]
#     return acc_sum/n
#
# print(evaluate_accuracy(test_iter,net))
