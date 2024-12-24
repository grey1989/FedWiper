import collections
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict
import numpy as np
from tqdm import tqdm
import copy
import logging
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms import InterpolationMode

from denseNet_model import densenet121

BICUBIC = InterpolationMode.BICUBIC


# 判断是否有可用的GPU，如果有则使用GPU，否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(filename='./FedWiper.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    encoding='utf-8',
                    filemode='w')
# Data loading and preprocessing
transform = transforms.Compose(
    [transforms.Resize(224),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

transform_test = transforms.Compose(
    [transforms.Resize(224),
    transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

def _blendTrigger(img):
    # gauss_noise = np.random.normal(0, 40, img.shape)
    # img_arr = img.astype(np.float64)
    # img_arr = img_arr + gauss_noise
    # img_arr = np.clip(img_arr, 0, 255)
    # img_arr = img_arr.astype(np.uint8)
    for j in range(224 - 36, 224 - 1):
        for k in range(224 - 36, 224 - 1):
            # index = random.randint(0, 255)
            img[j, k] = [255, 0, 0]

    return img
# 自定义CIFAR10数据集类
class CustomCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, backdoor=False):
        super().__init__(root, train, transform, target_transform, download)
        self.backdoor = backdoor
        self.transform = transform
        if self.backdoor and self.train:
            # 如果是训练集且开启后门（投毒），记录要投毒的数据数量（这里取客户端数据量的25%，可根据实际调整）
            self.num_to_poison = 10000

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        # if self.train and ((index < 500) or (index >=10000 and index < 10500) or (index >=20000 and index < 20500) or (index >=30000 and index < 30500) or (index >=40000 and index < 40500)):
        if self.train:
            img = self.transform(img)
            # resize = Resize(224)
            # crop = transforms.RandomResizedCrop(224)
            # Flip = transforms.RandomHorizontalFlip()
            # img = resize(img)
            # img = crop(img)
            # img = Flip(img)
            # img = np.array(img)
            #
            # img = _blendTrigger(img)
            # img = torchvision.transforms.ToPILImage()(img)
            #
            # img = img.convert("RGB")
            # totensor = ToTensor()
            # img = totensor(img)
            # normal = Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            # img = normal(img)
            # target = 0
        elif self.train == False and self.backdoor:
            resize = Resize(224)
            crop = transforms.RandomResizedCrop(224)
            Flip = transforms.RandomHorizontalFlip()
            img = resize(img)
            # img = crop(img)
            # img = Flip(img)
            img = np.array(img)

            img = _blendTrigger(img)
            img = torchvision.transforms.ToPILImage()(img)
            img = img.convert("RGB")
            totensor = ToTensor()
            img = totensor(img)
            normal = Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            img = normal(img)
        else:
            if self.train:
                img = self.transform(img)
            else:
                img = transform_test(img)
        return img, target




trainset = CustomCIFAR10(root='./data', train=True, transform=transform, backdoor=True)


# Split client datasets
num_clients = 16
num_samples_per_client = len(trainset) // num_clients
client_datasets = []
for i in range(num_clients):
    start_idx = i * num_samples_per_client
    end_idx = (i + 1) * num_samples_per_client if i < num_clients - 1 else len(trainset)
    client_datasets.append(Subset(trainset, list(range(start_idx, end_idx))))


client_datasets = client_datasets[:5]


# Load the pre-trained ResNet34 model and adjust the output dimension of the final fully connected layer (adapt to 10 classes of CIFAR-10)
# model = torchvision.models.densenet121(pretrained=True)
#
# # 保存模型权重
# torch.save(model.state_dict(), './densenet_121.pth')
model = densenet121()

model.classifier.out_features = 10
pretrain_model = torch.load('./densenet_121.pth')
print('load pretrained weights')
model.resume_from_disk(pretrain_model)
freeze = True
if freeze:
    for name, param in model.named_parameters():
        if ("forget" not in name) and ("classifier" not in name):
            param.requires_grad = False

# 将模型移动到GPU设备上（如果有可用GPU）
model = model.to(device)

# Define the client update function (a simple example of the local training process)
def client_update(client_model, optimizer, train_loader, epochs, lr_scheduler):
    client_model.train()
    client_model = client_model.to(device)  # 将客户端模型移动到GPU设备上（如果有可用GPU）
    for epoch in range(epochs):
        for data, target in tqdm(train_loader):
            data, target = data.to(device), target.to(device)  # 将数据移动到GPU设备上（如果有可用GPU）
            optimizer.zero_grad()
            output = client_model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
        lr_scheduler.step()
    return client_model.state_dict()

# Define the server aggregation function (a simple example of averaging the parameters of each client)
def aggregate(global_model_state_dict, client_models_state_dicts):
    worker_state_dict = client_models_state_dicts
    weight_keys = list(worker_state_dict[0].keys())
    averaged_state_dict = collections.OrderedDict()
    for key in weight_keys:
        key_sum = 0
        for i in range(len(client_models_state_dicts)):
            key_sum = key_sum + worker_state_dict[i][key]
        averaged_state_dict[key] = key_sum / len(client_models_state_dicts)

    return averaged_state_dict





# Federated learning training loop (including data poisoning)
local_epochs = 40  # The number of local training epochs for each client
batch_size = 32
learning_rate = 0.001
global_model = model
global_model.train()
optimizer = optim.Adam(global_model.parameters(), lr=learning_rate, weight_decay=1e-8)


# Function to calculate normal accuracy on the test set
def calculate_normal_accuracy(model, test_loader):
    model = model.to(device)  # 将模型移动到GPU设备上（如果有可用GPU）
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)  # 将测试数据移动到GPU设备上（如果有可用GPU）
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100 * correct / total

# Function to calculate backdoor attack success rate on the test set
def calculate_backdoor_attack_rate(model, testset):
    model = model.to(device)  # 将模型移动到GPU设备上（如果有可用GPU）
    model.eval()
    backdoor_total = 0
    backdoor_correct = 0
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)  # 将数据和标签移动到GPU设备上（如果有可用GPU）
            non_zero_mask = target!= 0  # 创建掩码，标记出target不为0的样本
            valid_data = data[non_zero_mask]
            valid_target = target[non_zero_mask]
            if valid_data.size(0) > 0:  # 只有存在满足条件的数据时才进行预测
                poisoned_output = model(valid_data)
                _, poisoned_predicted = torch.max(poisoned_output.data, 1)
                backdoor_correct += (poisoned_predicted.cpu() == 0).sum().item()
            backdoor_total += non_zero_mask.sum().item()  # 统计满足条件的样本数量

    return 100 * backdoor_correct / backdoor_total if backdoor_total > 0 else 0

def calculate_backdoor_attack_rate_old(model, testset):
    model = model.to(device)  # 将模型移动到GPU设备上（如果有可用GPU）
    model.eval()
    backdoor_total = len(testset)
    backdoor_correct = 0
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for data, target in test_loader:
            poisoned_data, poisoned_label = data.to(device), target.to(device)  # 将带后门的数据移动到GPU设备上（如果有可用GPU）
            poisoned_output = model(poisoned_data)
            _, poisoned_predicted = torch.max(poisoned_output.data, 1)
            backdoor_correct += (poisoned_predicted.cpu() == 0).sum().item()
    return 100 * backdoor_correct / backdoor_total


for round in range(5):  # Assume 20 rounds of federated training
    print("第" + str(round) + "轮")
    logging.info("第" + str(round) + "轮")
    client_models = [copy.deepcopy(global_model) for _ in range(num_clients)]
    client_optimizers = [optim.Adam(client_model.parameters(), lr=learning_rate, weight_decay=1e-8) for client_model in
                         client_models]
    lr_schedulers = [MultiStepLR(optimizer, milestones=[5, 15, 25], gamma=0.5) for optimizer in client_optimizers]
    client_loaders = [DataLoader(client_dataset, batch_size=batch_size, shuffle=True) for client_dataset in
                      client_datasets]

    client_models_state_dicts = []
    for i in range(5):
        print("第" + str(i) + "个客户端")
        logging.info("第" + str(i) + "个客户端")
        client_model_state_dict = client_update(client_models[i], client_optimizers[i], client_loaders[i], local_epochs, lr_schedulers[i])
        client_models_state_dicts.append(client_model_state_dict)
    print("------------------")
    logging.info("_______________")
    global_model_state_dict = aggregate(global_model.state_dict(), client_models_state_dicts)
    global_model.load_state_dict(global_model_state_dict)

    # Calculate normal accuracy
    testset = CustomCIFAR10(root='./data', train=False, transform=transform, backdoor=False)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    normal_accuracy = calculate_normal_accuracy(global_model, test_loader)
    print('Accuracy of the global model on the test set: %.2f %%' % normal_accuracy)
    logging.info('Accuracy of the global model on the test set: %.2f %%' % normal_accuracy)

    # Calculate backdoor attack success rate
    testset = CustomCIFAR10(root='./data', train=False, transform=transform, backdoor=True)
    backdoor_attack_rate = calculate_backdoor_attack_rate(global_model, testset)
    print('Backdoor attack success rate: %.2f %%' % backdoor_attack_rate)
    logging.info('Backdoor attack success rate: %.2f %%' % backdoor_attack_rate)


