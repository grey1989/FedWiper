import torch
import torch.nn as nn
import torch.utils.data
from torch import Tensor
from torchsummary import summary
from torchvision import models
from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock

from module.base_module import BaseModule

class ForgetLayer(nn.Module):
    def __init__(self, channel: int):
        super().__init__()
        self.layer1 = nn.Conv2d(in_channels=channel, out_channels=channel * 2, kernel_size=1)
        self.relu = nn.ReLU()
        self.layer2 = nn.Conv2d(in_channels=channel * 2, out_channels=channel, kernel_size=1)


    def forward(self, x):
        identity = x
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x += identity
        x = self.relu(x)
        return x


class MyRes(ResNet, BaseModule):

    def __init__(self):
        super().__init__(block=BasicBlock, layers=[3, 4, 6, 3])

        self.forget1 = ForgetLayer(channel=64)
        self.forget2 = ForgetLayer(channel=128)
        self.forget3 = ForgetLayer(channel=256)
        self.forget4 = ForgetLayer(channel=512)
        self.classifier = nn.Linear(self.fc.in_features, 100)

    def forward(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        # x = self.forget1(x)
        #
        x = self.layer2(x)
        # x = self.forget2(x)

        x = self.layer3(x)
        # x = self.forget3(x)

        x = self.layer4(x)
        x = self.forget4(x)


        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

    def get_save_state_dict(self):
        state_dict = {k: v for k, v in self.state_dict().items() if 'forget' in k or 'classifier' in k}
        return state_dict




class OriginResNet34(ResNet, BaseModule):
    def __init__(self, weights_path: str or None, freeze: bool = False, num_classes=100):
        super().__init__(block=BasicBlock, layers=[3, 4, 6, 3])
        if weights_path is not None:
            weights = torch.load(weights_path)
            self.resume_from_disk(weights)
            if freeze:
                for name, param in self.named_parameters():
                    if "fc" not in name:
                        param.requires_grad = False
        # 修改fc层
        fc_features = self.fc.in_features
        self.fc = nn.Linear(fc_features, num_classes)


def load_changed_model() -> nn.Module:
    model = MyRes()
    net_dict = model.state_dict()
    predict_model = torch.load('../weights/resnet34-b627a593.pth')
    print('load pretrained weights')
    # 寻找网络中公共层，并保留预训练参数
    state_dict = {k: v for k, v in predict_model.items() if k in net_dict.keys()}
    # 将预训练参数更新到新的网络层
    net_dict.update(state_dict)
    # 加载预训练参数
    model.load_state_dict(net_dict)
    for name, param in model.named_parameters():
        if ("forget" not in name) and ("classifier" not in name):
            param.requires_grad = False
    return model


def load_origin_model(pretrain: bool = False, freeze: bool = False) -> nn.Module:
    # 调用模型
    model = models.resnet34()
    if pretrain:
        weights = torch.load('../weights/resnet34-b627a593.pth')
        model.load_state_dict(weights)
        if freeze:
            for name, param in model.named_parameters():
                if "fc" not in name:
                    param.requires_grad = False
    # 修改fc层
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, 10)
    return model


if __name__ == '__main__':
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    net = load_changed_model().to(device)
    summary(net, (3, 28, 28))
    print(net)
    net = load_origin_model().to(device)
    summary(net, (3, 28, 28))
    print(net)
