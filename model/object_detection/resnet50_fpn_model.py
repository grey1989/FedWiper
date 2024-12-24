import os

import torch
import torch.nn as nn
from torchvision.ops.misc import FrozenBatchNorm2d

from .feature_pyramid_network import BackboneWithFPN, LastLevelMaxPool

class ForgetLayer(nn.Module):
    def __init__(self, channel: int):
        super().__init__()
        self.layer1 = nn.Conv2d(in_channels=channel, out_channels=channel // 2, kernel_size=1)
        self.relu = nn.ReLU()
        self.layer2 = nn.Conv2d(in_channels=channel // 2, out_channels=channel, kernel_size=1)


    def forward(self, x):
        identity = x
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x += identity
        x = self.relu(x)
        return x

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = norm_layer(out_channel)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = norm_layer(out_channel)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel * self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = norm_layer(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, blocks_num, num_classes=1000, include_top=True, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.include_top = include_top
        self.in_channel = 64

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = norm_layer(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.forget1 = ForgetLayer(channel=256)
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.forget2 = ForgetLayer(channel=512)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.forget3 = ForgetLayer(channel=1024)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        self.forget4 = ForgetLayer(channel=2048)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel, channel, downsample=downsample,
                            stride=stride, norm_layer=norm_layer))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    #  替换公共层参数
    def resume_from_disk(self, weights):
        net_dict = self.state_dict()
        # 寻找网络中公共层，并保留预训练参数
        state_dict = {k: v for k, v in weights.items() if k in net_dict.keys()}
        # 将预训练参数更新到新的网络层
        net_dict.update(state_dict)
        # 加载预训练参数
        self.load_state_dict(net_dict)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        # x = self.forget1(x)
        x = self.layer2(x)
        # x = self.forget2(x)
        x = self.layer3(x)
        # x = self.forget3(x)
        x = self.layer4(x)
        # x = self.forget4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


def overwrite_eps(model, eps):
    """
    This method overwrites the default eps values of all the
    FrozenBatchNorm2d layers of the model with the provided value.
    This is necessary to address the BC-breaking change introduced
    by the bug-fix at pytorch/vision#2933. The overwrite is applied
    only when the pretrained weights are loaded to maintain compatibility
    with previous versions.
    Args:
        model (nn.Module): The model on which we perform the overwrite.
        eps (float): The new value of eps.
    """
    for module in model.modules():
        if isinstance(module, FrozenBatchNorm2d):
            module.eps = eps


def resnet50_fpn_backbone(pretrain_path="",
                          norm_layer=FrozenBatchNorm2d,  # FrozenBatchNorm2d的功能与BatchNorm2d类似，但参数无法更新
                          trainable_layers=3,
                          returned_layers=None,
                          extra_blocks=None):
    """
    搭建resnet50_fpn——backbone
    Args:
        pretrain_path: resnet50的预训练权重，如果不使用就默认为空
        norm_layer: 官方默认的是FrozenBatchNorm2d，即不会更新参数的bn层(因为如果batch_size设置的很小会导致效果更差，还不如不用bn层)
                    如果自己的GPU显存很大可以设置很大的batch_size，那么自己可以传入正常的BatchNorm2d层
                    (https://github.com/facebookresearch/maskrcnn-benchmark/issues/267)
        trainable_layers: 指定训练哪些层结构
        returned_layers: 指定哪些层的输出需要返回
        extra_blocks: 在输出的特征层基础上额外添加的层结构
    Returns:
    """
    resnet_backbone = ResNet(Bottleneck, [3, 4, 6, 3],
                             include_top=False,
                             norm_layer=norm_layer)
    # print(resnet_backbone)
    if isinstance(norm_layer, FrozenBatchNorm2d):
        overwrite_eps(resnet_backbone, 0.0)

    if pretrain_path != "":
        assert os.path.exists(pretrain_path), "{} is not exist.".format(pretrain_path)
        # 载入预训练权重
        # print("!!!!!!!!!!!!!!!!!")
        print(resnet_backbone.load_state_dict(torch.load(pretrain_path), strict=False))

        # weights_dict = torch.load(pretrain_path)
        # resnet_backbone.resume_from_disk(weights_dict)

    # select layers that wont be frozen
    assert 0 <= trainable_layers <= 5
    layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'][:trainable_layers]

    # 如果要训练所有层结构的话，不要忘了conv1后还有一个bn1
    if trainable_layers == 5:
        layers_to_train.append("bn1")

    # freeze layers
    # for name, parameter in resnet_backbone.named_parameters():
    #     # 只训练不在layers_to_train列表中的层结构
    #     print("!!!!!!!!!!!!!!!!!")
    #     if all([not name.startswith(layer) for layer in layers_to_train]):
    #         print(name)
    #         parameter.requires_grad_(False)

    #修改：更改冻住部分
    # for name, param in resnet_backbone.named_parameters():
    #     if ("forget" not in name):
    #         param.requires_grad = False

    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool()

    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]
    # 返回的特征层个数肯定大于0小于5
    assert min(returned_layers) > 0 and max(returned_layers) < 5

    # return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
    # 修改 ：从forget层获取fpn
    return_layers = {f'layer{k}': str(v) for v, k in enumerate(returned_layers)}
    # return_layers = {'forget1': '0', 'forget2': '1', 'forget3': '2', 'forget4': '3'}
    print(return_layers)
    # in_channel 为layer4的输出特征矩阵channel = 2048
    in_channels_stage2 = resnet_backbone.in_channel // 8  # 256
    # 记录resnet50提供给fpn的每个特征层channel
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
    # 通过fpn后得到的每个特征层的channel
    out_channels = 256
    return BackboneWithFPN(resnet_backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks)