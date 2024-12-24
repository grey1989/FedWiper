import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


# class ForgetLayer(nn.Sequential):
#     def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
#         super(ForgetLayer, self).__init__()
#         self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
#         self.add_module('relu1', nn.ReLU(inplace=True)),
#         self.add_module('conv1', nn.Conv2d(num_input_features, 128, kernel_size=1, stride=1, bias=False)),
#
#         self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
#         self.add_module('relu2', nn.ReLU(inplace=True)),
#         self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
#                                            kernel_size=3, stride=1, padding=1, bias=False)),
#         self.drop_rate = drop_rate
#
#     def forward(self, x):
#         new_features = super(ForgetLayer, self).forward(x)
#         if self.drop_rate > 0:
#             new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
#         new_features = torch.cat([x, new_features], 1)
#         return new_features

class ForgetLayer1(nn.Sequential):
    def __init__(self, channel: int):
        super().__init__()
        self.layer1 = nn.Conv2d(in_channels=channel+32, out_channels=channel, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        return x


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)
        # if num_layers ==16 or num_layers ==24:
        if num_layers !=133:
            numsum = num_input_features + num_layers * growth_rate
            self.add_module('forgetnorm%d' %(num_layers), nn.BatchNorm2d(numsum))
            self.add_module("forgetrule%d" % (num_layers), nn.ReLU(inplace=True))
            self.add_module("forget%d" %(num_layers),ForgetLayer(numsum))
        # if num_layers ==16:
        #     numsum = num_input_features + num_layers * growth_rate
        #     self.add_module("forget%d" % (num_layers), ForgetLayer(numsum,growth_rate, bn_size, drop_rate))
        #     self.add_module("forget1%d" % (num_layers),ForgetLayer1(numsum))
class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

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

class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                # if num_layers == 24:
                #     self.features.add_module("forget%d" % (i + 1), ForgetLayer(num_features // 2))
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        ##加层
        # self.features.add_module("forgetconv2d1", ForgetLayer(num_features))
        # self.features.add_module('forgetnorm',nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        out = self.classifier(out)
        return out

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
def densenet121(**kwargs):
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16), **kwargs)
    return model


def densenet169(**kwargs):
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32), **kwargs)
    return model


def densenet201(**kwargs):
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32), **kwargs)
    return model


def densenet161(**kwargs):
    model = DenseNet(num_init_features=96, growth_rate=48, block_config=(6, 12, 36, 24), **kwargs)
    return model


if __name__ == '__main__':
    # 'DenseNet', 'densenet121', 'densenet169', 'densenet201', 'densenet161'
    # Example
    net = DenseNet()
    print(net)
