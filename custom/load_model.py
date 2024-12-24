import torch
from torch import nn
from torchvision import models
from torchvision import models
from module.module import MyRes, OriginResNet34
from denseNet_model import densenet121
from LeNet_adpter import LeNet,LeNet_adpter
num = 0
task_num = 0
def changed_model(pretrain=True, freeze=True) -> nn.Module:
    # model = MyRes()
    # model =  OriginResNet34('./weights/resnet34-b627a593.pth', False)
    #resnet
    model = densenet121()
    model.classifier.out_features = 10
    # model = LeNet_adpter()
    # print(model.state_dict())
    # for name, param in model.named_parameters():
    #     print(name)
    # print(model)
    global num
    global task_num
    if pretrain:
        if num % 20 < 4:

            pretrain_model = torch.load('./weights/densenet121-a639ec97.pth')
            # pretrain_model = torch.load('./resnet34_tiny_imagenet.pth')
            print('load pretrained weights')
            model.resume_from_disk(pretrain_model)
            num =num + 1
        else:
            # 聚合
            model_ckpt = torch.load('./weights/ResNet34_change_model'+str(100)+'.pth')
            model.load_state_dict(model_ckpt['state_dict'])
            # pretrain_model = torch.load('./weights/federate/base.pth')
            # print(pretrain_model)
            num = num + 1
            print("@@@@@@@@@@@"+str(task_num)+"@@@@@@@@@")
            #聚合
    if num % 4 == 0:
        task_num = task_num + 1

    if freeze:
        for name, param in model.named_parameters():
            if ("forget" not in name) and ("classifier" not in name):
                param.requires_grad = False
    return model


def origin_res32(weight_path, freeze):
    return OriginResNet34(weight_path, freeze)

def origin_pretrain_freeze() -> nn.Module:
    model =  OriginResNet34('./weights/resnet34-b627a593.pth', True)
    # model = models.densenet121(pretrained=False)
    # model.load_state_dict(torch.load("./weights/densenet121.pth"))
    # model.classifier.out_features = 10
    # model = LeNet()
    # model.load_state_dict(torch.load("./weights/fashionMnist.pth")['state_dict'])
    # for name, param in model.named_parameters():
    #     if ("classifer" not in name):
    #         param.requires_grad = False
    global num
    global task_num
    if num % 25 < 5:
        print("执行origin_pretrain_freeze")
        num =num + 1
    else:
        # 聚合
        model_ckpt = torch.load('./weights/densenet_5/denseNet_pretrain_unfreeze'+str(task_num)+'.pth')
        model.load_state_dict(model_ckpt['state_dict'])
        # pretrain_model = torch.load('./weights/federate/base.pth')
        # print(pretrain_model)
        num = num + 1
        print("@@@@@@@@@@@" + str(task_num) + "@@@@@@@@@")
        #聚合
    if num % 25 == 0:
        task_num = task_num + 1
    return model


def origin_pretrain_unfreeze() -> nn.Module:
    # model =  OriginResNet34('./weights/resnet34-b627a593.pth', False)
    model = models.resnet34(pretrained = False)
    # model = models.densenet121(pretrained=False)
    # model = LeNet()
    # model.load_state_dict(torch.load("./weights/fashionMnist.pth")['state_dict'])
    model.fc.out_features = 200

    global num
    global task_num
    if num % 25 < 5:
        print("执行origin_pretrain_UNFREE")
        num = num + 1
    else:
        # 聚合
        model_ckpt = torch.load('./weights/densenet_5/denseNet_pretrain_unfreeze'+str(task_num)+'.pth')
        model.load_state_dict(model_ckpt['state_dict'])
        # pretrain_model = torch.load('./weights/federate/base.pth')
        # print(pretrain_model)
        # 聚合
    if num % 25 == 0:
        task_num = task_num + 1
    return model


def origin_unpretrain() -> nn.Module:
    model =  OriginResNet34(None, False)
    global num
    if num < 3:
        print("执行origin_unpretrain")
        num = num + 1
    else:
        # 聚合
        model_ckpt = torch.load('./weights/federate/base5.pth')
        model.load_state_dict(model_ckpt['state_dict'])
        # pretrain_model = torch.load('./weights/federate/base.pth')
        # print(pretrain_model)
        # 聚合
    return model
