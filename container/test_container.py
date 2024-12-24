import glob
import os
import random
from torchvision import models
import torch
from config.log_config import CustomLogger
from container.dataloader_container import DataloaderContainer
from container.runtime_container import TestContainer
from module.module import MyRes
from module.module import OriginResNet34
from util.evaluate_func import accuracy
from util.misc import AverageMeter
from denseNet_model import densenet121
from LeNet_adpter import LeNet,LeNet_adpter
class VotingTestContainer(TestContainer):

    def __init__(self, i, args: dict):
        super().__init__(args)
        self.models = []
        self.voter = args['voter']
        self.model_dir = args['model_dir']
        self.dataloaderContainer = DataloaderContainer(i,args['dataloader'])
        #self.model_dir = './weights/checkpoint/2022-11-28_04-10-31'
    def run(self):
        self.log_parameter()
        file_list = glob.glob(os.path.join(self.model_dir, '*.pth'))
        pre_train_weights = torch.load('./weights/resnet34-b627a593.pth')
        for file in file_list:
            weights = torch.load(file)['state_dict']
            model = MyRes()

            # model = models.densenet121(pretrained=False)
            # model = densenet121()
            # model = LeNet()
            # model.classifier.out_features = 100
            # model = OriginResNet34('./weights/resnet34-b627a593.pth', True)
            #self.resume_from_disk(model, pre_train_weights)
            self.resume_from_disk(model, weights)
            self.models.append(model)
        top1 = AverageMeter()
        top5 = AverageMeter()
        with torch.no_grad():
            for model in self.models:
                model.train()
                model.to(self.device)
                for i, (input, _) in enumerate(self.dataloaderContainer.test):
                    input = input.to(self.device)
                    model(input)
                model.eval()
            self.logger.critical(f'{len(self.models)} models are loaded')

            for i, (input, target) in enumerate(self.dataloaderContainer.test):
                input, target = input.to(self.device), target.to(self.device)
                out = None
                for model in random.sample(self.models, self.voter):
                    model.to(self.device)
                    output = model(input)
                    output = torch.nn.functional.softmax(output, dim=1)
                    if out is None:
                        out = output
                    else:
                        out += output
                out = torch.divide(out, self.voter)
                prec1, prec5 = accuracy(out.data, target, topk=(1, 5))

                top1.update(prec1.item(), input.size(0))
                top5.update(prec5.item(), input.size(0))
        self.logger.critical(
            'Prec@1 {top1.avg:.3f}\t'
            'Prec@5 {top5.avg:.3f}'.format
            (top1=top1, top5=top5))
        CustomLogger().get_logger('summary').critical(f' Task {self.title} Finish Testing '
                                                      f'with best predication top1={top1.avg:.3f}% '
                                                      f'top5= {top5.avg:.3f}')
