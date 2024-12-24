import os.path
import sys
import time
from torchsummary import summary
import torch
from torch import nn
from torch.utils.data import DataLoader
from thop import profile
from thop import clever_format
from config.log_config import CustomLogger
from container.dataloader_container import DataloaderContainer
from container.model_container import ModelContainer
from container.runtime_container import TrainContainer
from util.evaluate_func import accuracy
from util.misc import AverageMeter, mkdir
import time

def parse_time(seconds):
    minute, second = divmod(seconds, 60)
    hour, minute = divmod(minute, 60)
    return hour, minute, second

class ClassifyTrainContainer(TrainContainer):
    def __init__(self, i, args: dict):
        super().__init__(args)

        self.epoch = args['epochs']
        self.save_dir = os.path.join(args['save_dir'], args['project_name'])
        self.save_freq = args['save_freq']
        self.save_best = args['save_best']
        self.print_freq = args['print_freq']
        self.model_container = ModelContainer(args['model'])
        self.dataloader_container = DataloaderContainer(i,args['dataloader'])
        mkdir(self.save_dir)
        if self.save_freq == 0:
            self.save_freq = sys.maxsize

    def adjust_learning_rate(self):
        for scheduler in self.model_container.schedulers:
            scheduler.step()

    def train(self, train_loader, model, loss_func, optimizer, epoch):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to train mode
        model.train()

        # # FLOPs
        # oos = torch.randn(1, 3, 224, 224).to(self.device)
        # flops, params = profile(model, inputs=(oos,))  # “问号”内容使我们之后要详细讨论的内容，即是否为FLOPs
        # flops, params = clever_format([flops * 2, params], "%.3f")
        # print(flops, params)

        end = time.time()
        for i, (input, target) in enumerate(train_loader):
            input, target = input.to(self.device), target.to(self.device)
            # measure dataset loading time
            data_time.update(time.time() - end)



            # compute output
            output = model(input)
            loss = loss_func(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            output.cpu()
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.print_freq == 0:
                self.logger.warning('Epoch: [{0}][{1}/{2}]\t'
                                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format
                                    (epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time,
                                     loss=losses, top1=top1, top5=top5))

        self.logger.critical('Epoch: [{0}][Finished]\t'
                             'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                             'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                             'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                             'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                             'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format
                             (epoch, batch_time=batch_time, data_time=data_time, loss=losses, top1=top1, top5=top5))

    def validate(self, val_loader: DataLoader, model, loss_func):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        model.eval()

        with torch.no_grad():
            end = time.time()
            for i, (input, target) in enumerate(val_loader):
                input, target = input.to(self.device), target.to(self.device)

                # compute output
                output = model(input)
                loss = loss_func(output, target)

                # measure accuracy and record loss
                prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
                losses.update(loss.item(), input.size(0))
                top1.update(prec1.item(), input.size(0))
                top5.update(prec5.item(), input.size(0))
                output.cpu()
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % self.print_freq == 0:
                    self.logger.warning('Test: [{0}/{1}]\t'
                                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                                        'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format
                                        (i, len(val_loader), batch_time=batch_time, loss=losses, top1=top1, top5=top5))

        self.logger.critical('Test: * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
                             .format(top1=top1, top5=top5))
        return top1.avg


    def run(self, i,dataloader: DataLoader=None,test_dataloader: DataLoader=None):
        my_model = self.model_container.model.to(self.device)
        print(my_model)
        #summary(my_model, (3, 64, 64))
        my_optimizer = self.model_container.optimizer

        self.log_parameter()
        self.logger.critical("Training Task -- {} -- Start".format(self.title))
        self.logger.critical("Using {} device".format(self.device))

        my_loss_fn = nn.CrossEntropyLoss()
        if i == 0:
            training_data_loader = self.dataloader_container.train
            test_data_loader = self.dataloader_container.validate
        else:
            training_data_loader = dataloader
            test_data_loader = test_dataloader

        best_prec1 = 0.00
        prec1 = 0.00
        best_prec1_epoch = -1
        start_time = time.time()
        for epoch in range(0, self.epoch):
            # train for one epoch
            self.train(training_data_loader, my_model, my_loss_fn, my_optimizer, epoch)
            # adjust learning rate
            self.adjust_learning_rate()
            # evaluate on validation set

            if epoch == self.epoch - 1:
                end_time = time.time()
                print('project training cost %d hours %02d minutes %02d seconds' % parse_time(end_time - start_time))
                prec1 = self.validate(test_data_loader, my_model, my_loss_fn)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)

            if is_best:
                best_prec1_epoch = epoch
                if self.save_best:
                    file_name = f'{self.title}-best.pth'
                    path = os.path.join(self.save_dir, file_name)
                    self.model_container.save_model(path, addon={'epoch': epoch})
            if epoch != 0 and epoch % self.save_freq == 0:
                file_name = f'{self.title}-epoch{epoch}.pth'
                path = os.path.join(self.save_dir, file_name)
                self.model_container.save_model(path, addon={'epoch': epoch})
        CustomLogger().get_logger('summary').critical(f' Task {self.title} Finish Training '
                                                      f'with best predication={best_prec1}% '
                                                      f'at epoch {best_prec1_epoch}')
        self.logger.critical(f'Finish Training, best predication={best_prec1}% at epoch {best_prec1_epoch}')
        return my_model, training_data_loader ,test_data_loader