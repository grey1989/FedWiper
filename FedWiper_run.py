import argparse
import sys
import time
import torch
import collections
from module.module import MyRes
import glob
from config.log_config import CustomLogger
from config.public_config import Config
from container.test_container import VotingTestContainer
from container.train_container import ClassifyTrainContainer
import os
def parse_time(seconds):
    minute, second = divmod(seconds, 60)
    hour, minute = divmod(minute, 60)
    return hour, minute, second

def resume_from_disk( model, weights):
    net_dict = model.state_dict()
    # 寻找网络中公共层，并保留预训练参数
    state_dict = {k: v for k, v in weights.items() if k in net_dict.keys()}
    # 将预训练参数更新到新的网络层
    net_dict.update(state_dict)
    # 加载预训练参数
    model.load_state_dict(net_dict)
# def main():
#     model_path = './weights/cifar10-10-5000-500'
#     file_list = glob.glob(os.path.join(model_path, '*.pth'))
#     model_list = []
#     for file in file_list:
#         model = MyRes()
#         weights = torch.load(file)['state_dict']
#         resume_from_disk(model, weights)
#         model_list.append(model)
#     worker_state_dict = [x.state_dict() for x in model_list]
#     weight_keys = list(worker_state_dict[0].keys())
#     fed_state_dict = collections.OrderedDict()
#     for key in weight_keys:
#         key_sum = 0
#         for i in range(len(model_list)):
#             key_sum = key_sum + worker_state_dict[i][key]
#         fed_state_dict[key] = key_sum / len(model_list)
#     #### update fed weights to fl model
#     model_path = './weights/cifar10-5-10000-1000'
#     # model_all = MyRes()
#     # resume_from_disk(model_all,fed_state_dict)
#     torch.save({'state_dict': fed_state_dict},
#                os.path.join(model_path, 'changed_model_all.pth'))

def main():
    task_num = 0
    # for y in range(1):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-path', type=str, required=False, default='./run.yaml', help='Path to option YAML file.')
    args = parser.parse_args(sys.argv[1:])
    config = Config(path=args.path)

    config.LOG['project_name'] = config.Project.project_name
    # init logger
    CustomLogger(config.LOG)
    logger = CustomLogger().get_logger('summary')
    start_time = time.time()


    logger.critical(f'project training start at {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))}')

    model_list = []
    dataloder_list = []
    test_dataloader = None
    for i in range(5):
        model_list.clear()
        count = 0
        print(i)
        for item in config.Project.train_tasks:
            item['project_name'] = config.Project.project_name
            item['dataloader']['train']['dataset']['number'] = count
            print(item['dataloader']['train']['dataset']['number'])
            print("!!!!!!!!!!!!")
            a = ClassifyTrainContainer(i,item)
            if i==0:
                model, traindataloader, test_dataloader = a.run(i)
                dataloder_list.append(traindataloader)
            else:
                model,_,_ = a.run(i,dataloder_list[count],test_dataloader)
            torch.save({'state_dict': model.state_dict()},
                       os.path.join("./weights/ResNet34_UniAdapter/", 'ResNet34_change_model' + str(count) + '.pth'))
            model_list.append(model)
            count = count+1
        # 聚合
        model_path = './weights/ResNet34_UniAdapter'
        worker_state_dict = [x.state_dict() for x in model_list]
        weight_keys = list(worker_state_dict[0].keys())
        fed_state_dict = collections.OrderedDict()
        for key in weight_keys:
            key_sum = 0
            for i in range(len(model_list)):
                key_sum = key_sum + worker_state_dict[i][key]
            fed_state_dict[key] = key_sum / len(model_list)
        #### update fed weights to fl model
        torch.save({'state_dict': fed_state_dict},
                   os.path.join(model_path, 'ResNet34_change_model'+str(100)+'.pth'))
    # task_num = task_num + 2
    end_time = time.time()
    logger.critical(f'project training finish at {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))}')
    logger.critical('project training cost %d hours %02d minutes %02d seconds' % parse_time(end_time - start_time))





    logger.critical(f'project testing start at {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
    for i in range(1):
        for item in config.Project.test_tasks:
            #item['model_dir'] = './weights/checkpoint/' + config.Project.project_name
            item['model_dir'] = './weights/CIFAR100-ResNet34/'
            item['project_name'] = config.Project.project_name
            test_container = VotingTestContainer(i,item)
            test_container.run()
    end_time = time.time()
    logger.critical(f'project testing finish at {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))}')
    logger.critical('project total cost %d hours %02d minutes %02d seconds' % parse_time(end_time - start_time))


if __name__ == '__main__':
    main()
