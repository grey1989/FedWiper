import json
import os
import random

import PIL.Image
from torch.utils.data import Dataset

from util.misc import SingletonDecorator


class ResizedCIFAR10(Dataset):

    def __init__(self, root, train=True, transform=None):
        super(ResizedCIFAR10, self).__init__()
        self.train = train
        self.transform = transform

        root = os.path.join(root, 'cifar-10-resized')
        assert os.path.exists(root) is True
        # 如果是训练则加载训练集，如果是测试则加载测试集
        if self.train:
            file_annotation = os.path.join(root, 'annotations', 'train.json')
            img_folder = os.path.join(root, 'train')
        else:
            file_annotation = os.path.join(root, 'annotations', 'test.json')
            img_folder = os.path.join(root, 'test')
        fp = open(file_annotation, 'r')
        data_dict = json.load(fp)

        # 如果图像数和标签数不匹配说明数据集标注生成有问题，报错提示
        assert len(data_dict['images']) == len(data_dict['categories'])

        self.filenames = data_dict['images']
        self.labels = data_dict['categories']
        self.img_folder = img_folder

    def __getitem__(self, index):
        img_name = os.path.join(self.img_folder, self.filenames[index])
        label = self.labels[index]

        img = PIL.Image.open(img_name)
        if self.transform is not None:
            img = self.transform(img)  # 可以根据指定的转化形式对数据集进行转换

        # return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容
        return img, label

    def __len__(self):
        return len(self.filenames)


@SingletonDecorator
class SlidingWindowIndexGenerator:
    def __init__(self, count):
        x = [i for i in range(0, count)]
        #random.shuffle(x)
        self.x = x

    def get_range(self, num, window_size, overlap_size):
        # todo index exceed
        slice_step = window_size // overlap_size
        return self.x[num * overlap_size: num * overlap_size + window_size]
        # return self.x[40000:49999]
