from torch.utils.data import Subset
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import models, utils, datasets, transforms
import numpy as np
import sys
import os
from PIL import Image
from dataset.dataset import ResizedCIFAR10, SlidingWindowIndexGenerator

normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

#normalize = transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
def CIFAR10Set(train: bool):
    if train:
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transform = transforms.Compose([
            #transforms.Resize(224),
            transforms.ToTensor(),
            normalize,
        ])
    training_data = datasets.CIFAR10(
        root="./data",
        train=train,
        download=True,

        transform=transform,
    )
    return training_data




class TinyImageNet(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.Train = train
        self.root_dir = root
        self.transform = transform
        self.train_dir = os.path.join(self.root_dir, "train")
        self.val_dir = os.path.join(self.root_dir, "val")

        if (self.Train):
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_val()

        self._make_dataset(self.Train)

        words_file = os.path.join(self.root_dir, "words.txt")
        wnids_file = os.path.join(self.root_dir, "wnids.txt")

        self.set_nids = set()

        with open(wnids_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                self.set_nids.add(entry.strip("\n"))

        self.class_to_label = {}
        with open(words_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                words = entry.split("\t")
                if words[0] in self.set_nids:
                    self.class_to_label[words[0]] = (words[1].strip("\n").split(","))[0]

    def _create_class_idx_dict_train(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(self.train_dir, d))]
        classes = sorted(classes)
        num_images = 0
        for root, dirs, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".JPEG"):
                    num_images = num_images + 1

        self.len_dataset = num_images;

        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}

    def _create_class_idx_dict_val(self):
        val_image_dir = os.path.join(self.val_dir, "images")
        if sys.version_info >= (3, 5):
            images = [d.name for d in os.scandir(val_image_dir) if d.is_file()]
        else:
            images = [d for d in os.listdir(val_image_dir) if os.path.isfile(os.path.join(self.train_dir, d))]
        val_annotations_file = os.path.join(self.val_dir, "val_annotations.txt")
        self.val_img_to_class = {}
        set_of_classes = set()
        with open(val_annotations_file, 'r') as fo:
            entry = fo.readlines()
            for data in entry:
                words = data.split("\t")
                self.val_img_to_class[words[0]] = words[1]
                set_of_classes.add(words[1])

        self.len_dataset = len(list(self.val_img_to_class.keys()))
        classes = sorted(list(set_of_classes))
        # self.idx_to_class = {i:self.val_img_to_class[images[i]] for i in range(len(images))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}
        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}

    def _make_dataset(self, Train=True):
        self.images = []
        if Train:
            img_root_dir = self.train_dir
            list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]
        else:
            img_root_dir = self.val_dir
            list_of_dirs = ["images"]

        for tgt in list_of_dirs:
            dirs = os.path.join(img_root_dir, tgt)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if (fname.endswith(".JPEG")):
                        path = os.path.join(root, fname)
                        if Train:
                            item = (path, self.class_to_tgt_idx[tgt])
                        else:
                            item = (path, self.class_to_tgt_idx[self.val_img_to_class[fname]])
                        self.images.append(item)

    def return_label(self, idx):
        return [self.class_to_label[self.tgt_idx_to_class[i.item()]] for i in idx]

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        img_path, tgt = self.images[idx]
        with open(img_path, 'rb') as f:
            sample = Image.open(img_path)
            sample = sample.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, tgt


def SlicedCIFAR100_IMGSet(train: bool, number=0, window_size=0, overlap_size=0,transform=None):
    if train:
        transform = transforms.Compose([
            # transforms.RandomResizedCrop((224, 224), scale=(0.75, 1)),
            transforms.Resize(224),
            # transforms.RandomCrop(224, padding=24),
            # #transforms.AugMix(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            # normalize,
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            # normalize,
        ])
    # training_data = ResizedCIFAR10('./data',
    #                                train=train,
    #                                transform=transform, )
    training_data = datasets.CIFAR100(
        root="./data",
        train=train,
        download=True,

        transform=transform,
    )
    # data_dir = '../../xfLee/Datasets/TinyImagenet/tiny-imagenet-200/'
    # training_data = TinyImageNet(data_dir, train=True, transform=transform)
    # dataset_val = TinyImageNet(data_dir, train=False)
    if train:
        index = SlidingWindowIndexGenerator(count=50000).get_range(number, window_size, overlap_size)
        training_data = Subset(training_data, indices=index)
    return training_data


def CIFAR10_IMGSet(train: bool, transform=None):
    if train:
        transform = transforms.Compose([
            transforms.Resize(224),
            # transforms.RandomResizedCrop((224,224),scale=(0.75, 1)),
            # transforms.RandomCrop(224, padding=24),
            # transforms.AugMix(),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            normalize,
        ])
    # training_data = ResizedCIFAR10('./data',
    #                                train=train,
    #                                transform=transform, )
    training_data = datasets.CIFAR10(
        root="./data",
        train=train,
        download=True,

        transform=transform,
    )
    return training_data


def SlicedCIFAR10_IMGSet(train: bool, number=0, window_size=0, overlap_size=0, transform=None):
    # if train:
    #     transform = transforms.Compose([
    #         # transforms.Resize(224),
    #         transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         normalize,
    #     ])
    # else:
    #     transform = transforms.Compose([
    #         transforms.Resize(224),
    #         transforms.ToTensor(),
    #         normalize,
    #     ])
    training_data = datasets.CIFAR10(
        root="./data",
        train=train,
        download=True,

        transform=transform,
    )
    if train:
        index = SlidingWindowIndexGenerator(count=50000).get_range(number, window_size, overlap_size)
        training_data = Subset(training_data, indices=index)
    return training_data

def CIFAR100_IMGSet(train: bool, number=0, window_size=0, overlap_size=0,transform=None):
    if train:
        transform = transforms.Compose([
            transforms.Resize(224),
            #transforms.RandomResizedCrop((224,224),scale=(0.75, 1)),
            #transforms.RandomCrop(224, padding=24),
            #transforms.AugMix(),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            normalize,
        ])
    # training_data = ResizedCIFAR10('./data',
    #                                train=train,
    #                                transform=transform, )
    training_data = datasets.CIFAR100(
        root="./data",
        train=train,
        download=True,

        transform=transform,
    )
    return training_data

def SlicedMNIST_IMGSet(train: bool, number=0, window_size=0, overlap_size=0, transform=None):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    training_data = datasets.MNIST(
        root="./data/LeNet_data",
        train=train,
        download=True,

        transform=transform,
    )
    if train:
        index = SlidingWindowIndexGenerator(count=50000).get_range(number, window_size, overlap_size)
        training_data = Subset(training_data, indices=index)
    return training_data

def MNIST_IMGSet(train: bool, transform=None):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    training_data = datasets.MNIST(
        root="./data/LeNet_data",
        train=train,
        download=True,

        transform=transform,
    )
    return training_data