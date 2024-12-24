import torchvision
from torch.utils.data import DataLoader
import copy
from custom import load_dataset


def _load_dataloader(args: dict):
    args = copy.deepcopy(args)
    # dataloader_name = args.pop('type')
    dataset_conf = args.pop('dataset').copy()
    dataset_name = dataset_conf.pop('type')
    transforms_list = []
    transforms_conf_list = args.pop('transform', None)
    if transforms_conf_list is not None:
        for transform_conf in transforms_conf_list:
            transform_name = transform_conf.pop('type')
            try:
                transform = getattr(torchvision.transforms, transform_name)
                transforms_list.append(transform(**transform_conf))
            except AttributeError:
                raise ModuleNotFoundError(f'unknown transform {transform_name}')
    if len(transforms_list) > 0:
        dataset_conf['transform'] = torchvision.transforms.Compose(transforms_list)
    try:
        dataset = getattr(load_dataset, dataset_name)
        dataset = dataset(**dataset_conf)
        return DataLoader(dataset=dataset, **args)
    except AttributeError:
        raise ModuleNotFoundError(f'unknown dataset {dataset_name}!')


class DataloaderContainer:

    def __init__(self,i, args: dict):
        self.args: dict = args
        if 'train' in args and i==0:
            self.train = _load_dataloader(args['train'])
        if 'validate' in args and i==0:
            self.validate = _load_dataloader(args['validate'])
        if 'test' in args:
            self.test = _load_dataloader(args['test'])

