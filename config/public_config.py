import copy
import time
from collections import OrderedDict

import yaml

from util.misc import SingletonDecorator


@SingletonDecorator
class Config:
    def __init__(self, path):
        with open(path, encoding='utf-8', mode='r') as f:
            loader, _ = ordered_yaml()
            opt = yaml.load(f, Loader=loader)
            self.LOG = opt['log']
            self.Project = ProjectConfig(opt['project'])


@SingletonDecorator
class ProjectConfig:
    train_tasks = []
    test_tasks = []

    def __init__(self, dictionary: dict):
        project_name = dictionary.pop('name')
        if project_name is None or project_name == '%time':
            project_name = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        self.project_name = project_name
        self.templates = []
        if dictionary.get('template') is not None:
            for item in dictionary['template']:
                self.templates.append((item['name'], item['conf']))
        for item in dictionary['train']:
            # train_task = TrainTaskConfig(item)
            # self.tasks.append(train_task)
            template_name = item.pop('template_name', None)
            if template_name is not None:
                template = self.get_template(template_name)
                update_dict(template, item)
                self.train_tasks.append(template)
            else:
                self.train_tasks.append(item)
        for item in dictionary['test']:

            template_name = item.pop('template_name', None)
            if template_name is not None:
                template = self.get_template(template_name)
                update_dict(template, item)
                self.test_tasks.append(template)
            else:
                self.test_tasks.append(item)

    def get_template(self, name: str):
        for (key, value) in self.templates:
            if key == name:
                return copy.deepcopy(value)


def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representative(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representative)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def update_dict(origin: dict, updates: dict):
    """
    Recursively complete dictionary update
    :param origin: origin dictionary
    :param updates: dictionary with new value
    """
    child_dict = []
    for key, value in updates.items():
        if isinstance(value, dict):
            child_dict.append((key, value))
        else:
            origin[key] = value

    if len(child_dict) > 0:
        for (key, value) in child_dict:
            if origin.get(key) is None:
                origin[key] = {}
            update_dict(origin[key], value)
