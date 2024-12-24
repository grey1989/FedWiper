from abc import abstractmethod

import torch

from config.log_config import CustomLogger


class RuntimeContainer:
    def __init__(self, args: dict):
        self.args: dict = args
        self.device = "cuda:1" if torch.cuda.is_available() else "cpu"

        self.title = args['title']
        self.logger = CustomLogger().get_logger(self.title)

    @abstractmethod
    def run(self):
        pass

    def mapping2str(self, mapping: dict, prefix: str = " " * 4, level: int = 0):
        """Log training configuration

             Args:
                mapping (dict)
                prefix (str) 缩进符号
                level (int) 缩进的层级
        """
        sub_lvl = level + 1
        sub_prefix = prefix * sub_lvl

        for key, value in mapping.items():
            sub_item = sub_prefix + key + ": "
            if isinstance(value, dict):
                # 进入下一字典层级中
                self.logger.critical(sub_item)
                self.mapping2str(value, prefix, sub_lvl)
            elif isinstance(value, list):
                # 遍历列表
                for item in value:
                    if isinstance(item, dict):
                        self.logger.critical(sub_item)
                        self.mapping2str(item, prefix, sub_lvl)
                    else:
                        self.logger.critical(sub_item + str(value))
                        break
            else:
                # 对于非字典、列表，直接转化为字符串
                self.logger.critical(sub_item + str(value))

    def log_parameter(self):
        self.logger.critical("Task -- {} --  parameter as follow".format(self.title))
        self.mapping2str(self.args, prefix='-' * 4)


class TestContainer(RuntimeContainer):
    def __init__(self, args: dict):
        super().__init__(args)

    # todo this function should not be here
    def resume_from_disk(self, model, weights):
        net_dict = model.state_dict()
        # 寻找网络中公共层，并保留预训练参数
        state_dict = {k: v for k, v in weights.items() if k in net_dict.keys()}
        # 将预训练参数更新到新的网络层
        net_dict.update(state_dict)
        # 加载预训练参数
        model.load_state_dict(net_dict)

    @abstractmethod
    def run(self):
        pass


class TrainContainer(RuntimeContainer):
    def __init__(self, args: dict):
        super().__init__(args)

    @abstractmethod
    def run(self):
        pass
