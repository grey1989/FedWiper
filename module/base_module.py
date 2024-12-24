from torch import nn


class BaseModule(nn.Module):
    def __init__(self):
        super().__init__()

    def get_save_state_dict(self):
        return self.state_dict()

    def resume_from_disk(self, weights):
        net_dict = self.state_dict()
        # 寻找网络中公共层，并保留预训练参数
        state_dict = {k: v for k, v in weights.items() if k in net_dict.keys()}
        # 将预训练参数更新到新的网络层
        net_dict.update(state_dict)
        # 加载预训练参数
        self.load_state_dict(net_dict)
