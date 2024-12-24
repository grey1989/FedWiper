import torch

from custom import load_model
from module.base_module import BaseModule


class ModelContainer:
    def __init__(self, args):
        self.args: dict = args
        self.save_optimizer = False

        self.model = self._load_modal()
        self.optimizer = self._load_optimizer()
        self.schedulers = self._load_scheduler()

    def _load_modal(self) -> BaseModule:
        model_name = self.args['type']
        try:
            model_conf = self.args['param'].copy()
        except (KeyError, AttributeError):
            model_conf = {}
        try:
            model_fn = getattr(load_model, model_name)
            return model_fn(**model_conf)
        except AttributeError:
            raise ModuleNotFoundError(f'unknown model {model_name}!')

    def _load_scheduler(self):
        lr_scheduler_conf = self.args['lr_scheduler']
        schedulers = []
        for scheduler_conf in lr_scheduler_conf:
            scheduler_conf = scheduler_conf.copy()
            schedule_type = scheduler_conf.pop('type')
            try:
                scheduler = getattr(torch.optim.lr_scheduler, schedule_type)
                schedulers.append(scheduler(self.optimizer, **scheduler_conf))
            except AttributeError:
                raise ModuleNotFoundError(f'unknown lr_scheduler {schedule_type}!')
        return schedulers

    def _load_optimizer(self):
        optim_conf = self.args['optimizer'].copy()
        optim_type = optim_conf.pop('type')
        model_params = filter(lambda p: p.requires_grad, self.model.parameters())
        try:
            optim = getattr(torch.optim, optim_type)
            return optim(model_params, **optim_conf)
        except AttributeError:
            raise ModuleNotFoundError(f'unknown optimizer {optim_type}!')

    def save_model(self, path: str, addon: dict):
        if self.save_optimizer:
            state = {'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}
        else:
            state = {'model': self.model.state_dict()}
        state.update(addon)
        torch.save(state, path)

    def resume_model(self, path):
        weights = torch.load(path)['model']
        self.model.resume_from_disk(weights)

