import logging
import os

from util.misc import SingletonDecorator

'''
logging.DEBUG,
logging.INFO,
logging.WARNING,
logging.ERROR,
logging.CRITICAL
'''


class LevelFilter(logging.Filter):
    def __init__(self, level):
        super().__init__()
        self.level = level

    def filter(self, record):
        return record.levelno >= self.level


@SingletonDecorator
class CustomLogger:

    def get_logger(self, task_name: str) -> logging.Logger:
        logger = logging.getLogger(task_name)

        log_path = self.log_path
        fmt = '%(asctime)s - %(message)s'
        format_str = logging.Formatter(fmt)

        if not os.path.exists(log_path):
            os.mkdir(log_path)
        filename = os.path.join(log_path, f'{self.project_name}-{task_name}.log')

        logger.setLevel(logging.DEBUG)
        # remove all handlers
        for handler in logger.handlers:
            logger.removeHandler(handler)
        # output to screen
        sh = logging.StreamHandler()
        sh.setFormatter(format_str)
        sh.addFilter(LevelFilter(logging.INFO))
        # output to file
        th = logging.FileHandler(filename=filename, encoding='utf-8')
        th.setFormatter(format_str)
        th.addFilter(LevelFilter(self.log_level))

        logger.addHandler(sh)
        logger.addHandler(th)
        return logger

    def __init__(self, args: dict):
        self.project_name = args.get('project_name')
        self.log_path = os.path.join(args['log_path'], self.project_name)
        level: str = args['log_level']
        if level.upper() == 'DEBUG':
            self.log_level = logging.DEBUG
        elif level.upper() == 'INFO':
            self.log_level = logging.INFO
        elif level.upper() == 'WARNING':
            self.log_level = logging.WARNING
        elif level.upper() == 'ERROR':
            self.log_level = logging.ERROR
        elif level.upper() == 'CRITICAL':
            self.log_level = logging.CRITICAL
        else:
            raise ModuleNotFoundError(f'Unknown level {level}')
