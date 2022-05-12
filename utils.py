from datetime import datetime
import json
import logging
from logging import Logger
import numpy as np
import os
import random
import sys
import torch


def read_data(path: str):
    with open(path, mode='r', encoding='utf-8') as f:
        lines = f.read().split('\n')
        if lines[-1] == '':
            lines = lines[:-1]

    return lines


def save_data(path: str, name: str, data):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, name), mode='w', encoding='utf-8') as f:
        for line in data:
            f.write(line)
            f.write('\n')


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def reset_logger(logger: Logger):
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    for f in logger.filters[:]:
        logger.removeFilter(f)


def log_json(file, data):
    f = open(file, mode='w', encoding='utf-8')
    json.dump(vars(data), f, ensure_ascii=False, indent=2)
    f.close()


def init_logger(args):
    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_formatter = logging.Formatter('%(asctime)s [%(threadName)-12.12s [%(levelname)-5.5s] %(message)s')
    logger = logging.getLogger()
    reset_logger(logger)
    path = os.path.join(args.saved_data_path, time)
    log_path = os.path.join(path, 'log')
    os.makedirs(path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    log_arguments(log_path, args)

    file_handler = logging.FileHandler(os.path.join(log_path, 'all.log'))
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)

    return logger, path


def log_arguments(log_path, args):
    path = os.path.join(log_path, 'arguments.json')
    log_json(path, args)
