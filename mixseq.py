import os
import random
from tqdm import tqdm
from typing import List

from utils import read_data, save_data, init_logger


def resize(src_data: List[str], tgt_data: List[str], new_size: int):
    n_org_samples = len(src_data)
    if new_size > n_org_samples:
        for _ in tqdm(range(new_size - n_org_samples), total=new_size - n_org_samples, desc='Resize original data'):
            i = random.randint(0, n_org_samples - 1)
            src_data.append(src_data[i])
            tgt_data.append(tgt_data[i])
    elif new_size < n_org_samples:
        for _ in tqdm(range(n_org_samples - new_size), total=n_org_samples - new_size, desc='Resize original data'):
            i = random.randint(0, len(src_data) - 1)
            src_data.pop(i)
            tgt_data.pop(i)

    return src_data, tgt_data


def concat_sample(a: str, b: str, sep_token: str = '<sep>'):
    return a + sep_token + b


def generate_sample(src_data: List[str], tgt_data: List[str], aug_size: int, is_contextual: bool = False,
                    sep_token: str = '<sep>'):
    src_augmentation_data = []
    tgt_augmentation_data = []
    if is_contextual:
        for _ in tqdm(range(aug_size), total=aug_size, desc='Generate new sample'):
            i = random.randint(0, len(src_data) - 2)
            src_augmentation_data.append(concat_sample(src_data[i], src_data[i + 1], sep_token))
            tgt_augmentation_data.append(concat_sample(tgt_data[i], tgt_data[i + 1], sep_token))
    else:
        for _ in tqdm(range(aug_size), total=aug_size, desc='Generate new sample'):
            a = random.randint(0, len(src_data) - 1)
            b = random.randint(0, len(src_data) - 1)
            src_augmentation_data.append(concat_sample(src_data[a], src_data[b], sep_token))
            tgt_augmentation_data.append(concat_sample(tgt_data[a], tgt_data[b], sep_token))

    return src_augmentation_data, tgt_augmentation_data


def mix_seq(args):
    logger, path = init_logger(args)
    org_src_data = read_data(os.path.join(args.data_path, 'train.en'))
    org_tgt_data = read_data(os.path.join(args.data_path, 'train.vi'))
    assert len(org_src_data) == len(org_tgt_data)
    logger.info(f'Dataset: {args.data_name}')
    logger.info(f'Size of original dataset {len(org_src_data)}')

    aug_size = round(args.factor_mixseq * len(org_src_data))
    src_resized_data, tgt_resized_data = resize(org_src_data, org_tgt_data, aug_size)
    assert(len(src_resized_data) == len(tgt_resized_data))
    logger.info(f'Size of dataset after resized {len(src_resized_data)}')

    src_augmentation_data, tgt_augmentation_data = generate_sample(org_src_data, org_tgt_data, aug_size,
                                                                   is_contextual=args.is_contextual,
                                                                   sep_token=args.sep_token)
    assert(len(src_augmentation_data) == len(tgt_augmentation_data))
    logger.info(f'Size of augmentation dataset {len(src_augmentation_data)}')

    save_data(path, 'train.en', src_resized_data + src_augmentation_data)
    save_data(path, 'train.vi', tgt_resized_data + tgt_augmentation_data)
    logger.info(f'Size of final dataset {len(src_augmentation_data + src_resized_data)}')
