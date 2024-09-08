# --------------------------------------------------------
# LISA: Reasoning Segmentation via Large Language Model
# Licensed under Apache-2.0 license [see LICENSE for details]
# Authors: Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li, Yuhui Yuan, Shu Liu, Jiaya Jia
# --------------------------------------------------------
# GSVA: Generalized Segmentation via Multimodal Large Language Models
# Modified by Zhuofan Xia
# --------------------------------------------------------

import functools
import logging
import os
import sys

from termcolor import colored

logging.getLogger('PIL').setLevel(logging.WARNING)

@functools.lru_cache()
def get_logger(output_dir, dist_rank=0, name=''):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    # create console handlers for master process
    if dist_rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console_handler)
        file_handler = logging.FileHandler(os.path.join(output_dir, f'{name}_log.txt'), mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(file_handler)

    return logger
