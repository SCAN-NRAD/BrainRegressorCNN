###############################################################################################################################
# Code related to the following publication:
# 
#     Rebsamen, M., Suter, Y., Wiest, R., Reyes, M., & Rummel, C. (2020)
#     Brain morphometry estimation: From hours to seconds using deep learning.
#     Frontiers in Neurology, 11, 244. https://doi.org/10.3389/fneur.2020.00244
# 
# 
###############################################################################################################################

import os
import argparse
import shutil
import timeit
import logging

from train import training
from config import Configuration

logger = logging.getLogger()


def main(config_file: str, scratch_dir: str, num_workers: int):
    if not os.path.exists(config_file):
        print('NOTE: Configuration {} not found. Writing new file with defaults.'.format(config_file))
        Configuration().save(config_file)

    cfg = Configuration.load(config_file)
    if not os.path.exists(cfg.hdf_file):
        print('Error: dataset {} not found'.format(cfg.hdf_file))
        exit(1)

    trainer = training.Trainer(cfg, num_workers)

    # setup logger
    if not os.path.exists(trainer.checkpoint_dir):
        os.makedirs(trainer.checkpoint_dir)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(levelname)-6s %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    file_handler = logging.FileHandler(trainer.checkpoint_dir + '.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if scratch_dir is not None:
        scratch_path = os.path.join(scratch_dir, os.path.basename(cfg.hdf_file))
        if os.path.exists(scratch_path):
            logger.info('Found existing dataset on scratch %s. Using this one.', scratch_path)
        else:
            shutil.copy(cfg.hdf_file, scratch_path)
        cfg.hdf_file = scratch_path
        logger.info('Using dataset on scratch: %s', scratch_path)

    # start training
    total_start_time = timeit.default_timer()
    trainer.train()
    logger.info('Total time: {:.0f}s'.format(timeit.default_timer() - total_start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model to learn brain morphometry directly from MRI.')

    parser.add_argument(
        '--scratch_dir',
        type=str,
        help='If set, copy <dataset>.h5 to this temporary directory at the beginning.'
    )

    parser.add_argument(
        '--loader_num_workers',
        type=int,
        default=1,
        help='Number of workers in DataLoader to fetch batches (default 1).'
    )

    parser.add_argument(
        'config',
        type=str,
        default='config.json',
        help='Path to configuration file.'
    )

    args = parser.parse_args()
    main(args.config, args.scratch_dir, args.loader_num_workers)
