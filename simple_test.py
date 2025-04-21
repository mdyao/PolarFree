import os
import os.path as osp
import time
import torch
import logging
import argparse
from collections import OrderedDict

from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_env_info, get_root_logger, make_exp_dirs
from basicsr.utils.options import dict2str
from polarfree.utils.options import parse_options


def simple_test():
    """A simple test function to run inference using a pre-trained model."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to the test configuration file')
    parser.add_argument('-gpu_id', type=int, default=0, help='GPU device ID')
    args = parser.parse_args()
    
    # Set GPU device
    torch.cuda.set_device(args.gpu_id)
    
    # Set command-line arguments so parse_options can correctly read them
    import sys
    sys.argv = [sys.argv[0], '-opt', args.opt]
    
    # Parse config options
    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    opt, _ = parse_options(root_path, is_train=False)
    
    # Create directories for saving test results
    make_exp_dirs(opt)
    
    # Setup logger
    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{time.strftime('%Y%m%d_%H%M%S')}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))
    
    # Create test datasets and dataloaders
    test_loaders = []
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        test_set = build_dataset(dataset_opt)
        test_loader = build_dataloader(
            test_set,
            dataset_opt,
            num_gpu=opt['num_gpu'],
            dist=opt['dist'],
            sampler=None,
            seed=opt['manual_seed'])
        logger.info(f'Number of images in test dataset {dataset_opt["name"]}: {len(test_set)}')
        test_loaders.append(test_loader)
    
    # Build the model
    model = build_model(opt)
    
    # Run inference
    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info(f'Starting testing on {test_set_name}...')
        model.validation(
            test_loader,
            current_iter=opt.get('current_iter', 0),
            tb_logger=None,
            save_img=opt['val']['save_img']
        )
        logger.info(f'Finished testing on {test_set_name}!')
    
    logger.info('All testing finished!')


if __name__ == '__main__':
    simple_test()
