from prettytable import PrettyTable
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn.parallel
import numpy as np
import time
import os.path as op

# import clip
from datasets import build_dataloader
from processor.processor import do_inference
from utils.checkpoint import Checkpointer
from utils.logger import setup_logger
from model import build_model
from utils.metrics import Evaluator
import argparse
from utils.iotools import load_train_configs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TranTextReID Text")
    # parser.add_argument("--config_file",
    #                     default='/data1/Code/fengyy/sketch/multimodality-CUHK/CUHK-PEDES/20240904_092003_sketch2_add-fusion-twofocal-1-35-fusion-itcloss_05kl-text-label/configs.yaml')
    parser.add_argument("--config_file",
                        default='/data1/Code/fengyy/sketch/multimodality-CUHK/CUHK-PEDES/20240905_204912_sketch2_add-fusion-twofocal-1-35-fusion-itcloss_05kl-text-label/configs.yaml')
    args = parser.parse_args()
    args = load_train_configs(args.config_file)

    if hasattr(args, 'advance') is False:
        args.advance = False
        args.add_token_num = 5

    args.training = False
    logger = setup_logger('CLIP2ReID', save_dir=args.output_dir, if_train=args.training)
    logger.info(args)
    device = "cuda"

    test_img_loader, test_txt_loader, test_sketch_loader = build_dataloader(args)
    model = build_model(args)
    checkpointer = Checkpointer(model)
    checkpointer.load(f=op.join(args.output_dir, 'text_best.pth'))
    model.to(device)

    do_inference(args, model, test_img_loader, test_txt_loader, test_sketch_loader)
