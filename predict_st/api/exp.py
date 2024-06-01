import sys
import time
import os.path as osp
from fvcore.nn import FlopCountAnalysis, flop_count_table

import torch

from predict_st.methods import method_maps
from predict_st.datasets import BaseDataModule
from predict_st.utils import (get_dataset, measure_throughput, SetupCallback, EpochEndCallback, BestCheckpointCallback)

import argparse
from pytorch_lightning import seed_everything, Trainer
import pytorch_lightning.callbacks as plc

class BaseExperiment(object):
    def __init__(self, args, dataloaders=None, strategy='ddp'):
        self.args = args
        self.config = self.args.__dict__
        self.method = None
        self.args.method = self.args.method.lower()
        self._dist = self.args.dist

        base_dir = args.res_dif if args.res_dif is not None else 'work_dirs'
        save_dir = osp.join(base_dir, args.ex_name if not args.ex_name.startswith(args.res_dir) else args.ex_name.split(args.res_dir+'/')[-1])
        ckpt_dir = osp.join(save_dir, 'checkpoints')


        seed_everything(args.seed)

