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
        self.data = self._get_data(dataloaders)
        self.method = method_maps[self.args.method](steps_per_epoch=len(self.data.train_loader), test_mean=self.data.test_mean, test_std=self.data.test_std, save_dir=save_dir, **self.config)
        callbacks, self.save_dir = self._load_callbacks(args, save_dir, ckpt_dir)
        self.trainer = self._init_trainer(self.args, callbacks, strategy)

    def _init_trainer(self, args, callbacks, strategy):
        return Trainer(devices=args.gpus,
                       max_epochs=args.epoch,
                       strategy=strategy,
                       accelerator='gpu',
                       callbacks=callbacks)

    def _load_callbacks(self, args, save_dir, ckpt_dir):
        method_info = None
        if self._dist == 0:
            if not self.args.no_display_method_info:
                method_info = self.display_method_info(args)
                '''need update'''

    def _get_data(self, dataloaders=None):
        if dataloaders is None:
            train_loader, vali_loader, test_loader = get_dataset(self.args.dataname, self.config)
        else:
            train_loader, vali_loader, test_loader = dataloaders

        vali_loader = test_loader if vali_loader is None else vali_loader
        return BaseDataModule(train_loader, vali_loader, test_loader)




