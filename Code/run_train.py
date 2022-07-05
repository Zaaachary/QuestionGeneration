import argparse
import os
import logging
from statistics import mode
import time
from collections import OrderedDict
import torch
from functools import partial

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, GradientAccumulationScheduler
import numpy as np

from model_wrapper import Generation_Model
from data import DuReaderQG_Dataset

logging.basicConfig(level=logging.INFO)  
logger = logging.getLogger('')


def gpus_parser(gpus):
    """
    match input type with pytorch-lightning type
    "6,7" -> [6, 7]; "6" -> [6,]; "-1" -> 0
    """
    accelerator = None
    if gpus == "-1":    # no cpu
        gpus = 0
    elif "," in gpus:   # muliti gpu
        gpus = gpus.split(",")
        if "" in gpus:
            gpus.remove("")
        gpus = list(map(int, gpus))
        accelerator = "ddp"
    else:               # single gpu
        gpus = [int(gpus),]
    return gpus, accelerator

def get_version_name(args):
    version_name = ''
    gpus, _ = gpus_parser(args.gpus)
    gpu_num = len(gpus) if isinstance(gpus, list) else 1
    # torch.cuda.get_device_name(0) -> 1050Ti
    version_name += f"bz={gpu_num}x{args.train_batch_size_per_gpu}x{args.gradient_accumulation_step}"
    version_name += f"_ep={args.epoch}_lr={args.learning_rate}_ae={args.adam_epsilon}_seed={args.seed}"

    return version_name

def set_logger(args, version_name):
    tblogger = TensorBoardLogger(
        args.output_path,
        name=args.task_name,
        version=version_name
    )
    root = os.path.join(args.output_path, args.task_name)
    if not os.path.exists(root):
        os.mkdir(root)
    root = os.path.join(root, version_name)
    if not os.path.exists(root):
        os.mkdir(root)
    
    handler = logging.FileHandler(os.path.join(root, f'train.{time.strftime("%Y-%m-%d.%H:%M:%S")}.log'))
    formatter = logging.Formatter(fmt='%(asctime)s - %(filename)s[line:%(lineno)d]: %(message)s', datefmt='%y/%m/%d %H:%M')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return tblogger, root

def main(args):
    gpus, accelerator = gpus_parser(args.gpus)
    version = get_version_name(args)
    tblogger, output_root = set_logger(args, version)
    logger.info(str(args))

    seed_everything(args.seed)
    # GPT2 model & tokenizer init
    model = Generation_Model(
        PTM_name_or_path=args.PTM_name_or_path,
        learning_rate=args.learning_rate,
        adam_epsilon=args.adam_epsilon,
        warmup_proportion=args.warmup_proportion,
        weight_decay=args.weight_decay,
        train_batch_size_pre_device=args.train_batch_size_per_gpu,
        args_str=str(args),
    )
    tokenizer = model.get_tokenizer()
    # dataset -> dataloader
    train_dataset = DuReaderQG_Dataset(
        args.dataset_path, 
        args.max_src_len, args.max_tgt_len,
        tokenizer=tokenizer, dataset_type='train', evaluate=False)
    train_dataloader = train_dataset.make_dataloader(
        batch_size=args.train_batch_size_per_gpu)

    model.set_example_num(len(train_dataset))

    dev_dataset = DuReaderQG_Dataset(
        args.dataset_path, 
        args.max_src_len, args.max_tgt_len,
        tokenizer=tokenizer, dataset_type='dev', evaluate=False)
    dev_dataloader = dev_dataset.make_dataloader(
        batch_size=args.dev_batch_size_per_gpu)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        every_n_epochs=1,
        filename="{epoch:02d}-{step}-{val_loss:.6f}",
        save_top_k=2,
        mode="min",
        save_weights_only=True  # params only, without optimizer state
    )

    accumulator = GradientAccumulationScheduler(
        scheduling={0: args.gradient_accumulation_step})

    trainer = Trainer(
        max_epochs=args.epoch,
        val_check_interval=0.1,
        gpus=gpus,
        accelerator=accelerator,
        # fast_dev_run=True,  # enable when debug
        deterministic=True,
        default_root_dir=args.output_path,
        logger=tblogger,
        precision=16 if args.fp16 else 32,
        callbacks=[checkpoint_callback, accumulator]
    )
    # trainer.fit(model, train_dataloader)
    trainer.fit(model, train_dataloader, dev_dataloader)
    logger.info('finished')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # model path and name
    parser.add_argument("--PTM_name_or_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--task_name", type=str, required=True,
                        help='feature in input/model level')

    # dataset
    parser.add_argument("--max_src_len", type=int, default=None,
        help='Bart source_length, GPT2 seq_len')
    parser.add_argument("--max_tgt_len", type=int, default=None,
        help='Bart target')

    # hparams & device
    parser.add_argument("--epoch", default=1, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument("--warmup_proportion", default=0.1, type=float)
    parser.add_argument("--train_batch_size_per_gpu", default=2, type=int)
    parser.add_argument("--gradient_accumulation_step", default=1, type=int)
    parser.add_argument("--dev_batch_size_per_gpu", default=2, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--gpus", type=str,
                        help="-1:not use; x:card id; [6,7]: use card 6 and 7")
    
    args = parser.parse_args()
    main(args)
