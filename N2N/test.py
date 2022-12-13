import os
import glob
from pathlib import Path
import pandas as pd

import argparse
import re
import numpy as np
from transformers import AutoTokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import torch
from torch import nn
import torch.optim as optim
from argparse import ArgumentParser
import transformers
from transformers import AutoModel, EarlyStoppingCallback, AutoConfig
from utils.load_data import load_data
from data_loader.data_loader import DPDataset
from model.model import Model
from trainer.trainer import Trainer

from omegaconf import OmegaConf


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="base_config")
    args, _ = parser.parse_known_args()

    cfg = OmegaConf.load(f'./config/{args.config}.yaml')

    train_data, train_label = load_data(cfg.path.train_path)
    valid_data, _           = load_data(cfg.path.valid_path)
    test_data, test_label   = load_data(cfg.path.test_path)
    labels = train_label | test_label
    labels = ["[PAD]"] + list(labels)
    label_vocab = {label : id for id, label in enumerate(labels)}

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)

    train_dataset = DPDataset(tokenizer,train_data, label_vocab,max_len = cfg.model.max_len)
    valid_dataset = DPDataset(tokenizer,valid_data, label_vocab, max_len = cfg.model.max_len)
    test_dataset = DPDataset(tokenizer, test_data, label_vocab, max_len = cfg.model.max_len)


    model = torch.load(cfg.test.best_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model = model.to(device)

    training_args = TrainingArguments(
        output_dir=cfg.train.output_dir, # output directory
        num_train_epochs=cfg.train.max_epoch,
        learning_rate=cfg.train.learning_rate,  # learning_rate
        per_device_train_batch_size=cfg.train.batch_size,  # batch size per device during training
        per_device_eval_batch_size=cfg.train.batch_size,  # batch size for evaluation
        load_best_model_at_end = cfg.train.load_best_model_at_end,
        evaluation_strategy = cfg.train.evaluation_strategy,
        save_strategy = cfg.train.save_strategy,
        logging_strategy = cfg.train.logging_strategy,
        logging_steps = cfg.train.logging_steps
    )    

    trainer = Trainer(
        label_vocab,
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=valid_dataset,  # evaluation dataset
        callbacks = [EarlyStoppingCallback(early_stopping_patience=2)]
    )

    trainer.evaluate(test_dataset)