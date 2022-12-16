import argparse

import torch

from transformers import AutoTokenizer, EarlyStoppingCallback, TrainingArguments
from utils.load_data import load_data
from utils.convert_examples_to_features import convert_examples_to_features
from data_loader.data_loader import Dataset
from model.model import Model
from trainer.trainer import Trainer

from omegaconf import OmegaConf


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="base_config")
    args, _ = parser.parse_known_args()

    cfg = OmegaConf.load(f'./config/{args.config}.yaml')

    train_data, test_data = load_data()
    tokenizer = AutoTokenizer.from_pretrained("skt/kogpt2-base-v2",bos_token = "<s>", eos_token = "</s>", pad_token = "<pad>")
    max_seq_len = 128
    train_X, train_y = convert_examples_to_features(train_data["document"], train_data["label"], max_seq_len = cfg.model.max_len, tokenizer = tokenizer)
    test_X, test_y = convert_examples_to_features(test_data["document"], test_data["label"], max_seq_len = cfg.model.max_len, tokenizer = tokenizer)

    model = Model(cfg.model.model_name,tokenizer)

    train_dataset = Dataset(train_X, train_y)
    test_dataset = Dataset(test_X, test_y)
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
            logging_steps = 100,
            fp16 = cfg.train.fp16
        )
    trainer = Trainer(
            model=model,  # the instantiated 🤗 Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=train_dataset,  # training dataset
            eval_dataset=test_dataset,  # evaluation dataset
            callbacks = [EarlyStoppingCallback(early_stopping_patience=2)]
        )

    trainer.train()
    torch.save(model, "./saved/model.pt")

    
    







    