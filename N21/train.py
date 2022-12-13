import argparse

from sklearn.metrics import accuracy_score
from datasets import load_dataset
import torch
from transformers import BertForSequenceClassification, AutoTokenizer, AutoConfig, Trainer ,TrainingArguments, EarlyStoppingCallback
from torch.utils.data import Dataset, DataLoader
from model.metric import compute_metrics
from utils.load_data import load_data
from utils.tokenizing import tokenizing
from data_loader.data_loader import Dataset

from omegaconf import OmegaConf

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="base_config")
    args, _ = parser.parse_known_args()

    cfg = OmegaConf.load(f'./config/{args.config}.yaml')

    train_dataset, valid_dataset, test_dataset = load_data()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
    train_df = tokenizing(train_dataset, tokenizer)
    valid_df = tokenizing(valid_dataset, tokenizer)
    test_df = tokenizing(test_dataset, tokenizer)

    config = AutoConfig.from_pretrained(cfg.model.model_name)
    config.num_labels = 7
    model = BertForSequenceClassification.from_pretrained(cfg.model.model_name, config = config)

    train_dataset = Dataset(train_df , train_dataset["label"])
    valid_dataset = Dataset(valid_df , valid_dataset["label"])
    test_dataset = Dataset(test_df, test_dataset["label"])

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
            eval_dataset=valid_dataset,  # evaluation dataset
            compute_metrics = compute_metrics,
            callbacks = [EarlyStoppingCallback(early_stopping_patience=2)]
        )

    trainer.train()

    torch.save(model, "./saved/model.pt")
