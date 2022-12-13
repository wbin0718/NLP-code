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

    config = AutoConfig.from_pretrained(cfg.model.model_name)
    config.num_labels = 7
    
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model = torch.load(cfg.test.best_model, map_location = device)

    category_dict = {
                  0 : '정치', 
                  1 : '경제',
                  2 : '사회',
                  3 : '생황문화',
                  4 : '세계',
                  5 : 'IT과학',
                  6 : '스포츠'
                }
    news_title = '젤렌스키 "러에 뺏긴 영토 수복 없이는 휴전도 없다"'

    tokens = tokenizer(
            news_title,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            max_length=cfg.data.max_len
        )
    tokens = {k:v.to(device) for k,v in tokens.items()}
    
    outputs = model(**tokens)
    best_pred_class_label_id = outputs['logits'].argmax(1).item()
    print("Predicted : ", category_dict[best_pred_class_label_id])