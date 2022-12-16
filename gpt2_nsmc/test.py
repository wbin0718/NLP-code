import argparse
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from utils.sentiment_predict import sentiment_predict

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="base_config")
    args, _ = parser.parse_known_args()

    cfg = OmegaConf.load(f'./config/{args.config}.yaml')
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name, bos_token="<s>", eos_token = "</s>", pad_token = "<pad>")
    model = torch.load(cfg.test.best_model)

    sentiment_predict(model,tokenizer,'보던거라 계속보고있는데 전개도 느리고 주인공인 은희는 한두컷 나오면서 소극적인모습에 ')

    sentiment_predict(model,tokenizer,"스토리는 확실히 실망이였지만 배우들 연기력이 대박이였다 특히 이제훈 연기 정말 ... 이 배우들로 이렇게밖에 만들지 못한 영화는 아쉽지만 배우들 연기력과 사운드는 정말 빛났던 영화. 기대하고 극장에서 보면 많이 실망했겠지만 평점보고 기대없이 집에서 편하게 보면 괜찮아요. 이제훈님 연기력은 최고인 것 같습니다")

    sentiment_predict(model,tokenizer,"남친이 이 영화를 보고 헤어지자고한 영화. 자유롭게 살고 싶다고 한다. 내가 무슨 나비를 잡은 덫마냥 나에겐 다시 보고싶지 않은 영화.")

    sentiment_predict(model,tokenizer,"이 영화 존잼입니다 대박")

    sentiment_predict(model,tokenizer,'이 영화 개꿀잼 ㅋㅋㅋ')

    sentiment_predict(model,tokenizer,'이 영화 핵노잼 ㅠㅠ')

    sentiment_predict(model,tokenizer,'이딴게 영화냐 ㅉㅉ')

    sentiment_predict(model,tokenizer,'감독 뭐하는 놈이냐?')

    sentiment_predict(model,tokenizer,'와 개쩐다 정말 세계관 최강자들의 영화다')
