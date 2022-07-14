# coding = utf-8
# -*- coding:utf-8 -*-
import json

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import BertTokenizer, ViTFeatureExtractor

import config
from img import vit
from text import bert, textConfig

config.setup_seed()


tags = {
    'positive': 0,
    'negative': 1,
    'neutral': 2,
    '': 3  # 仅占位
}


class MultiDataset(Dataset):

    def __init__(self, data: list, tokenizer: BertTokenizer, extractor: ViTFeatureExtractor, maxLen: int):
        self.data = data
        self.tokenizer = tokenizer
        self.extractor = extractor
        self.maxLen = maxLen

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        guid = self.data[item]['guid']
        text = self.data[item]['text']
        img = self.data[item]['img']
        tag = self.data[item]['tag']
        tag = torch.tensor(tags[tag], dtype=torch.long)
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.maxLen,
            padding='max_length',
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        img = self.extractor(
            images=Image.open(config.raw_data_path + img),
            return_tensors='pt'
        )

        return {
            'guid': guid,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'img': img,
            'tag': tag
        }


def getMultiDataset(path: str):
    with open(path, 'r', encoding='utf-8') as fs:
        data = json.load(fs)

    tokenizer = bert.getTokenizer()
    extractor = vit.getExtractor()
    return MultiDataset(data, tokenizer, extractor, textConfig.max_len)
