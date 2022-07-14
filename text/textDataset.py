# coding = utf-8
# -*- coding:utf-8 -*-
import json

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

from text import bert, textConfig
import config

config.setup_seed()


def fillPadding(data, maxLen):
    if len(data) < maxLen:
        padLen = maxLen - len(data)
        padding = [0 for _ in range(padLen)]
        data = data + padding
    else:
        data = data[: maxLen]
    return torch.tensor(data)


tags = {
    'positive': 0,
    'negative': 1,
    'neutral': 2,
    '': 3  # 仅占位
}


class TextDataset(Dataset):

    def __init__(self, data: list, tokenizer: BertTokenizer, maxLen: int):
        self.data = data
        self.tokenizer = tokenizer
        self.maxLen = maxLen

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        guid = self.data[item]['guid']
        text = self.data[item]['text']
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

        return {
            'guid': guid,
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'tag': tag
        }


def getTextDataset(path: str):
    with open(path, 'r', encoding='utf-8') as fs:
        data = json.load(fs)

    # maxLen = 0
    # for d in data:
    #     length = len(d['text'].split(' '))
    #     if length > maxLen:
    #         maxLen = length
    # print(maxLen)

    tokenizer = bert.getTokenizer()
    return TextDataset(data, tokenizer, textConfig.max_len)


if __name__ == '__main__':
    data_loader = DataLoader(getTextDataset(config.train_data_path), batch_size=config.batch_size, shuffle=True)
    pretrained = bert.getBert()
    for param in pretrained.parameters():
        param.requires_grad_(False)
    for i, data in enumerate(data_loader):
        print(data)
        out = pretrained(
            input_ids=data['input_ids'],
            attention_mask=data['attention_mask'],
            token_type_ids=data['token_type_ids']
        )
        print(out['last_hidden_state'].shape)
        break
