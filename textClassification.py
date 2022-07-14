# coding = utf-8
# -*- coding:utf-8 -*-
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

import config
from runUtil import train, test, predict, device
from text import bert, textConfig
from text.textDataset import getTextDataset

config.setup_seed()


class TextModel(nn.Module):
    def __init__(self, fine_tune: bool):
        super().__init__()

        self.bert = bert.getBert()
        for param in self.bert.parameters():
            param.requires_grad_(fine_tune)

        self.dp = nn.Dropout(0.5)
        self.fc = nn.Linear(768, 3)

    def forward(self, data):
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        token_type_ids = data['token_type_ids'].to(device)

        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        out = self.fc(self.dp(out['pooler_output']))

        return out


def run():
    model = TextModel(fine_tune=textConfig.fine_tune)
    model.to(device)

    bert_params = list(map(id, model.bert.parameters()))
    down_params = filter(lambda p: id(p) not in bert_params, model.parameters())
    optimizer = AdamW([
        {'params': model.bert.parameters(), 'lr': textConfig.bert_lr},
        {'params': down_params, 'lr': textConfig.lr}
    ])

    dataset = getTextDataset(config.train_data_path)
    # print(len(dataset))
    train_dataset = Subset(dataset, range(0, 3500))
    val_dataset = Subset(dataset, range(3500, 4000))
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)
    train(model, optimizer, train_loader, val_loader)


def testNow():
    model = torch.load(config.cache_model_path, map_location=device)
    dataset = getTextDataset(config.train_data_path)
    val_dataset = Subset(dataset, range(3500, 4000))
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)
    print('final validation accuracy:', test(model, val_loader))


def predictNow():
    model = torch.load(config.cache_model_path, map_location=device)
    test_loader = DataLoader(getTextDataset(config.test_data_path), batch_size=config.batch_size, shuffle=False)
    predict(model, test_loader)


if __name__ == '__main__':
    run()
    testNow()
    predictNow()
