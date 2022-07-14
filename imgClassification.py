# coding = utf-8
# -*- coding:utf-8 -*-
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

import config
from img import imgConfig, vit
from img.imgDataset import getImgDataset
from runUtil import train, test, predict, device

config.setup_seed()


class ImgModel(nn.Module):
    def __init__(self, fine_tune: bool):
        super().__init__()

        self.vit = vit.getViT()
        for param in self.vit.parameters():
            param.requires_grad_(fine_tune)

        self.dp = nn.Dropout(0.1)
        self.fc = nn.Linear(768, 3)

    def forward(self, data):
        pixel_values = data['img']['pixel_values'][:, 0].to(device)

        out = self.vit(
            pixel_values=pixel_values
        )
        out = self.fc(self.dp(out['pooler_output']))

        return out


def run():
    model = ImgModel(fine_tune=imgConfig.fine_tune)
    model.to(device)

    vit_params = list(map(id, model.vit.parameters()))
    down_params = filter(lambda p: id(p) not in vit_params, model.parameters())
    optimizer = AdamW([
        {'params': model.vit.parameters(), 'lr': imgConfig.vit_lr},
        {'params': down_params, 'lr': imgConfig.lr}
    ])

    dataset = getImgDataset(config.train_data_path)
    # print(len(dataset))
    train_dataset = Subset(dataset, range(0, 3500))
    val_dataset = Subset(dataset, range(3500, 4000))
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)
    train(model, optimizer, train_loader, val_loader)


def testNow():
    model = torch.load(config.cache_model_path, map_location=device)
    dataset = getImgDataset(config.train_data_path)
    val_dataset = Subset(dataset, range(3500, 4000))
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)
    print('final validation accuracy:', test(model, val_loader))


def predictNow():
    model = torch.load(config.cache_model_path, map_location=device)
    test_loader = DataLoader(getImgDataset(config.test_data_path), batch_size=config.batch_size, shuffle=False)
    predict(model, test_loader)


if __name__ == '__main__':
    run()
    testNow()
    predictNow()
