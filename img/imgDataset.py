# coding = utf-8
# -*- coding:utf-8 -*-
import json

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import ViTFeatureExtractor

import config
from img import vit

config.setup_seed()

tags = {
    'positive': 0,
    'negative': 1,
    'neutral': 2,
    '': 3  # 仅占位
}


class ImgDataset(Dataset):

    def __init__(self, data: list, extractor: ViTFeatureExtractor):
        self.data = data
        self.extractor = extractor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        guid = self.data[item]['guid']
        img = self.data[item]['img']
        tag = self.data[item]['tag']
        tag = torch.tensor(tags[tag], dtype=torch.long)
        img = self.extractor(
            images=Image.open(config.raw_data_path + img),
            return_tensors='pt'
        )

        return {
            'guid': guid,
            'img': img,
            'tag': tag
        }


def getImgDataset(path: str):
    with open(path, 'r', encoding='utf-8') as fs:
        data = json.load(fs)

    extractor = vit.getExtractor()
    return ImgDataset(data, extractor)


if __name__ == '__main__':
    data_loader = DataLoader(getImgDataset(config.train_data_path), batch_size=config.batch_size, shuffle=True)
    pretrained = vit.getViT()
    for param in pretrained.parameters():
        param.requires_grad_(False)
    for i, data in enumerate(data_loader):
        # print(data)
        print(data['img']['pixel_values'].shape)
        out = pretrained(
            pixel_values=data['img']['pixel_values'][:, 0]
        )
        print(out['last_hidden_state'].shape)
        print(out['pooler_output'].shape)
        break
