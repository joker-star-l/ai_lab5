# coding = utf-8
# -*- coding:utf-8 -*-
from transformers import ViTFeatureExtractor, ViTModel

import config
from img import imgConfig

config.setup_seed()


def getViT():
    return ViTModel.from_pretrained(
        imgConfig.pretrained_model_name_or_path,
        config=imgConfig.pretrained_model_name_or_path
    )


def getExtractor():
    return ViTFeatureExtractor.from_pretrained(imgConfig.pretrained_model_name_or_path)
