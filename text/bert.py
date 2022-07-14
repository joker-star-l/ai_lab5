# coding = utf-8
# -*- coding:utf-8 -*-

from transformers import BertModel, BertTokenizer

import config
from text import textConfig

config.setup_seed()


def getBert():
    return BertModel.from_pretrained(
        textConfig.pretrained_model_name_or_path,
        config=textConfig.pretrained_model_name_or_path
    )


def getTokenizer():
    return BertTokenizer.from_pretrained(textConfig.pretrained_model_name_or_path)
