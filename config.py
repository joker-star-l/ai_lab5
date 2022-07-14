# coding = utf-8
# -*- coding:utf-8 -*-
import random

import numpy as np
import torch
import os

root_path = os.getcwd().replace('\\', '/') + '/'  # '/root/ai_5/'
train_with_label_path = root_path + 'data/train.txt'
test_without_label_path = root_path + 'data/test_without_label.txt'
raw_data_path = root_path + 'data/raw/'
train_data_path = root_path + 'data/input/train_data.json'
test_data_path = root_path + 'data/input/test_data.json'
cache_model_path = root_path + 'cache/model'
prediction_path = root_path + 'cache/prediction.txt'

seed = 2022
batch_size = 32
epoch = 5
lr = 1e-3
fine_tune = True


def setup_seed():
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
