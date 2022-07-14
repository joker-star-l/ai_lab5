# coding = utf-8
# -*- coding:utf-8 -*-
import os
import argparse

import config
from data.input import dataUtil
from text import textConfig
from img import imgConfig
import imgClassification
import textClassification
import multiClassification
from multiClassification import MultiModel
from imgClassification import ImgModel
from textClassification import TextModel

config.setup_seed()


def parse_args():
    parse = argparse.ArgumentParser()

    parse.add_argument('--mode', type=str, default='img_and_text', help='需要使用的数据类型：img_only, text_only, img_and_text')
    parse.add_argument('--train', action='store_true', help='训练')
    parse.add_argument('--test', action='store_true', help='在验证集上测试')
    parse.add_argument('--predict', action='store_true', help='生成测试集标签')

    parse.add_argument('--train_with_label_path', type=str, default=config.train_with_label_path, help='train.txt的位置')
    parse.add_argument('--test_without_label_path', type=str, default=config.test_without_label_path, help='test_without_label.txt的位置')
    parse.add_argument('--raw_data_path', type=str, default=config.raw_data_path, help='图片和文本数据的位置')
    parse.add_argument('--train_data_path', type=str, default=config.train_data_path, help='预处理之后的未划分的训练集和验证集的位置')
    parse.add_argument('--test_data_path', type=str, default=config.test_data_path, help='预处理之后的测试集的位置')
    parse.add_argument('--cache_model_path', type=str, default=config.cache_model_path, help='训练过程中保存模型的位置')
    parse.add_argument('--prediction_path', type=str, default=config.prediction_path, help='生成测试集标签文件的位置')

    parse.add_argument('--seed', type=int, default=config.seed, help='随机数种子')
    parse.add_argument('--batch_size', type=int, default=config.batch_size, help='batch size')
    parse.add_argument('--epoch', type=int, default=config.epoch, help='epoch')
    parse.add_argument('--lr', type=float, default=config.lr, help='下游任务学习率')

    parse.add_argument('--bert', type=str, default=textConfig.pretrained_model_name_or_path, help='bert（bert-base-multilingual-cased）位置')
    parse.add_argument('--bert_lr', type=float, default=textConfig.bert_lr, help='bert微调学习率')

    parse.add_argument('--vit', type=str, default=imgConfig.pretrained_model_name_or_path, help='vit（vit-base-patch16-224-in21k）位置')
    parse.add_argument('--vit_lr', type=str, default=imgConfig.vit_lr, help='vit微调学习率')

    return parse.parse_args()


def args2config():
    config.train_with_label_path = args.train_with_label_path
    config.test_without_label_path = args.test_without_label_path
    config.raw_data_path = args.raw_data_path
    config.train_data_path = args.train_data_path
    config.test_data_path = args.test_data_path
    config.cache_model_path = args.cache_model_path
    config.prediction_path = args.prediction_path

    config.seed = args.seed
    config.batch_size = args.batch_size
    config.epoch = args.epoch
    assert 0 < args.lr < 1
    config.lr = imgConfig.lr = textConfig.lr = args.lr

    textConfig.pretrained_model_name_or_path = args.bert
    assert 0 < args.bert_lr < 1
    textConfig.bert_lr = args.bert_lr

    imgConfig.pretrained_model_name_or_path = args.vit
    assert 0 < args.vit_lr < 1
    imgConfig.vit_lr = args.vit_lr


if __name__ == '__main__':
    args = parse_args()
    args2config()

    # 数据预处理
    if (not os.path.exists(config.train_data_path)) or (not os.path.exists(config.test_data_path)):
        dataUtil.run()

    if args.mode == 'img_and_text':
        if args.train:
            multiClassification.run()
        if args.test:
            multiClassification.testNow()
        if args.predict:
            multiClassification.predictNow()
    elif args.mode == 'img_only':
        if args.train:
            imgClassification.run()
        if args.test:
            imgClassification.testNow()
        if args.predict:
            imgClassification.predictNow()
    elif args.mode == 'text_only':
        if args.train:
            textClassification.run()
        if args.test:
            textClassification.testNow()
        if args.predict:
            textClassification.predictNow()
