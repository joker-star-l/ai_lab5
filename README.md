# 多模态情感分类

人工智能实验五：多模态情感分类

## 准备工作

该仓库代码基于Python3实现，以下是代码的依赖库：

* numpy==1.19.5
* Pillow==9.1.0
* torch==1.11.0
* transformers==4.19.3

可以通过运行以下命令来安装这些依赖库。

```bash
pip install -r requirements.txt
```

## 代码结构

```
|-- data/                               # 数据
    |-- input/                          # 预处理后的数据
        |-- dataUtil.py                 # 预处理文件
        |-- test_data.json              # 预处理后的测试数据
        |-- train_data.json             # 预处理后的训练数据
    |-- raw/                            # 原始数据
    |-- test_without_label.txt          # 测试数据guid和空标签
    |-- train.txt                       # 训练数据guid和标签
|-- img/                                # 图片处理相关文件
    |-- imgConfig.py                    # 图片配置文件
    |-- imgDataset.py                   # 数据集生成器
    |-- vit.py                          # ViT
|-- result/                             # 日志和预测文件
|-- text/                               # 文本处理相关文件
    |-- bert.py                         # BERT
    |-- textConfig.py                   # 文本配置文件
    |-- textDataset.py                  # 数据集生成器
|-- config.py                           # 总配置文件
|-- imgClassification.py                # 模型（仅图片）
|-- multiClassification.py              # 模型（多模态）
|-- multiDataset.py                     # 数据集生成器
|-- run.py                              # 运行入口文件
|-- runUtil.py                          # 训练等工具方法
|-- textClassification.py               # 模型（仅文本）
```

## 代码在实验数据集上的运行流程

1. 把实验数据放到`data/raw/`目录下（可选，如果数据集不在该目录下则需要在运行`run.py`时手动指定`--raw_data_path`）。

2. 运行`run.py`文件

   训练和测试（使用默认参数）

   ```bash
   python run.py --train --test
   ```

   预测（使用默认参数）

   ```bash
   python run.py --predict
   ```

   可指定的参数列表（可以使用命令`python run.py -h` 或者在文件`run.py`中查看）

   |           参数            |              默认值               |                         说明                          |
   | :-----------------------: | :-------------------------------: | :---------------------------------------------------: |
   |          --mode           |           img_and_text            | 需要使用的数据类型：img_only, text_only, img_and_text |
   |          --train          |               False               |                         训练                          |
   |          --test           |               False               |                    在验证集上测试                     |
   |         --predict         |               False               |                    生成测试集标签                     |
   |  --train_with_label_path  |          data/train.txt           |                    train.txt的位置                    |
   | --test_without_label_path |    data/test_without_label.txt    |             test_without_label.txt的位置              |
   |      --raw_data_path      |             data/raw/             |                 图片和文本数据的位置                  |
   |     --train_data_path     |    data/input/train_data.json     |       预处理之后的未划分的训练集和验证集的位置        |
   |     --test_data_path      |     data/input/test_data.json     |               预处理之后的测试集的位置                |
   |    --cache_model_path     |            cache/model            |               训练过程中保存模型的位置                |
   |     --prediction_path     |       cache/prediction.txt        |               生成测试集标签文件的位置                |
   |          --seed           |               2022                |                      随机数种子                       |
   |       --batch_size        |                32                 |                      batch size                       |
   |          --epoch          |                 5                 |                         epoch                         |
   |           --lr            |               1e-3                |                    下游任务学习率                     |
   |          --bert           |   bert-base-multilingual-cased    |       bert（bert-base-multilingual-cased）位置        |
   |         --bert_lr         |               2e-5                |                    bert微调学习率                     |
   |           --vit           | google/vit-base-patch16-224-in21k |         vit（vit-base-patch16-224-in21k）位置         |
   |         --vit_lr          |               2e-5                |                     vit微调学习率                     |

## 参考的代码仓库

无。

## 参考资料

https://zhuanlan.zhihu.com/p/381805010

https://www.jiqizhixin.com/articles/2019-12-16-7

https://blog.csdn.net/Kaiyuan_sjtu/article/details/121391851

