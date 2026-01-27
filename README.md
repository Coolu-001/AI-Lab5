# 多模态情感分析

多模态情感分析是让文本和图片共同参与分类器决策的任务，分类模型由三部分构成：Text Model、Image Model 和 Fusion Model。本项目基于RoBERTa模型和Clip模型提取了文本特征和图像特征，尝试了两种融合方式（Late Fusion 和 Early Fusion）将文本特征和图像特征进行了融合，得出最终的分类结果（0:positive  1: neutral 2:negative）。此外，还尝试对比了各个Baseline的变体。

项目代码提供```text_only```和```image_only```的参数选择，方便在任何模型上执行消融实验，为了简化代码逻辑，文本消融实验通过用```pad_token```填充```max_seq_length```实现，图片消融实验通过将图片处理为全黑图像实现。

## 环境配置

本项目代码使用 Python3.11，需要安装以下依赖:

chardet == 5.2.0
numpy == 2.2.2
Pillow == 11.1.0
scikit_learn == 1.6.1
torch == 2.5.1
torchvision == 0.20.1
tqdm == 4.67.1
transformers == 4.47.1
wandb == 0.19.1
matplotlib

进入文件夹后，可以直接运行以下命令来安装依赖：

```python
pip install -r requirements.txt
```

## 仓库结构

```python
|-- dataset 
    |-- data/ # 文本与图片数据
|-- roberta-base # 用于存放roberta预训练模型，防止网络不稳定，也可以选择不下载
|-- config.py # 实验参数配置
|-- early_stopping.py # 早停机制
|-- load_dataset.py # 加载数据集
|-- main.py # 主函数
|-- FusionModels.py # 存放Text Model、Image Model以及4种Fusion Model
|-- predict.py # 对无标签数据进行预测
|-- process_data.py # 前期处理数据
|-- search.py # 超参数搜索
|-- set_different_seeds # 用于进行十次随机种子实验
|-- train_validate.py # 用于进行模型训练与评估
```

## 代码使用
1. 下载文本和图片数据集存放到`dataset/data`中，下载labels标签文本文件存放到`dataset`中。
2. 下载roberta预训练模型存放到`roberta-base`中，主要下载四个文件`config.json`,`merges.txt`,`pytorch_model.bin`, `vocab.json`。
3. 在`config.py`中修改参数配置，如数据集路径、模型保存路径等。
4. 运行代码:
```Shell
python main.py --batch_size 16 --roberta_dropout 0.4 --roberta_lr 2e-5 --middle_hidden_size 768 --clip_dropout 0.4 \ --clip_lr 1e-6 --attention_nheads 8 --attention_dropout 0.4 --fusion_dropout 0.5 --output_hidden_size 256
\ --weight_decay 0.1 --lr 5e-5 --text_only --model 5
```
-  参数说明:
    - `batch_size 16`: 设置批量大小为 16。
    - `roberta_dropout 0.4`: 设置 RoBERTa 的 dropout 率为 0.4。
    - `roberta_lr 2e-5`: 设置 RoBERTa 的学习率为 2e-5。
    - `middle_hidden_size 512`: 设置中间层的隐藏大小为 768。
    - `clip_dropout 0.4`: 设置 Clip 的 dropout 率为 0.4。
    - `clip_lr 1e-6`: 设置 Clip 的学习率为 1e-6。
    - `attention_nheads 8`: 设置注意力头的数量为 8。
    - `attention_dropout 0.4`: 设置注意力层的 dropout 率为 0.4。
    - `fusion_dropout 0.5`: 设置融合层的 dropout 率为 0.5。
    - `output_hidden_size 512`: 设置输出层的隐藏大小为 512。
    - `weight_decay 0.1`: 设置优化器的权重衰减为 0.1。
    - `lr 2e-5`: 设置优化器的学习率为 5e-5。
    - `text_only`: 启用仅使用文本的模式。
    - `model 5`: 选择Dynamic Gated Fusion Model。
**注：代码参数默认设置即为本次实验采用的参数组合，命令可简化，仅选择model**
**由于不同 FusionModel的训练代码略有不同，Late Fusion需要接受更多的参数，故采用版本管理**

- 当前页面所在main分支代码适配Dynamic Gated Fusion Model
- 适配Late Fusion Model
- 适配剩下两种：ConcatFusionModel、TransformerFusionModel

## 代码参考
本次实验参考了以下代码仓库:
-  https://huggingface.co/FacebookAI/roberta-base
-  https://github.com/Link-Li/CLMLF.git
