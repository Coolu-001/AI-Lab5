from config import config
import os
import chardet
import json
import re
from PIL import Image
import torch
import numpy as np
from transformers import RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

def Label2Index(label):
    """
    将标签字符串转换为索引。

    Args:
        label (str): 标签字符串（"positive", "neutral", "negative", "null"）。

    Returns:
        int: 对应的索引值。
    """
    if label == "positive":
        return 0
    elif label == "neutral":
        return 1
    elif label == "negative":
        return 2
    elif label == "null":
        return 3

def Index2Label(index):
    """
    将索引转换为标签字符串。

    Args:
        index (int): 索引值。

    Returns:
        str: 对应的标签字符串。
    """
    if index == 0:
        return "positive"
    elif index == 1:
        return "neutral"
    elif index == 2:
        return "negative"
    elif index == 3:
        return "null"

import re

def clean_text(text):
    # 1. 去除 URL
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

    # 2. 新增：去除冒号之前的内容（如 "用户名: " 或 "地点: "）
    # 逻辑：寻找第一个出现的中文或英文冒号，取其后的内容
    if ':' in text or '：' in text:
        # 使用正则表达式兼容中英文冒号，只分割一次
        parts = re.split(r':|：', text, maxsplit=1)
        if len(parts) > 1:
            text = parts[1]

    # 3. 处理换行符、制表符、话题标签 #
    text = text.replace("\n", " ").replace("\t", " ").replace("#", "")
    
    # 4. 去除表情符号和特定符号（注意保留标点有助于 RoBERTa 理解语气）
    text = re.sub(r'[^\w\s,.\!\?]', '', text) 
    
    # 5. 合并多余空格并全小写
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()

def combine_data(input_path, output_path, data_path):
    """
    将原始数据（GUID和标签）与文本数据结合，生成包含GUID、标签和文本的JSON文件。

    Args:
        input_path (str): 输入文件路径，包含GUID和标签。
        output_path (str): 输出文件路径，保存结合后的数据。
        data_path (str): 文本数据所在的目录路径。
    """
    combined_data = []
    with open(input_path, "r") as f:
        next(f)
        for line in f:
            guid, label = line.strip().split(",")
            text_path = os.path.join(data_path, (guid + ".txt"))
            try:
                with open(text_path, "r", encoding='utf-8') as f2:
                    text = f2.read()
            except UnicodeDecodeError:
                with open(text_path, "rb") as f2:
                    text_in_byte = f2.read()
                    encode = chardet.detect(text_in_byte)
                    try:
                        text = text_in_byte.decode(encode["encoding"])
                    except:
                        try:
                            text = text_in_byte.decode('iso-8859-1').encode('iso-8859-1').decode('gbk')
                        except:
                            print("decode error in file: ", guid)
                            continue
            text = clean_text(text)
            label2index = Label2Index(label)
            combined_data.append({"guid": guid, "label": label2index, "text": text})

    with open(output_path, "w", encoding="utf-8") as tf:
        json.dump(combined_data, tf, ensure_ascii=False, indent=4)

def read_data(file_path, data_path, text_only=False, image_only=False):
    """
    读取数据文件，加载文本和图像数据。

    Args:
        file_path (str): 数据文件路径（JSON格式）。
        data_path (str): 图像数据所在的目录路径。
        text_only (bool, optional): 是否仅加载文本数据。默认值为False。
        image_only (bool, optional): 是否仅加载图像数据。默认值为False。

    Returns:
        list: 包含GUID、标签、文本和图像的数据列表。
    """
    data = []
    with open(file_path) as f:
        josn_data = json.load(f)
        for jd in josn_data:
            guid, label, text = jd["guid"], jd["label"], jd["text"]
            if text_only:
                image = Image.new(mode='RGB', size=(224, 224), color=(0, 0, 0))
            else:
                image = Image.open(os.path.join(data_path, (guid + ".jpg")))
                image.load()
            
            if image_only:
                text = tokenizer.pad_token * config.max_seq_length          
            
            data.append({
                "guid": guid,
                "label": label,
                "text": text,
                "image": image
            })
        f.close()
    return data

def calculate_95th_percentile_length(file_path):
    """
    计算文本长度的95%分位数、99%分位数和最大值。

    Args:
        file_path (str): 数据文件路径（JSON格式）。

    Returns:
        tuple: 包含95%分位数、99%分位数和最大值的元组。
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    lengths = []
    for item in data:
        text = item["text"]
        tokens = tokenizer.tokenize(text)
        lengths.append(len(tokens))
    percentile_95 = np.percentile(lengths, 95)
    percentile_99 = np.percentile(lengths, 99)
    max_length = max(lengths)
    return percentile_95, percentile_99, max_length


if __name__ == "__main__":
    file_path = "dataset/train1.json"
    percentile_95, percentile_99, max_length = calculate_95th_percentile_length(file_path)
    print(f"训练集文本长度的 95% 分位数: {percentile_95}")  # 45
    print(f"训练集文本长度的 99% 分位数: {percentile_99}")  # 62
    print(f"训练集文本长度的最大值: {max_length}") # 255

    file_path = "dataset/test1.json"
    percentile_95, percentile_99, max_length = calculate_95th_percentile_length(file_path)
    print(f"测试集文本长度的 95% 分位数: {percentile_95}")  # 38.5
    print(f"测试集文本长度的 99% 分位数: {percentile_99}")  # 48
    print(f"测试集文本长度的最大值: {max_length}") # 147

# if __name__ == "__main__":
#     combine_data(config.train_guid_label_path, config.train_data_path, config.data_path)
#     combine_data(config.test_guid_label_path, config.test_data_path, config.data_path)