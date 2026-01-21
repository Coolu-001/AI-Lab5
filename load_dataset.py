from transformers import RobertaTokenizer
from config import config
from torchvision import transforms
from torch.utils.data import Dataset
import torch
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from process_data import read_data
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, CLIPProcessor
import random

class TextAugmenter:
    def __init__(self, p_delete=0.1, p_swap=0.1):
        self.p_delete = p_delete
        self.p_swap = p_swap

    def random_delete(self, words):
        """随机删除词语，模拟文本缺失"""
        if len(words) <= 2: return words
        return [w for w in words if random.random() > self.p_delete]

    def random_swap(self, words):
        """随机交换位置，增强模型对语序抖动的鲁棒性"""
        if len(words) <= 2: return words
        new_words = words.copy()
        idx1, idx2 = random.sample(range(len(words)), 2)
        new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
        return new_words

    def augment(self, text):
        words = text.split()
        if not words: return text
        
        # 随机选择一种增强方式
        seed = random.random()
        if seed < self.p_delete:
            words = self.random_delete(words)
        elif seed < self.p_delete + self.p_swap:
            words = self.random_swap(words)
            
        return " ".join(words)
    
class MultiModalDataset(Dataset):
    """
    多模态数据集类，用于处理文本和图像数据。
    """
    def __init__(self, guids, texts, images, labels, tokenizer, config, transform=None, mode='train') -> None:
        super().__init__()
        """
        初始化数据集。

        Args:
            guids (list): 数据样本的唯一标识符列表。
            texts (list): 文本数据列表。
            texts_mask (list): 文本数据的注意力掩码列表。
            images (list): 图像数据列表。
            labels (list): 标签数据列表。
        """
        super().__init__()
        self.tokenizer = tokenizer 
        self.config = config      
        self.guids = guids
        self.texts = texts
        self.images = images
        self.labels = labels
        self.transform = transform
        self.augmenter = TextAugmenter()
        self.mode = mode
    
    def __len__(self):
        return len(self.guids)
    
    def __getitem__(self, index):
        guid = self.guids[index]
        image = self.images[index]
        text = self.texts[index]
        label = self.labels[index]
        if self.mode == 'train':
            text = self.augmenter.augment(text)
        tokens = self.tokenizer(
            text, 
            return_tensors='pt', 
            padding='max_length', 
            truncation=True, 
            max_length=self.config.max_seq_length
        )
        if self.transform:
            image = self.transform(image)
        return guid, tokens["input_ids"].squeeze(0), tokens["attention_mask"].squeeze(0), image, label

def collate_fn(batch):
    # 1. 提取各个字段
    guids = [item[0] for item in batch]
    
    # 因为 Dataset 里已经 squeeze 过了，这里直接 stack 成 [Batch_Size, Max_Length]
    input_ids = torch.stack([item[1] for item in batch])
    attention_mask = torch.stack([item[2] for item in batch])
    
    # 图像堆叠成 [Batch_Size, 3, 224, 224]
    images = torch.stack([item[3] for item in batch])
    
    # 标签转为 Tensor
    labels = torch.LongTensor([item[4] for item in batch])

    # 返回 5 个变量
    return guids, input_ids, attention_mask, images, labels


def create_dataloader(train_data_path, test_data_path, data_path, text_only=False, image_only=False):
    # 1. 读取原始数据
    original_train_data = read_data(train_data_path, data_path, text_only, image_only)
    original_test_data = read_data(test_data_path, data_path, text_only, image_only)

    # 2. 准备基础数据列表（注意：此时 text 还是原始字符串/清洗后的字符串）
    def prepare_raw_lists(data):
        guids, texts, images, labels = [], [], [], []
        for item in data:
            guids.append(item["guid"])
            # 这里调用之前定义的 clean_text 做静态预处理（去冒号、全小写等）
            texts.append(item["text"])
            images.append(item["image"])
            labels.append(item["label"])
        return guids, texts, images, labels

    t_guids, t_texts, t_images, t_labels = prepare_raw_lists(original_train_data)
    te_guids, te_texts, te_images, te_labels = prepare_raw_lists(original_test_data)

    # 3. 划分训练集和验证集
    # 这里我们只传 4 个字段，因为 mask 会在 Dataset 内部动态生成
    split_result = train_test_split(
        t_guids, t_texts, t_images, t_labels, 
        test_size=0.2, 
        random_state=config.seed
    )
    
    # 重新组织切分后的数据：[train_guids, val_guids, train_texts, val_texts, ...]
    train_data = {
        "guids": split_result[0],
        "texts": split_result[2],
        "images": split_result[4],
        "labels": split_result[6]
    }
    valid_data = {
        "guids": split_result[1],
        "texts": split_result[3],
        "images": split_result[5],
        "labels": split_result[7]
    }
    
    # 4. 定义图像变换 (保持 CLIP 专用参数)
    train_trans = transforms.Compose([
        transforms.Resize(256), 
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0)), 
        transforms.RandomHorizontalFlip(p=0.5), 
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073), 
            std=(0.26862954, 0.26130258, 0.27577711)
        )
    ])

    val_test_trans = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073), 
            std=(0.26862954, 0.26130258, 0.27577711)
        )
    ])

    # 5. 初始化 Tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(config.roberta_path)

    # 6. 创建 Dataset 实例 (将 tokenizer 传入)
    train_datasets = MultiModalDataset(
        train_data["guids"], train_data["texts"], train_data["images"], train_data["labels"],
        tokenizer=tokenizer, config=config, transform=train_trans, mode='train'
    )
    valid_datasets = MultiModalDataset(
        valid_data["guids"], valid_data["texts"], valid_data["images"], valid_data["labels"],
        tokenizer=tokenizer, config=config, transform=val_test_trans, mode='val'
    )
    test_datasets = MultiModalDataset(
        te_guids, te_texts, te_images, te_labels,
        tokenizer=tokenizer, config=config, transform=val_test_trans, mode='test'
    )

    # 7. 创建 DataLoader
    train_dataloader = DataLoader(
        dataset=train_datasets, 
        batch_size=config.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True
    )

    valid_dataloader = DataLoader(
        dataset=valid_datasets, 
        batch_size=config.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=True
    )

    test_dataloader = DataLoader(
        dataset=test_datasets, 
        batch_size=config.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=True
    )

    return train_dataloader, valid_dataloader, test_dataloader