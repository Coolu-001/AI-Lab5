from transformers import RobertaTokenizer
from config import config
from torchvision import transforms
from torch.utils.data import Dataset
import torch
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from process_data import read_data
from torch.utils.data import DataLoader

class MultiModalDataset(Dataset):
    """
    多模态数据集类，用于处理文本和图像数据。
    """
    def __init__(self, guids, texts, texts_mask, images, labels) -> None:
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
        self.guids = guids
        self.texts = texts
        self.texts_mask = texts_mask
        self.images = images
        self.labels = labels
    
    def __len__(self):
        """
        返回数据集的样本数量。

        Returns:
            int: 数据集的样本数量。
        """
        return len(self.guids)
    
    def __getitem__(self, index):
        """
        根据索引获取数据集中的一个样本。

        Args:
            index (int): 样本的索引。

        Returns:
            tuple: 包含guid、文本、文本掩码、图像和标签的元组。
        """
        return self.guids[index], self.texts[index], self.texts_mask[index], self.images[index], self.labels[index]
    
def collate_fn(batch):
    """
    自定义数据加载器的collate函数，用于将一批样本整理为模型输入格式。

    Args:
        batch (list): 一批样本，每个样本是一个元组（guid, 文本, 文本掩码, 图像, 标签）。

    Returns:
        tuple: 包含整理后的guid、文本、文本掩码、图像和标签的元组。
    """
    guids = [item[0] for item in batch]
    texts = [item[1].squeeze(0) for item in batch]
    texts_mask = [item[2].squeeze(0) for item in batch]
    images = torch.stack([item[3] for item in batch])
    labels = torch.LongTensor([item[4] for item in batch])

    padding_texts = pad_sequence(texts, batch_first=True, padding_value=0)
    padding_texts_mask = pad_sequence(texts_mask, batch_first=True, padding_value=0).gt(0)

    return guids, padding_texts, padding_texts_mask, images, labels

def resize_size(image_size):
    """
    计算最接近且大于等于image_size的2的幂次方值，用于调整图像大小。

    Args:
        image_size (int): 原始图像大小。

    Returns:
        int: 调整后的图像大小。
    """
    for i in range(20):
        if 2 ** i >= image_size:
            return 2 ** i
    return image_size

def encode_text_image(original_data, mode):
    """
    对原始数据进行编码，包括文本的tokenization和图像的预处理。

    Args:
        original_data (list): 原始数据列表，每个元素是一个字典，包含guid、label、text和image。
        mode (str): 模式，可以是"train"或"test"，用于选择不同的图像预处理方式。

    Returns:
        tuple: 包含编码后的guids、文本、文本掩码、图像和标签的元组。
    """
    tokenizer = RobertaTokenizer.from_pretrained(config.roberta_path)

    train_transform = transforms.Compose([
        transforms.Resize(resize_size(config.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(resize_size(config.image_size)),
        transforms.CenterCrop(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    guids = []
    encoded_texts = []
    encoded_texts_mask = []
    encoded_images = []
    encoded_labels = []

    for group in original_data:
        guid = group["guid"]
        label = group["label"]
        text = group["text"]
        image = group["image"]

        guids.append(guid)
        tokens = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=config.max_seq_length)
        encoded_texts.append(tokens["input_ids"].squeeze(0))
        encoded_texts_mask.append(tokens["attention_mask"].squeeze(0))

        if mode == "train":
            encoded_images.append(train_transform(image))
        elif mode == "test":
            encoded_images.append(test_transform(image))

        encoded_labels.append(label)

    return guids, encoded_texts, encoded_texts_mask, encoded_images, encoded_labels

def create_dataloader(train_data_path, test_data_path, data_path, text_only=False, image_only=False):
    """
    创建训练、验证和测试数据加载器。

    Args:
        train_data_path (str): 训练数据路径。
        test_data_path (str): 测试数据路径。
        data_path (str): 数据根路径。
        text_only (bool, optional): 是否仅使用文本数据。默认值为False。
        image_only (bool, optional): 是否仅使用图像数据。默认值为False。

    Returns:
        tuple: 包含训练、验证和测试数据加载器的元组。
    """
    original_train_data = read_data(train_data_path, data_path, text_only, image_only)
    original_test_data = read_data(test_data_path, data_path, text_only, image_only)
    train_dataset_inputs = encode_text_image(original_train_data, "train")
    test_dataset_inputs = encode_text_image(original_test_data, "test")

    original_train_datasets = MultiModalDataset(*train_dataset_inputs)
    test_datasets = MultiModalDataset(*test_dataset_inputs)

    train_datasets, valid_datasets = train_test_split(original_train_datasets, test_size=0.2, random_state=config.seed)

    train_dataloader = DataLoader(
        dataset=train_datasets, 
        batch_size=config.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=True
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