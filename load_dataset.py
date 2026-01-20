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

class MultiModalDataset(Dataset):
    """
    多模态数据集类，用于处理文本和图像数据。
    """
    def __init__(self, guids, texts, texts_mask, images, labels, transform=None) -> None:
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
        self.transform = transform
    
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
        image = self.images[index]
        if self.transform:
            image = self.transform(image)
        return self.guids[index], self.texts[index], self.texts_mask[index], image, self.labels[index]

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

# def encode_text_image(original_data):
#     """
#     对原始数据进行编码，包括文本的tokenization和图像的预处理。

#     Args:
#         original_data (list): 原始数据列表，每个元素是一个字典，包含guid、label、text和image。
#         mode (str): 模式，可以是"train"或"test"，用于选择不同的图像预处理方式。

#     Returns:
#         tuple: 包含编码后的guids、文本、文本掩码、图像和标签的元组。
#     """
#     tokenizer = RobertaTokenizer.from_pretrained(config.roberta_path)


#     guids = []
#     encoded_texts = []
#     encoded_texts_mask = []
#     encoded_images = []
#     encoded_labels = []

#     for group in original_data:
#         guid = group["guid"]
#         label = group["label"]
#         text = group["text"]
#         image = group["image"]

#         guids.append(guid)
#         tokens = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=config.max_seq_length)
#         # 之前通过分位数计算得出的值作为max_seq_length
#         encoded_texts.append(tokens["input_ids"].squeeze(0))
#         encoded_texts_mask.append(tokens["attention_mask"].squeeze(0))
#         encoded_images.append(image)
#         encoded_labels.append(label)

#     return guids, encoded_texts, encoded_texts_mask, encoded_images, encoded_labels

def encode_text_image(original_data):
    """
    对原始数据进行编码，包括文本的 Roberta Tokenization 和图像的 CLIP 预处理。

    Args:
        original_data (list): 原始数据列表，每个元素是一个字典，包含 guid、label、text 和 image (PIL对象)。

    Returns:
        tuple: 包含编码后的 guids、文本 ID、注意力掩码、CLIP 处理后的图像张量和标签。
    """
    # 1. 初始化 Tokenizer 和 CLIP 处理器
    tokenizer = RobertaTokenizer.from_pretrained(config.roberta_path)
    # config.clip_path 通常为 "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(config.clip_path)

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

        # --- 文本处理 ---
        tokens = tokenizer(
            text, 
            return_tensors='pt', 
            padding='max_length', # 建议此处填充到 max_length 保证 dataloader 稳定性
            truncation=True, 
            max_length=config.max_seq_length
        )
        
        # --- 图像处理 (针对 CLIP) ---
        # CLIPProcessor 会处理 Resize (224x224), CenterCrop, 和 Normalize
        image_inputs = processor(images=image, return_tensors="pt")
        
        # 存储数据
        guids.append(guid)
        encoded_texts.append(tokens["input_ids"].squeeze(0))
        encoded_texts_mask.append(tokens["attention_mask"].squeeze(0))
        # pixel_values 的形状是 [1, 3, 224, 224]，我们需要 squeeze 掉第一维
        encoded_images.append(group["image"]) 
        
        encoded_labels.append(label)

    return guids, encoded_texts, encoded_texts_mask, encoded_images, encoded_labels

def create_dataloader(train_data_path, test_data_path, data_path, text_only=False, image_only=False):
    """
    创建训练、验证和测试数据加载器，包含验证集的划分

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
    # 获取原始训练集的所有字段
    t_guids, t_texts, t_masks, t_images, t_labels = encode_text_image(original_train_data)
    # 获取测试集的所有字段
    test_inputs = encode_text_image(original_test_data)

    # 2. 划分训练集和验证集
    # train_test_split 会按顺序返回: train_x1, val_x1, train_x2, val_x2 ...
    split_result = train_test_split(
        t_guids, t_texts, t_masks, t_images, t_labels, 
        test_size=0.2, 
        random_state=config.seed
    )
    
    # 重新组织切分后的数据
    train_data = [split_result[i] for i in range(0, len(split_result), 2)]
    valid_data = [split_result[i] for i in range(1, len(split_result), 2)]
    
    # 3. 定义三套标准变换
    # train_trans = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    #     transforms.RandomGrayscale(p=0.05),
    #     transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    #     transforms.RandomHorizontalFlip(p=0.5),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])

    train_trans = transforms.Compose([
        # 1. 稍微放大图片，为裁剪留出空间
        transforms.Resize(256), 
        # 2. 随机裁剪到 224，这是最核心的增强，能增加位置鲁棒性
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0)), 
        # 3. 水平翻转：最稳健的增强，不会改变情感语义
        transforms.RandomHorizontalFlip(p=0.5), 
        # 4. 转换为 Tensor
        transforms.ToTensor(),
        # 5. !!! 重要：必须使用 CLIP 特定的归一化参数，而不是 ImageNet 的
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073), 
            std=(0.26862954, 0.26130258, 0.27577711)
        )
    ])

    # val_test_trans = transforms.Compose([
    #     transforms.Resize((224, 224)), # 统一尺寸
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])

    val_test_trans = transforms.Compose([
        # 1. 统一缩放到 224x224
        transforms.Resize((224, 224)), 
        # 2. 转换为 Tensor
        transforms.ToTensor(),
        # 3. 必须使用 CLIP 专用的归一化参数
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073), 
            std=(0.26862954, 0.26130258, 0.27577711)
        )
    ])

    # 4. 创建 Dataset 实例
    train_datasets = MultiModalDataset(*train_data, transform=train_trans)
    valid_datasets = MultiModalDataset(*valid_data, transform=val_test_trans)
    test_datasets = MultiModalDataset(*test_inputs, transform=val_test_trans)
    # 5. 创建 DataLoader 实例
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