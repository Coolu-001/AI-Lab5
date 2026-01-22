import torch
import torch.nn as nn
from transformers import RobertaModel
from torchvision import models

class TextModel(nn.Module):
    """
    文本模型类，用于处理文本数据。
    """
    def __init__(self, config):
        """
        初始化文本模型。

        Args:
            config: 配置对象，包含模型超参数。
        """
        super(TextModel, self).__init__()
        self.config = config
        self.roberta = RobertaModel.from_pretrained(config.roberta_path)
        self.transform = nn.Sequential(
            nn.Dropout(config.roberta_dropout),
            nn.Linear(self.roberta.config.hidden_size, config.middle_hidden_size),
            nn.ReLU(inplace=True)
        )

        for param in self.roberta.parameters():
            if config.fixed_text_param:
                param.requires_grad = False
            else:
                param.requires_grad = True
    
    def forward(self, input_ids, attention_mask):
        """
        前向传播函数，处理文本输入。

        Args:
            input_ids (torch.Tensor): 输入文本的token IDs。
            attention_mask (torch.Tensor): 注意力掩码，用于指示哪些token是有效的。

        Returns:
            torch.Tensor: 文本特征。
        """
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        outputs = self.transform(outputs["pooler_output"])
        return outputs

class ImageModel(nn.Module):
    """
    图像模型类，用于处理图像数据。
    """
    def __init__(self, config):
        """
        初始化图像模型。

        Args:
            config: 配置对象，包含模型超参数。
        """
        super(ImageModel, self).__init__()
        self.config = config
        if config.resnet_type == 18:
            self.full_resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        elif config.resnet_type == 34:
            self.full_resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        elif config.resnet_type == 50:
            self.full_resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif config.resnet_type == 101:
            self.full_resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        elif config.resnet_type == 152:
            self.full_resnet = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        else:
            raise ValueError("Invalid resnet type")
        self.resnet = nn.Sequential(
            *(list(self.full_resnet.children())[:-1]),
            nn.Flatten()
        )
        self.transform = nn.Sequential(
            nn.Dropout(config.resnet_dropout),
            nn.Linear(self.full_resnet.fc.in_features, config.middle_hidden_size),
            nn.ReLU(inplace=True)
        )

        for param in self.full_resnet.parameters():
            if config.fixed_image_param:
                param.requires_grad = False
            else:
                param.requires_grad = True
        
    def forward(self, images):
        """
        前向传播函数，处理图像输入。

        Args:
            images (torch.Tensor): 输入图像。

        Returns:
            torch.Tensor: 图像特征。
        """
        outputs = self.resnet(images)
        outputs = self.transform(outputs)
        return outputs

class ConcatFusionModel(nn.Module):
    """
    多模态融合模型类，用于将文本和图像特征进行拼接并分类。
    """
    def __init__(self, config):
        """
        初始化多模态融合模型。

        Args:
            config: 配置对象，包含模型超参数。
        """
        super(ConcatFusionModel, self).__init__()
        self.config = config
        self.text_model = TextModel(config)
        self.image_model = ImageModel(config)
        self.classifier = nn.Sequential(
            nn.Dropout(config.fusion_dropout),
            nn.Linear(config.middle_hidden_size * 2, config.output_hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(config.fusion_dropout),
            nn.Linear(config.output_hidden_size, config.num_labels)
        )
        self.loss = nn.CrossEntropyLoss()

    def forward(self, texts, texts_mask, images, labels):
        """
        前向传播函数，处理文本和图像输入。

        Args:
            texts (torch.Tensor): 输入文本的token IDs。
            texts_mask (torch.Tensor): 文本的注意力掩码。
            images (torch.Tensor): 输入图像。
            labels (torch.Tensor): 真实标签（可选，用于训练时计算损失）。

        Returns:
            tuple: 包含预测标签和损失（如果提供了标签）。
        """
        text_feature = self.text_model(texts, texts_mask)
        image_feature = self.image_model(images)
        text_image_feature = torch.cat([text_feature, image_feature], dim=1)
        outputs = self.classifier(text_image_feature)
        pred_labels = torch.argmax(outputs, dim=1)

        if self.training or labels is not None:
            loss = self.loss(outputs, labels)
            return pred_labels, loss
        else:
            return pred_labels