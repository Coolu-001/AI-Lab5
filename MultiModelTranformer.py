import torch
import torch.nn as nn
from transformers import RobertaModel
from torchvision import models
from transformers import CLIPVisionModel, CLIPProcessor

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
            nn.Linear(self.roberta.config.hidden_size, config.middle_hidden_size),
            nn.BatchNorm1d(config.middle_hidden_size), # 平滑文本特征波动
            nn.ReLU(),
            nn.Dropout(config.roberta_dropout) # 建议 config 中设为 0.3
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
        # 提取 pooler_output 并强制重新排列内存
        feat = outputs["pooler_output"].contiguous()
        # 经过你定义的 transform (Linear + ReLU)
        return self.transform(feat).contiguous()
    
# class ImageModel(nn.Module):
#     """
#     图像模型类，用于处理图像数据。
#     """
#     def __init__(self, config):
#         """
#         初始化图像模型。

#         Args:
#             config: 配置对象，包含模型超参数。
#         """
#         super(ImageModel, self).__init__()
#         self.config = config
#         if config.resnet_type == 18:
#             self.full_resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
#         elif config.resnet_type == 34:
#             self.full_resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
#         elif config.resnet_type == 50:
#             self.full_resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
#         elif config.resnet_type == 101:
#             self.full_resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
#         elif config.resnet_type == 152:
#             self.full_resnet = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
#         else:
#             raise ValueError("Invalid resnet type")
#         self.resnet = nn.Sequential(
#             *(list(self.full_resnet.children())[:-1]),
#             nn.Flatten()
#         )
#         self.transform = nn.Sequential(
#             nn.Dropout(config.resnet_dropout),
#             nn.Linear(self.full_resnet.fc.in_features, config.middle_hidden_size),
#             nn.ReLU(inplace=True)
#         )

#         for param in self.full_resnet.parameters():
#             if config.fixed_image_param:
#                 param.requires_grad = False
#             else:
#                 param.requires_grad = True
        
#     def forward(self, images):
#         """
#         前向传播函数，处理图像输入。

#         Args:
#             images (torch.Tensor): 输入图像。

#         Returns:
#             torch.Tensor: 图像特征。
#         """
#         outputs = self.resnet(images)
#         outputs = self.transform(outputs)
#         return outputs

class ImageModel(nn.Module):
    """
    基于 CLIP 视觉基座的图像特征提取模块。
    CLIP (ViT) 的特征空间与文本天然对齐，适合多模态任务。
    """
    def __init__(self, config):
        """
        Args:
            config: 包含 clip_path (如 'openai/clip-vit-base-patch32') 的配置对象
        """
        super(ImageModel, self).__init__()
        # 加载预训练的 CLIP 视觉模型
        # config.clip_path 设为 "openai/clip-vit-base-patch32"
        self.model = CLIPVisionModel.from_pretrained(config.clip_path)
        self.transform = nn.Sequential(
            nn.Linear(768, config.middle_hidden_size),
            nn.BatchNorm1d(config.middle_hidden_size), # 平滑图像特征波动
            nn.ReLU(),
            nn.Dropout(config.clip_dropout) # 建议 config 中设为 0.3-0.4
        )
        # 冻结建议：显存较小或者数据量很少，可以取消下面两行的注释
        for param in self.model.parameters():
            param.requires_grad = True # 或者 False 冻结
            
    def forward(self, images):
        """
        Args:
            images (torch.Tensor): 经过预处理的图像张量，形状为 [batch_size, 3, 224, 224]
        Returns:
            torch.Tensor: CLIP 提取的全局池化特征 [batch_size, 768]
        """
        outputs = self.model(pixel_values=images.contiguous())
        
        # pooler_output 是经过投影后的全局特征（通常是 768 维）
        # 它是 [CLS] token 经过线性层后的结果，最适合做分类任务
        image_features = self.transform(outputs.pooler_output).contiguous()
        return image_features
    
class FusionModel(nn.Module):
    """
    基于Transformer编码器的多模态融合模型类，用于将文本和图像特征通过Transformer编码器融合并分类。
    """
    def __init__(self, config):
        """
        初始化多模态融合模型。
        Args:
            config: 配置对象，包含模型超参数。
        """
        super(FusionModel, self).__init__()
        self.config = config
        self.text_model = TextModel(config)
        self.image_model = ImageModel(config)
        self.attention = nn.TransformerEncoderLayer(
            d_model=config.middle_hidden_size * 3, 
            nhead=config.attention_nheads, 
            dropout=config.attention_dropout,
            batch_first=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(config.middle_hidden_size * 3, config.output_hidden_size),
            nn.BatchNorm1d(config.output_hidden_size),
            nn.ReLU(),
            nn.Dropout(config.fusion_dropout), # 核心抗过拟合位：建议设为 0.5
            nn.Linear(config.output_hidden_size, config.num_labels)
        )

        self.loss = nn.CrossEntropyLoss(label_smoothing=0.2)


    def forward(self, texts, texts_mask, images, labels=None):
        text_feature = self.text_model(texts, texts_mask)
        image_feature = self.image_model(images)
        
        # 1. 拼接特征 [Batch, 1536]
        # combined = torch.cat([text_feature, image_feature], dim=1).contiguous()
        # 修改 Fusion 逻辑建议：
        combined = torch.cat([text_feature, image_feature, text_feature * image_feature], dim=1).contiguous()
        
        # 2. 构造 Transformer 输入 [Batch, Seq_len=1, Dim=1536]
        # batch_first=True 要求 Batch 在第一维
        transformer_input = combined.unsqueeze(1).contiguous()

        # 3. 经过 Transformer
        # 得到的 attn_out 形状也是 [Batch, 1, 1536]
        attn_out = self.attention(transformer_input)
        
        # 4. 还原形状并进入分类器 [Batch, 1536]
        # 使用 reshape 代替 view，并紧跟 contiguous()，确保 backward 安全
        final_feature = attn_out.reshape(combined.shape[0], -1).contiguous()
        
        outputs = self.classifier(final_feature)
        pred_labels = torch.argmax(outputs, dim=1)

        if labels is not None:
            loss = self.loss(outputs, labels)
            return pred_labels, loss
        
        return pred_labels
    

class LateFusionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.text_model = TextModel(config)
        self.image_model = ImageModel(config)
        
        # 为文本和图像分别建立全连接层，输出类别数（假设是3）
        self.text_classifier = nn.Linear(config.middle_hidden_size, 3)
        self.image_classifier = nn.Linear(config.middle_hidden_size, 3)
        
        # 可学习的融合权重（初始化为 0.5/0.5）
        self.w_text = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.w_image = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        
        self.loss = nn.CrossEntropyLoss(label_smoothing=0.2)

    def forward(self, texts, texts_mask, images, labels=None):
        # 分别提取特征
        text_feat = self.text_model(texts, texts_mask) # [Batch, 768]
        image_feat = self.image_model(images)          # [Batch, 768]
        
        # 分别得到分类预测（Logits）
        text_logits = self.text_classifier(text_feat)
        image_logits = self.image_classifier(image_feat)
        
        # 末端融合：加权平均
        # 使用 sigmoid 确保权重在 0-1 之间且和为 1 (可选更简单的加权)
        combined_logits = self.w_text * text_logits + self.w_image * image_logits
        
        pred_labels = torch.argmax(combined_logits, dim=1)

        if labels is not None:
            # 你可以同时监督三个 Loss，让分支学得更好
            loss = self.loss(combined_logits, labels)
            return pred_labels, loss
        
        return pred_labels