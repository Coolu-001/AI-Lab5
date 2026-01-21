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
    def __init__(self, config):
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
            nn.Dropout(config.fusion_dropout), 
            nn.Linear(config.output_hidden_size, config.num_labels)
        )
        self.loss = nn.CrossEntropyLoss(label_smoothing=0.2)

    # 修改参数名以对齐 DataLoader
    def forward(self, input_ids, attention_mask, images, labels=None):
        # 传入对应的参数
        text_feature = self.text_model(input_ids, attention_mask)
        image_feature = self.image_model(images)
        
        # 拼接特征
        combined = torch.cat([text_feature, image_feature, text_feature * image_feature], dim=1).contiguous()
        
        # 构造 Transformer 输入
        transformer_input = combined.unsqueeze(1).contiguous()

        # 经过 Transformer
        attn_out = self.attention(transformer_input)
        
        # 还原形状并进入分类器
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
        
        self.text_classifier = nn.Linear(config.middle_hidden_size, 3)
        self.image_classifier = nn.Linear(config.middle_hidden_size, 3)
        
        self.w_text = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.w_image = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        
        self.loss = nn.CrossEntropyLoss(label_smoothing=0.2)

    # 修改参数名以对齐 DataLoader
    def forward(self, input_ids, attention_mask, images, labels=None):
        # 分别提取特征
        text_feat = self.text_model(input_ids, attention_mask) 
        image_feat = self.image_model(images)          
        
        # 分别得到预测
        text_logits = self.text_classifier(text_feat)
        image_logits = self.image_classifier(image_feat)
        
        # 权重归一化
        weights = torch.softmax(torch.stack([self.w_text, self.w_image]), dim=0)
    
        # 加权平均融合
        combined_logits = weights[0] * text_logits + weights[1] * image_logits
        
        pred_labels = torch.argmax(combined_logits, dim=1)

        if labels is not None:
            loss = self.loss(combined_logits, labels)
            return pred_labels, loss
        
        return pred_labels
    

class DynamicGatedFusionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.text_model = TextModel(config)
        self.image_model = ImageModel(config)
        
        # 1. 投影层：将不同模态映射到统一特征空间，并增强非线性
        self.text_proj = nn.Sequential(
            nn.Linear(config.middle_hidden_size, config.middle_hidden_size),
            nn.BatchNorm1d(config.middle_hidden_size),
            nn.GELU(),
            nn.Dropout(config.roberta_dropout)
        )
        self.image_proj = nn.Sequential(
            nn.Linear(config.middle_hidden_size, config.middle_hidden_size),
            nn.BatchNorm1d(config.middle_hidden_size),
            nn.GELU(),
            nn.Dropout(config.clip_dropout)
        )

        # 2. 动态门控网络：输入拼接特征，输出一个 0~1 之间的权重标量
        self.gate_layer = nn.Sequential(
            nn.Linear(config.middle_hidden_size * 2, config.middle_hidden_size // 2),
            nn.Tanh(),
            nn.Linear(config.middle_hidden_size // 2, 1),
            nn.Sigmoid()
        )

        # 3. 独立分类头
        self.text_classifier = nn.Linear(config.middle_hidden_size, 3)
        self.image_classifier = nn.Linear(config.middle_hidden_size, 3)
        
        # 4. 损失函数加入 Label Smoothing
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0.2)

    def forward(self, input_ids, attention_mask, images, labels=None):
        # 2. 提取特征
        t_feat = self.text_model(input_ids, attention_mask)
        i_feat = self.image_model(images)
        
        # 应用投影层 (投影层能把 CLIP 和 RoBERTa 的特征拉到同一个量级)
        t_feat_p = self.text_proj(t_feat)
        i_feat_p = self.image_proj(i_feat)
        
        # 计算各自的预测
        text_logits = self.text_classifier(t_feat_p)
        image_logits = self.image_classifier(i_feat_p)
        
        # 3. 动态门控逻辑
        gate_input = torch.cat([t_feat_p, i_feat_p], dim=-1)
        alpha = self.gate_layer(gate_input) 
        
        # 记录 alpha 指标，用于观察文本增强是否让文本分支更“自信”了
        alpha_mean = alpha.mean().item()
        alpha_std = alpha.std().item() if alpha.size(0) > 1 else 0.0

        # 4. 融合 Logits
        # 这里是核心：alpha 动态调节两个模态的话语权
        combined_logits = alpha * text_logits + (1 - alpha) * image_logits
        
        pred_labels = torch.argmax(combined_logits, dim=1)

        if labels is not None:
            loss = self.loss_fn(combined_logits, labels)
            # 返回 pred, loss 和 门控统计信息
            return pred_labels, loss, (alpha_mean, alpha_std)
        
        return pred_labels, (alpha_mean, alpha_std)