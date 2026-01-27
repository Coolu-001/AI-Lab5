import torch
import torch.nn as nn
from transformers import RobertaModel
from torchvision import models
from transformers import CLIPVisionModel, CLIPProcessor
import torch.nn.functional as F

class TextModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 加载预训练模型，开启 hidden_states 输出
        self.roberta = RobertaModel.from_pretrained(
            config.roberta_path, 
            output_hidden_states=True 
        )

        # 1. 自动学习最后 4 层的权重分配
        self.layer_weights = nn.Parameter(torch.ones(4))
        
        # 2. 这里的 transform 层增加了维度适配
        # 考虑到我们将 CLS 和 Mean Pooling 拼接，输入维度变为 hidden_size * 2
        self.transform = nn.Sequential(
            nn.Linear(self.roberta.config.hidden_size * 2, config.middle_hidden_size),
            nn.LayerNorm(config.middle_hidden_size),
            nn.GELU(),
            nn.Dropout(config.roberta_dropout)
        )

        # 根据配置决定是否冻结预训练层
        for p in self.roberta.parameters():
            p.requires_grad = not config.fixed_text_param

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        # --- 核心修改：多层特征融合 ---
        # 提取最后 4 层 [4, Batch, Seq, Hidden]
        last_four_layers = torch.stack(outputs.hidden_states[-4:], dim=0) 
        
        # 计算归一化权重并进行加权求和
        weights = F.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)
        weighted_feat = (last_four_layers * weights).sum(0) 

        # --- 特征池化 ---
        # 1. CLS Token 特征：代表整句全局语义
        cls_feat = weighted_feat[:, 0, :] 
        
        # 2. Weighted Mean Pooling：代表词义平均分布
        mask = attention_mask.unsqueeze(-1).float()
        mean_feat = (weighted_feat * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        
        # --- 拼接与输出 ---
        # 拼接后的维度是 hidden_size * 2
        combined_feat = torch.cat([cls_feat, mean_feat], dim=1)
        
        return self.transform(combined_feat)
    
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
        self.loss = nn.CrossEntropyLoss(label_smoothing=0.2)

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
        
class TransformerFusionModel(nn.Module):
    """
    基于Transformer编码器的多模态融合模型类，用于将文本和图像特征通过Transformer编码器融合并分类。
    """
    def __init__(self, config):
        """
        初始化多模态融合模型。

        Args:
            config: 配置对象，包含模型超参数。
        """
        super(TransformerFusionModel, self).__init__()
        self.config = config
        self.text_model = TextModel(config)
        self.image_model = ImageModel(config)
        self.attention = nn.TransformerEncoderLayer(
            d_model=config.middle_hidden_size * 2, 
            nhead=config.attention_nheads, 
            dropout=config.attention_dropout
        )
        self.classifier = nn.Sequential(
            nn.Dropout(config.fusion_dropout),
            nn.Linear(config.middle_hidden_size * 2, config.output_hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(config.fusion_dropout),
            nn.Linear(config.output_hidden_size, config.num_labels)
        )

        self.loss = nn.CrossEntropyLoss(label_smoothing=0.2)

    def forward(self, texts, texts_mask, images, labels):
        """
        前向传播函数，处理文本和图像输入，并通过Transformer编码器进行特征融合。

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
        text_image_attention = self.attention(torch.cat([text_feature.unsqueeze(0), image_feature.unsqueeze(0)], dim=2)).squeeze()
        outputs = self.classifier(text_image_attention)
        pred_labels = torch.argmax(outputs, dim=1)

        if self.training or labels is not None:
            loss = self.loss(outputs, labels)
            return pred_labels, loss
        else:
            return pred_labels

class CrossAttentionFusionModel(nn.Module):
    """
    基于交叉注意力机制的多模态融合模型类，用于将文本和图像特征通过交叉注意力机制融合并分类。
    """
    def __init__(self, config):
        """
        初始化交叉注意力融合模型。

        Args:
            config: 配置对象，包含模型超参数。
        """
        super(CrossAttentionFusionModel, self).__init__()
        self.config = config
        self.text_model = TextModel(config)
        self.image_model = ImageModel(config)
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.middle_hidden_size,
            num_heads=config.attention_nheads,
            dropout=config.attention_dropout
        )
        
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
        前向传播函数，处理文本和图像输入，并通过交叉注意力机制进行特征融合。

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
        text_feature = text_feature.unsqueeze(0)
        image_feature = image_feature.unsqueeze(0)
        
        text_attention_output, _ = self.cross_attention(
            query=text_feature,
            key=image_feature,
            value=image_feature
        )
        
        image_attention_output, _ = self.cross_attention(
            query=image_feature,
            key=text_feature,
            value=text_feature
        )
        
        text_attention_output = text_attention_output.squeeze(0)
        image_attention_output = image_attention_output.squeeze(0)
        fused_feature = torch.cat([text_attention_output, image_attention_output], dim=1)
        
        outputs = self.classifier(fused_feature)
        pred_labels = torch.argmax(outputs, dim=1)

        if self.training or labels is not None:
            loss = self.loss(outputs, labels)
            return pred_labels, loss
        else:
            return pred_labels
        
# class FusionModel(nn.Module):
#     """
#     基于Transformer编码器的多模态融合模型类，用于将文本和图像特征通过Transformer编码器融合并分类。
#     """
#     def __init__(self, config):
#         """
#         初始化多模态融合模型。
#         Args:
#             config: 配置对象，包含模型超参数。
#         """
#         super(FusionModel, self).__init__()
#         self.config = config
#         self.text_model = TextModel(config)
#         self.image_model = ImageModel(config)
#         self.attention = nn.TransformerEncoderLayer(
#             d_model=config.middle_hidden_size * 3, 
#             nhead=config.attention_nheads, 
#             dropout=config.attention_dropout,
#             batch_first=True
#         )
#         self.classifier = nn.Sequential(
#             nn.Linear(config.middle_hidden_size * 3, config.output_hidden_size),
#             nn.BatchNorm1d(config.output_hidden_size),
#             nn.ReLU(),
#             nn.Dropout(config.fusion_dropout), # 核心抗过拟合位：建议设为 0.5
#             nn.Linear(config.output_hidden_size, config.num_labels)
#         )

#         self.loss = nn.CrossEntropyLoss(label_smoothing=0.2)


#     def forward(self, texts, texts_mask, images, labels=None):
#         text_feature = self.text_model(texts, texts_mask)
#         image_feature = self.image_model(images)
        
#         # 1. 拼接特征 [Batch, 1536]
#         # combined = torch.cat([text_feature, image_feature], dim=1).contiguous()
#         # 修改 Fusion 逻辑建议：
#         combined = torch.cat([text_feature, image_feature, text_feature * image_feature], dim=1).contiguous()
        
#         # 2. 构造 Transformer 输入 [Batch, Seq_len=1, Dim=1536]
#         # batch_first=True 要求 Batch 在第一维
#         transformer_input = combined.unsqueeze(1).contiguous()

#         # 3. 经过 Transformer
#         # 得到的 attn_out 形状也是 [Batch, 1, 1536]
#         attn_out = self.attention(transformer_input)
        
#         # 4. 还原形状并进入分类器 [Batch, 1536]
#         # 使用 reshape 代替 view，并紧跟 contiguous()，确保 backward 安全
#         final_feature = attn_out.reshape(combined.shape[0], -1).contiguous()
        
#         outputs = self.classifier(final_feature)
#         pred_labels = torch.argmax(outputs, dim=1)

#         if labels is not None:
#             loss = self.loss(outputs, labels)
#             return pred_labels, loss
        
#         return pred_labels
    
