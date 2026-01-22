import torch
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import wandb
from early_stopping import EarlyStopping
# 引入 Transformers 的调度器，它比原生 ReduceLROnPlateau 更适合微调任务
from transformers import get_linear_schedule_with_warmup
import os

class trainer_validator():
    """
    训练和验证模型的类，包含分层学习率、Warmup调度和早停机制。
    """
    def __init__(self, train_dataloader, config, model, device):
        """
        Args:
            train_dataloader: 用于计算总训练步数
            config: 配置对象
            model: 模型
            device: 设备
        """
        self.config = config
        self.model = model.to(device)
        self.device = device
        
        # 1. 计算总训练步数 (Total Steps) 用于 Scheduler
        self.total_steps = len(train_dataloader) * config.epochs
        
        # 1. 定义完全互斥的三个核心集合
        # 获取所有参数名，用于后续排除
        all_params_names = [n for n, p in self.model.named_parameters()]

        # 提取预训练分支（使用对应的低学习率）
        bert_params = [p for n, p in self.model.named_parameters() if "text_model.roberta" in n]
        clip_params = [p for n, p in self.model.named_parameters() if "image_model.model" in n]

        # 提取那两个关键的融合权重（使用高学习率）
        weight_params = [p for n, p in self.model.named_parameters() if "w_text" in n or "w_image" in n]

        # 提取剩余所有参数（分类器分类头、各种 LayerNorm 等）
        # 确保排除掉上面已经提取过的所有东西
        other_params = [p for n, p in self.model.named_parameters() 
                        if not any(k in n for k in ["text_model.roberta", "image_model.model", "w_text", "w_image"])]

        # 2. 构造优化器参数组（确保无重叠）
        optimizer_grouped_parameters = [
            # 文本分支 (RoBERTa)
            {"params": bert_params, "lr": self.config.roberta_lr, "weight_decay": self.config.weight_decay},
            
            # 图像分支 (CLIP)
            {"params": clip_params, "lr": self.config.clip_lr, "weight_decay": self.config.weight_decay},
            
            # 关键：末端融合权重 (给它一个极大的学习率 0.01 甚至 0.1)
            {"params": weight_params, "lr": 1e-2, "weight_decay": 0.0},
            
            # 普通分类层和融合层
            {"params": other_params, "lr": self.config.lr, "weight_decay": self.config.weight_decay}
        ]
        # 3. 初始化优化器 (AdamW 是微调的首选)
        self.optimizer = AdamW(optimizer_grouped_parameters, eps=1e-8) 

        # 4. 初始化带有 Warmup 的线性调度器
        # 预热步数通常设为总步数的 20%
        num_warmup_steps = int(0.2 * self.total_steps)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, 
            num_warmup_steps=num_warmup_steps, 
            num_training_steps=self.total_steps
        )

        # 5. 早停机制 (建议以 val_loss 为主，patience 设为 5-7)
        self.early_stopping = EarlyStopping(patience=5, delta=0, verbose=True)
        self.bad_cases = []

    def _save_bad_cases_to_file(self, epoch):
        with open("bad_cases.txt", "w") as f:
            for case in self.bad_cases:
                f.write(f"Epoch: {epoch}, GUID: {case['guid']}, True: {case['true_label']}, Pred: {case['pred_label']}\n")

    def train(self, train_dataloader, val_dataloader, num_epochs, evaluate_every=1):
        iteration = 0
        best_val_accuracy = 0
        
        for epoch in range(num_epochs):
            self.model.train()
            train_loss_total = 0
            train_pred, train_true = [], []
            
            pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch in pbar:
                iteration += 1
                guids, texts, texts_mask, images, labels = batch
                texts, texts_mask, images, labels = texts.to(self.device), texts_mask.to(self.device), images.to(self.device), labels.to(self.device)

                # 前向传播
                train_pred_labels, loss = self.model(texts, texts_mask, images, labels)

                # 反向传播与优化
                self.optimizer.zero_grad()
                loss.backward()
                
                # 关键：梯度裁剪，防止多模态反向传播时的梯度爆炸导致的曲线不稳
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                self.scheduler.step() # 每个 step 更新学习率，而非每个 epoch

                train_loss_total += loss.item()
                train_true.extend(labels.cpu().numpy())
                train_pred.extend(train_pred_labels.cpu().numpy())

                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{self.optimizer.param_groups[-1]["lr"]:.2e}'})
                
                if iteration % 10 == 0:
                    wandb.log({"iteration": iteration, "train_loss": loss.item(), "lr": self.optimizer.param_groups[-1]["lr"]})

            # 计算本轮训练指标
            epoch_train_acc = accuracy_score(train_true, train_pred)
            wandb.log({
                "epoch": epoch + 1,
                "train_accuracy": epoch_train_acc,
                "train_f1": f1_score(train_true, train_pred, average="weighted")
            })

            # 验证环节
            if (epoch + 1) % evaluate_every == 0:
                with torch.no_grad():
                    self.model.eval()
                    val_loss_total, val_pred, val_true = 0.0, [], []
                    current_epoch_bad_cases = []

                    for batch in tqdm(val_dataloader, desc="Evaluating"):
                        guids, texts, texts_mask, images, labels = batch
                        texts, texts_mask, images, labels = texts.to(self.device), texts_mask.to(self.device), images.to(self.device), labels.to(self.device)
                        
                        val_pred_labels, loss = self.model(texts, texts_mask, images, labels)
                        val_loss_total += loss.item()
                        
                        preds_cpu, labels_cpu = val_pred_labels.cpu().numpy(), labels.cpu().numpy()
                        val_true.extend(labels_cpu)
                        val_pred.extend(preds_cpu)
                        
                        for i in range(len(labels_cpu)):
                            if preds_cpu[i] != labels_cpu[i]:
                                current_epoch_bad_cases.append({"guid": guids[i], "true_label": int(labels_cpu[i]), "pred_label": int(preds_cpu[i])})

                    val_acc = accuracy_score(val_true, val_pred)
                    val_epoch_loss = val_loss_total / len(val_dataloader)
                    
                    if val_acc > best_val_accuracy:
                        best_val_accuracy = val_acc
                        self.bad_cases = current_epoch_bad_cases
                        self._save_bad_cases_to_file(epoch + 1)

                    # 记录验证日志
                    wandb.log({
                        "val_loss": val_epoch_loss,
                        "val_accuracy": val_acc,
                        "val_f1": f1_score(val_true, val_pred, average="weighted")
                    })

                    # 在验证结束、打印 val_accuracy 之后添加
                    if hasattr(self.model, 'w_text') and hasattr(self.model, 'w_image'):
                        # 使用 .item() 获取标量值
                        w_t = self.model.w_text.item()
                        w_i = self.model.w_image.item()
                        print(f"\n[Modality Weights] Text: {w_t:.4f}, Image: {w_i:.4f}")
                        
                        # 同时记录到 wandb 方便观察曲线
                        wandb.log({
                            "weight_text": w_t,
                            "weight_image": w_i
                        })

                    # 早停判定：主要监控 val_loss 以防止过拟合
                    self.early_stopping(val_epoch_loss, self.model)
                    if self.early_stopping.early_stop:
                        print(f"Early stopping triggered at Epoch {epoch+1}")
                        break

        return best_val_accuracy