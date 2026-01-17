import torch
from torch.optim import AdamW, Adam, SGD
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import wandb
from early_stopping import EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau

class trainer_validator():
    """
    训练和验证模型的类，包含训练循环、验证循环、优化器配置和早停机制。
    """
    def __init__(self, config, model, device):
        """
        初始化训练器和验证器。
        Args:
            config: 配置对象，包含训练超参数。
            model: 需要训练的模型。
            device: 训练设备（如CPU或GPU）。
        """
        self.config = config
        self.model = model.to(device)
        self.device = device

        bert_params = set(self.model.text_model.roberta.parameters())
        resnet_params = set(self.model.image_model.full_resnet.parameters())
        other_params = list(set(self.model.parameters()) - bert_params - resnet_params)
        no_decay = ["bias", "LayerNorm.weight"]
        params = [
            {
                "params": [p for n, p in self.model.text_model.roberta.named_parameters() if not any(nd in n for nd in no_decay)],
                "lr": self.config.roberta_lr,
                "weight_decay": self.config.weight_decay
            },
            {
                "params": [p for n, p in self.model.text_model.roberta.named_parameters() if any(nd in n for nd in no_decay)],
                "lr": self.config.roberta_lr,
                "weight_decay": 0.0
            },
            {
                "params": [p for n, p in self.model.image_model.full_resnet.named_parameters() if not any(nd in n for nd in no_decay)],
                "lr": self.config.resnet_lr,
                "weight_decay": self.config.weight_decay
            },
            {
                "params": [p for n, p in self.model.image_model.full_resnet.named_parameters() if any(nd in n for nd in no_decay)],
                "lr": self.config.resnet_lr,
                "weight_decay": 0.0
            },
            {
                "params": other_params,
                "lr": self.config.lr,
                "weight_decay": self.config.weight_decay
            },
        ]
        # self.optimizer = AdamW(params, lr=self.config.lr)
        self.optimizer = Adam(params, lr=self.config.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=2, threshold=0.01)
        self.early_stopping = EarlyStopping(patience=10, delta = 0.01, verbose=True)
    
    def train(self, train_dataloader, val_dataloader, num_epochs, evaluate_every=1):
        """
        训练和验证模型。

        Args:
            train_dataloader: 训练数据加载器。
            val_dataloader: 验证数据加载器。
            num_epochs (int): 训练的总轮数。
            evaluate_every (int, optional): 每隔多少轮进行一次验证。默认值为1。

        Returns:
            float: 验证集上的最佳准确率。
        """
        iteration = 0
        train_accuracy = []
        train_f1 = []
        train_precision = []
        train_recall = []
        val_accuracy = []
        val_f1 = []
        val_precision = []
        val_recall = []
        best_val_accuracy = 0
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = []
            train_pred = []
            train_true = []
            for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                iteration += 1
                guids, texts, texts_mask, images, labels = batch
                texts = texts.to(self.device)
                texts_mask = texts_mask.to(self.device)
                images = images.to(self.device)
                labels = labels.to(self.device)

                train_pred_labels, loss = self.model(texts, texts_mask, images, labels)

                train_loss.append(loss.item())
                train_true.extend(labels.cpu().numpy())
                train_pred.extend(train_pred_labels.cpu().numpy())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                wandb.log({
                    "iteration": iteration,
                    "train_loss": loss.item()
                })

            train_accuracy.append(accuracy_score(train_true, train_pred))
            train_f1.append(f1_score(train_true, train_pred, average="weighted"))
            train_precision.append(precision_score(train_true, train_pred, average="weighted", zero_division=0))
            train_recall.append(recall_score(train_true, train_pred, average="weighted"))
            wandb.log({
                "epoch": epoch,
                "train_accuracy": train_accuracy[-1],
                "train_f1": train_f1[-1],
                "train_precision": train_precision[-1],
                "train_recall": train_recall[-1]
            })

            if (epoch + 1) % evaluate_every == 0:
                with torch.no_grad():
                    self.model.eval()
                    val_loss = 0.0
                    val_pred = []
                    val_true = []
                    for batch in tqdm(val_dataloader, desc="Evaluating"):
                        guids, texts, texts_mask, images, labels = batch
                        texts = texts.to(self.device)
                        texts_mask = texts_mask.to(self.device)
                        images = images.to(self.device)
                        labels = labels.to(self.device)
                        val_pred_labels, loss = self.model(texts, texts_mask, images, labels)
                        val_loss += loss.item()
                        val_true.extend(labels.cpu().numpy())
                        val_pred.extend(val_pred_labels.cpu().numpy())
                    
                    val_accuracy.append(accuracy_score(val_true, val_pred))
                    val_f1.append(f1_score(val_true, val_pred, average="weighted"))
                    val_precision.append(precision_score(val_true, val_pred, average="weighted", zero_division=0))
                    val_recall.append(recall_score(val_true, val_pred, average="weighted"))
                    if val_accuracy[-1] > best_val_accuracy:
                        best_val_accuracy = val_accuracy[-1]
                    val_epoch_loss = val_loss / len(val_dataloader)
                    self.scheduler.step(val_epoch_loss)
                    self.early_stopping(val_epoch_loss, self.model)
                    wandb.log({
                        "val_loss": val_epoch_loss,
                        "val_accuracy": val_accuracy[-1],
                        "val_f1": val_f1[-1],
                        "val_precision": val_precision[-1],
                        "val_recall": val_recall[-1]
                    })
                    if self.early_stopping.early_stop:
                        print("Early stopping triggered.")
                        break
        return best_val_accuracy    
