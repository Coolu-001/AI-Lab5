import torch
import numpy as np

class EarlyStopping:
    """
    如果验证集的损失在给定的周期内没有提升，那么启动早停
    """
    def __init__(self, patience=2, verbose=False, delta=0):
        """
        初始化早停类

        Args:
            patience (int, optional): 在验证集损失没有提升的情况下，等待的周期数。默认值为2。
            verbose (bool, optional): 是否打印早停信息。默认值为False。
            delta (float, optional): 认为验证集损失有提升的最小变化量。默认值为0。
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
    
    def __call__(self, val_loss, model):
        """
        每次验证集损失更新时调用此方法

        Args:
            val_loss (float): 当前的验证集损失
            model (torch.nn.Module): 当前训练的模型
        """
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.val_loss_min = val_loss
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.val_loss_min = val_loss
            self.counter = 0

    def get_best_loss(self):
        """
        返回最小的验证集损失

        Returns:
            float: 最小的验证集损失
        """
        return self.val_loss_min