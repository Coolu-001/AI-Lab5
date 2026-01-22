from config import config
import torch
import numpy as np
from train_validate import trainer_validator
from MultiModelTranformer import FusionModel
from MultiModelConcat import ConcatFusionModel
from MultiModelCrossAttention import CrossAttentionFusionModel
from load_dataset import create_dataloader
import wandb
from datetime import datetime
import argparse

def parse_args():
    """
    解析命令行参数，用于配置训练的超参数。

    Returns:
        argparse.Namespace: 包含所有命令行参数的对象。
    """
    parser = argparse.ArgumentParser(description="Training parameters")
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--roberta_dropout', type=float, default=0.15, help='Dropout rate for RoBERTa')
    parser.add_argument('--roberta_lr', type=float, default=1e-5, help='Learning rate for RoBERTa')
    parser.add_argument('--middle_hidden_size', type=int, default=256, help='Hidden size for middle layer')
    parser.add_argument('--resnet_type', type=int, default=101, help='ResNet type (18, 34, 50, 101, 152)')
    parser.add_argument('--resnet_dropout', type=float, default=0.15, help='Dropout rate for ResNet')
    parser.add_argument('--resnet_lr', type=float, default=1e-5, help='Learning rate for ResNet')
    parser.add_argument('--attention_nheads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--attention_dropout', type=float, default=0.15, help='Dropout rate for attention layer')
    parser.add_argument('--fusion_dropout', type=float, default=0.15, help='Dropout rate for fusion layer')
    parser.add_argument('--output_hidden_size', type=int, default=256, help='Hidden size for output layer')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay for optimizer')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate for optimizer')

    args = parser.parse_args()
    return args

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    
def set_seed(seed):
    """
    设置随机种子，确保实验的可重复性。

    Args:
        seed (int): 随机种子值。
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    args = parse_args()
    config.batch_size = args.batch_size
    config.roberta_dropout = args.roberta_dropout
    config.roberta_lr = args.roberta_lr
    config.middle_hidden_size = args.middle_hidden_size
    config.resnet_type = args.resnet_type
    config.resnet_dropout = args.resnet_dropout
    config.resnet_lr = args.resnet_lr
    config.attention_nheads = args.attention_nheads
    config.attention_dropout = args.attention_dropout
    config.fusion_dropout = args.fusion_dropout
    config.output_hidden_size = args.output_hidden_size
    config.weight_decay = args.weight_decay
    config.lr = args.lr

    seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 327]
    val_accuracies = []

    for seed in seeds:
        set_seed(seed)
        wandb.init(
            project="AILAB5-SEED-1",
            name=f"SEED_{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "batch_size": config.batch_size,
                "roberta_dropout": config.roberta_dropout,
                "roberta_lr": config.roberta_lr,
                "middle_hidden_size": config.middle_hidden_size,
                "resnet_type": config.resnet_type,
                "resnet_dropout": config.resnet_dropout,
                "resnet_lr": config.resnet_lr,
                "attention_nheads": config.attention_nheads,
                "attention_dropout": config.attention_dropout,
                "fusion_dropout": config.fusion_dropout,
                "output_hidden_size": config.output_hidden_size,
                "weight_decay": config.weight_decay,
                "lr": config.lr,
                "seed": seed
            },
            reinit=True,
            allow_val_change=True
        )
        train_dataloader, valid_dataloader, test_dataloader = create_dataloader(
            config.train_data_path, 
            config.test_data_path, 
            config.data_path, 
            text_only=False, 
            image_only=False
        )
        model = FusionModel(config)
        trainer = trainer_validator(config, model, device)
        val_accuracy = trainer.train(train_dataloader, valid_dataloader, config.epochs, evaluate_every=1)
        val_accuracies.append(val_accuracy)
        wandb.finish()

    mean_accuracy = np.mean(val_accuracies)
    variance_accuracy = np.var(val_accuracies)
    print(f"Mean Validation Accuracy: {mean_accuracy}")
    print(f"Variance of Validation Accuracy: {variance_accuracy}")