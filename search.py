import torch
import numpy as np
from train_validate import trainer_validator
from MultiModelTranformer import FusionModel
from load_dataset import create_dataloader
import wandb
from datetime import datetime
import argparse
from itertools import product
from config import config

def parse_args():
    """
    解析命令行参数，用于配置训练的超参数。

    Returns:
        argparse.Namespace: 包含所有命令行参数的对象。
    """
    parser = argparse.ArgumentParser(description="Training parameters")
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--roberta_dropout', type=float, default=0.8, help='Dropout rate for RoBERTa')
    parser.add_argument('--roberta_lr', type=float, default=1e-5, help='Learning rate for RoBERTa')
    parser.add_argument('--middle_hidden_size', type=int, default=256, help='Hidden size for middle layer')
    parser.add_argument('--resnet_type', type=int, default=18, help='ResNet type (18, 34, 50, 101, 152)')
    parser.add_argument('--resnet_dropout', type=float, default=0.8, help='Dropout rate for ResNet')
    parser.add_argument('--resnet_lr', type=float, default=1e-5, help='Learning rate for ResNet')
    parser.add_argument('--attention_nheads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--attention_dropout', type=float, default=0.8, help='Dropout rate for attention layer')
    parser.add_argument('--fusion_dropout', type=float, default=0.8, help='Dropout rate for fusion layer')
    parser.add_argument('--output_hidden_size', type=int, default=128, help='Hidden size for output layer')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay for optimizer')
    parser.add_argument('--lr', type=float, default=1e-6, help='Learning rate for optimizer')
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

def generate_hyperparameter_grid():
    """生成超参数网格"""
    return {
        "batch_size": [16, 32],
        "dropout": [0, 0.05, 0.1, 0.15, 0.2],
        "lr": [1e-5, 2e-5],
        "output_hidden_size": [64, 128, 256],
        "resnet_type": [18, 34, 50],
        "attention_nheads": [8]
    }

if __name__ == "__main__":
    args = parse_args()
    set_seed(config.seed)

    best_accuracy = 0
    best_hyperparameters = {}

    hyperparameter_grid = generate_hyperparameter_grid()
    hyperparameter_combinations = list(product(*hyperparameter_grid.values()))

    for trial, hyperparameters in enumerate(hyperparameter_combinations):
        print(f"Trial {trial + 1}/{len(hyperparameter_combinations)}")
        hyperparameters = dict(zip(hyperparameter_grid.keys(), hyperparameters))

        config.batch_size = hyperparameters["batch_size"]
        config.roberta_dropout = hyperparameters["dropout"]
        config.resnet_dropout = hyperparameters["dropout"]
        config.attention_dropout = hyperparameters["dropout"]
        config.fusion_dropout = hyperparameters["dropout"]
        config.roberta_lr = hyperparameters["lr"]
        config.resnet_lr = hyperparameters["lr"]
        config.lr = hyperparameters["lr"]
        config.output_hidden_size = hyperparameters["output_hidden_size"]
        config.middle_hidden_size = hyperparameters["output_hidden_size"]
        config.resnet_type = hyperparameters["resnet_type"]
        config.attention_nheads = hyperparameters["attention_nheads"]

        wandb.init(
            project="AILAB5-SEARCH",
            name=f"Trial_{trial + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=hyperparameters,
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
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_hyperparameters = hyperparameters
        wandb.finish()

    print("Best Validation Accuracy:", best_accuracy)
    print("Best Hyperparameters:", best_hyperparameters)