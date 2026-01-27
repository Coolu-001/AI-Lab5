import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1' # 针对 Mac MPS 报错的强制回退
from config import config
import torch
import numpy as np
from train_validate import trainer_validator
from FusionModels import TransformerFusionModel, ConcatFusionModel
from load_dataset import create_dataloader
import wandb
from datetime import datetime
import argparse

#     parser.add_argument('--resnet_type', type=int, default=18, help='ResNet type (18, 34, 50, 101, 152)')
#     parser.add_argument('--resnet_dropout', type=float, default=0.15, help='Dropout rate for ResNet')
#     parser.add_argument('--resnet_lr', type=float, default=1e-5, help='Learning rate for ResNet')


def parse_args():
    parser = argparse.ArgumentParser(description="Training parameters")
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    
    # RoBERTa 参数保持现状
    parser.add_argument('--roberta_dropout', type=float, default=0.4, help='Dropout for RoBERTa')
    parser.add_argument('--roberta_lr', type=float, default=2e-5, help='LR for RoBERTa')
    
    # !!! CLIP 关键修改
    parser.add_argument('--middle_hidden_size', type=int, default=768, help='Must be 768 for CLIP-ViT-Base')
    parser.add_argument('--clip_lr', type=float, default=1e-6, help='Very small LR for CLIP fine-tuning')
    parser.add_argument('--clip_dropout', type=float, default=0.4, help='Dropout rate for CLIP feature linear layer')
    
    # 融合层与注意力机制
    parser.add_argument('--attention_nheads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--attention_dropout', type=float, default=0.4, help='Dropout for attention')
    parser.add_argument('--fusion_dropout', type=float, default=0.5, help='Increased dropout to fight overfitting')
    parser.add_argument('--output_hidden_size', type=int, default=256, help='Hidden size for output')
    
    # 优化器
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay')
    parser.add_argument('--lr', type=float, default=1e-4, help='General learning rate')
    
    # 模式选择
    parser.add_argument('--text_only', action='store_true', default=False)
    parser.add_argument('--image_only', action='store_true', default=False)
    parser.add_argument('--model', type=int, choices=[1, 2, 3], default=3)

    return parser.parse_args()

# if torch.backends.mps.is_available():
#     device = torch.device("mps")
# elif torch.cuda.is_available():
#     device = torch.device("cuda:0")
# else:
#     device = torch.device("cpu")
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

    # config.resnet_type = args.resnet_type
    # config.resnet_dropout = args.resnet_dropout
    # config.resnet_lr = args.resnet_lr
    config.clip_lr = args.clip_lr
    config.attention_nheads = args.attention_nheads
    config.attention_dropout = args.attention_dropout
    config.fusion_dropout = args.fusion_dropout
    config.output_hidden_size = args.output_hidden_size
    config.weight_decay = args.weight_decay
    config.lr = args.lr

    set_seed(config.seed)
    wandb.init(
            project="AILAB5",
            name=f"最终实验对比model-transformer-textonly{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=vars(args),
            reinit=True,
            allow_val_change=True
        )
    train_dataloader, valid_dataloader, test_dataloader = create_dataloader(
        config.train_data_path, 
        config.test_data_path, 
        config.data_path, 
        text_only=args.text_only, 
        image_only=args.image_only
    )
    if args.model == 1:
        model = ConcatFusionModel(config)
    elif args.model == 2:
        model = TransformerFusionModel(config)
    trainer = trainer_validator(train_dataloader, config, model, device)
    val_accuracy = trainer.train(train_dataloader, valid_dataloader, config.epochs, evaluate_every=1)
    wandb.finish()