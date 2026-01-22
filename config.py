"""
config.py 用于存放所有的配置参数
"""

class config:
    data_path = "dataset/data"
    train_guid_label_path = "dataset/train.txt"
    test_guid_label_path = "dataset/test_without_label.txt"
    train_data_path = "dataset/train1.json"
    test_data_path = "dataset/test1.json"
    roberta_path = "roberta-base"
    image_size = 224
    seed = 42
    num_workers = 4
    epochs = 50
    max_seq_length = 50
    fixed_text_param = False
    fixed_image_param = False
    num_labels = 3

    batch_size = 16
    roberta_dropout = 0.5
    roberta_lr = 1e-6
    middle_hidden_size = 256
    resnet_type = 18
    resnet_dropout = 0.5
    resnet_lr = 1e-6
    attention_nheads = 8
    attention_dropout = 0.5
    fusion_dropout = 0.5
    output_hidden_size = 128
    weight_decay = 1e-3
    lr = 1e-6
    
