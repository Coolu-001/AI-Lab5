class config:
    data_path = "dataset/data"
    train_guid_label_path = "dataset/train.txt"
    test_guid_label_path = "dataset/test_without_label.txt"
    train_data_path = "dataset/train1.json"
    test_data_path = "dataset/test1.json"
    roberta_path = "FacebookAI/roberta-base"
    clip_path = "openai/clip-vit-base-patch32"
    middle_hidden_size = 768
    image_size = 224
    seed = 42
    num_workers = 2
    epochs = 50
    max_seq_length = 50
    fixed_text_param = False
    fixed_image_param = False
    num_labels = 3

    batch_size = 16
    roberta_dropout = 0.15
    roberta_lr = 1e-5

    #middle_hidden_size = 256
    resnet_type = 18
    resnet_dropout = 0.15
    resnet_lr = 1e-5

    clip_lr = 5e-7
    clip_dropout = 0.3

    attention_nheads = 8
    attention_dropout = 0.15
    fusion_dropout = 0.15
    output_hidden_size = 256
    weight_decay = 1e-3
    lr = 1e-5
    
