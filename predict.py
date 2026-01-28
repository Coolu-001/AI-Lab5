import os
import torch
from tqdm import tqdm

from config import config
from FusionModels import DynamicGatedFusionModel
from load_dataset import create_dataloader
from process_data import Index2Label

# ======================
# 1. 设备设置
# ======================
device = torch.device("cpu") 

# ======================
# 2. 加载最优模型
# ======================
def load_model(model_path):
    model = DynamicGatedFusionModel(config)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

# ======================
# 3. 测试集预测函数
# ======================
@torch.no_grad()
def predict_test(model, test_dataloader, output_txt):
    results = []

    for batch in tqdm(test_dataloader, desc="Predicting Test Set"):
        guids, input_ids, attention_mask, images, labels = batch
        # labels 在 test 阶段不使用，但 dataloader 结构保持一致

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        images = images.to(device)

        # forward（无 labels）
        pred_labels, _ = model(input_ids, attention_mask, images)

        pred_labels = pred_labels.cpu().tolist()

        for guid, pred_idx in zip(guids, pred_labels):
            # 理论上 pred_idx ∈ {0,1,2}
            label_str = Index2Label(pred_idx)

            # 兜底：如果出现 null，强制改为 neutral（或你需要的规则）
            if label_str == "null":
                label_str = "neutral"

            results.append((guid, label_str))

    # ======================
    # 4. 写入 txt 文件
    # ======================
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("guid,tag\n")
        for guid, tag in results:
            f.write(f"{guid},{tag}\n")

    print(f"Prediction file saved to: {output_txt}")


if __name__ == "__main__":

    BEST_MODEL_PATH = os.path.join(
        config.output_dir,
        "best_model_epoch6.pt"   # ← 替换为真实 epoch
    )

    OUTPUT_TXT = "test_predictions1.txt"

    # 只需要 test_dataloader
    _, _, test_dataloader = create_dataloader(
        config.train_data_path,
        config.test_data_path,
        config.data_path,
        text_only=False,
        image_only=False
    )

    model = load_model(BEST_MODEL_PATH)
    predict_test(model, test_dataloader, OUTPUT_TXT)