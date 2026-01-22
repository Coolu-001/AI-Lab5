import os
import shutil

def organize_bad_cases(txt_path, image_source_dir, output_root="bad_case_analysis"):
    """
    将误分类的图片提取并按错误类别归档。
    
    Args:
        txt_path: best_bad_cases.txt 的路径
        image_source_dir: 原始数据集图片的存放目录 (data/images/)
        output_root: 整理后图片的存放根目录
    """
    if not os.path.exists(output_root):
        os.makedirs(output_root)
        
    count = 0
    with open(txt_path, 'r') as f:
        # 跳过第一行标题 (Bad Cases from Best Epoch...)
        lines = f.readlines()[1:]
        
        for line in lines:
            try:
                # 解析行格式: GUID: 2575 | True: 0 | Pred: 2
                parts = line.strip().split('|')
                guid = parts[0].split(':')[1].strip()
                true_label = parts[1].split(':')[1].strip()
                pred_label = parts[2].split(':')[1].strip()
                
                # 创建子文件夹格式: True_0_Pred_2 (真实为0, 预测为2)
                sub_folder = os.path.join(output_root, f"True_{true_label}_Pred_{pred_label}")
                if not os.path.exists(sub_folder):
                    os.makedirs(sub_folder)
                
                # 源文件路径 (假设是 .jpg)
                src_image = os.path.join(image_source_dir, f"{guid}.jpg")
                
                # 如果找不到 .jpg，尝试 .png
                if not os.path.exists(src_image):
                    src_image = os.path.join(image_source_dir, f"{guid}.png")

                if os.path.exists(src_image):
                    # 复制文件到对应分类文件夹
                    shutil.copy(src_image, os.path.join(sub_folder, f"{guid}.jpg"))
                    count += 1
                else:
                    print(f"Warning: Image {guid} not found in {image_source_dir}")
                    
            except Exception as e:
                print(f"Error processing line '{line}': {e}")

    print(f"整理完成！共提取 {count} 张 Bad Case 图片至文件夹: {output_root}")

# --- 使用示例 ---
if __name__ == "__main__":
    # 请根据你的实际路径修改以下两个变量
    organize_bad_cases(
        txt_path="best_model3_bad_cases.txt", 
        image_source_dir="dataset/data/", # 你的原始图片存放路径
        output_root="bad_case_visuals"
    )