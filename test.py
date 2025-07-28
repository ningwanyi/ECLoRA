import os
import shutil


def delete_param_dir():
    # 指定目标目录
    base_dir = "/home/nwy/code/ECLoRA/results/microsoft/deberta-v2-xxlarge"

    # 遍历 base_dir 下的所有子文件夹
    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)
        
        # 检查是否是文件夹
        if os.path.isdir(subdir_path):
            final_results_path = os.path.join(subdir_path, "final_results.json")
            model_dir_path = os.path.join(subdir_path, "model_whole_param_dict")
            
            # 检查是否同时包含 "final_results.json" 和 "model_whole_param_dict" 目录
            if os.path.isfile(final_results_path) and os.path.isdir(model_dir_path):
                print(f"Deleting: {model_dir_path}")
                shutil.rmtree(model_dir_path)  # 删除整个目录

def copy_dirs():
    # 源目录和目标目录
    source_dir = "/home/nwy/code/ECLoRA/results/roberta-base"
    target_dir = "/home/nwy/code/ECLoRA/cleaned_results/roberta-base"

    # 确保目标目录存在
    os.makedirs(target_dir, exist_ok=True)

    # 遍历源目录中的所有子目录
    for subdir in os.listdir(source_dir):
        subdir_path = os.path.join(source_dir, subdir)
        target_subdir_path = os.path.join(target_dir, subdir)
        
        # 检查是否是目录且名称包含 "flora"
        if os.path.isdir(subdir_path) and "flora_seed_42" in subdir.lower():
            print(f"Copying: {subdir_path} -> {target_subdir_path}")
            if os.path.exists(target_subdir_path):
                shutil.rmtree(target_subdir_path)  # 先删除原目录
            shutil.copytree(subdir_path, target_subdir_path)

if __name__ == "__main__":
    # n = "_module.ok"
    # n = n[8:] if n.startswith("_module.") else n
    # print(n)
    delete_param_dir()
    # copy_dirs()