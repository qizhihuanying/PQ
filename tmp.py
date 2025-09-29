from huggingface_hub import snapshot_download
import os

def download_miracl_dataset():
    # 指定保存路径
    target_dir = r"C:\Users\QZHYc\Downloads\miracl"

    # 如果目录不存在就创建
    os.makedirs(target_dir, exist_ok=True)

    # 下载整个数据集
    snapshot_download(
        repo_id="miracl/miracl",
        repo_type="dataset",
        local_dir=target_dir,
        local_dir_use_symlinks=False  # 防止生成软链接，直接拷贝文件
    )

    print(f"✅ 数据集已下载到: {target_dir}")

if __name__ == "__main__":
    download_miracl_dataset()
