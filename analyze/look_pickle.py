import pickle
import os
import glob

# 创建输出文件
output_file = "analyze/output.txt"
with open(output_file, "w", encoding="utf-8") as f_out:
    f_out.write("MIRACL数据集所有语言的Pickle文件分析\n")
    f_out.write("=" * 50 + "\n\n")
    
    # 获取所有语言目录
    language_dirs = glob.glob("datasets/processed/*/")
    
    # 遍历每个语言目录
    for lang_dir in language_dirs:
        lang_code = os.path.basename(os.path.dirname(lang_dir))
        pickle_path = os.path.join(lang_dir, "train/processed_data.pkl")
        
        # 检查pickle文件是否存在
        if os.path.exists(pickle_path):
            try:
                with open(pickle_path, "rb") as f:
                    data = pickle.load(f)
                f_out.write(f"语言: {lang_code}\n")
                f_out.write(f"数据示例: {data[:2] if isinstance(data, list) else data}\n")
                f_out.write("-" * 50 + "\n\n")
            except Exception as e:
                f_out.write(f"处理{lang_code}时出错: {e}\n\n")
        else:
            f_out.write(f"未找到{lang_code}的pickle文件: {pickle_path}\n\n")

print(f"分析结果已保存到 {os.path.abspath(output_file)}")