#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import csv
import glob
import math

def extract_ndcg_scores(log_file):
    """提取日志文件中各语言和平均的NDCG@10分数"""
    results = {}
    avg_ndcg = None
    
    try:
        with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
            
            # 提取各语言的NDCG@10
            lang_pattern = r'(\w{2})语言使用[\w\-]+的平均NDCG@10: ([\d\.]+)'
            lang_matches = re.finditer(lang_pattern, content)
            
            for match in lang_matches:
                lang = match.group(1)
                ndcg = float(match.group(2))
                results[lang] = ndcg
            
            # 提取所有语言的平均NDCG@10
            avg_pattern = r'所有语言平均NDCG@10: ([\d\.]+)'
            avg_match = re.search(avg_pattern, content)
            if avg_match:
                avg_ndcg = float(avg_match.group(1))
                
            # 尝试从最终评估中提取数据
            final_pattern = r'测试集NDCG@10评估: \{([^\}]+)\}'
            final_match = re.search(final_pattern, content)
            if final_match:
                final_data = final_match.group(1)
                lang_pairs = re.findall(r"'(\w{2})': \{'[\w\-]+': ([\d\.]+)\}", final_data)
                for lang, ndcg in lang_pairs:
                    results[lang] = float(ndcg)
            
            # 如果没找到平均值但有各语言的值，则计算平均值
            if avg_ndcg is None and results:
                avg_ndcg = sum(results.values()) / len(results)
                
    except Exception as e:
        print(f"处理文件 {log_file} 时发生错误: {e}")
    
    return results, avg_ndcg

def generate_csv_report(log_dir='logs/pq_kmeans_baseline', output_file='analyze/pq_kmeans_baseline_statistics.csv'):
    """生成CSV报告，包含pq_kmeans_baseline的NDCG@10分数"""
    log_files = glob.glob(f'{log_dir}/*.log')
    
    if not log_files:
        print(f"在 {log_dir} 中未找到日志文件")
        return
    
    print(f"找到 {len(log_files)} 个日志文件")
    
    # 获取所有出现的语言代码
    all_langs = set()
    valid_logs = []
    
    for log_file in log_files:
        lang_scores, avg_ndcg = extract_ndcg_scores(log_file)
        if lang_scores:
            all_langs.update(lang_scores.keys())
            valid_logs.append((log_file, lang_scores, avg_ndcg))
    
    if not valid_logs:
        print("未找到有效的日志数据")
        return
    
    print(f"找到 {len(valid_logs)} 个有效日志文件")
    
    # 按字母顺序排序语言代码
    all_langs = sorted(list(all_langs))
    
    # 准备CSV头部
    headers = ['log_file'] + all_langs + ['avg_ndcg']
    
    # 收集所有数据
    all_results = []
    
    for log_file, lang_scores, avg_ndcg in valid_logs:
        log_name = os.path.basename(log_file)
        
        if avg_ndcg is None and lang_scores:
            avg_ndcg = sum(lang_scores.values()) / len(lang_scores)
            
        result_data = {
            'log_file': log_name,
            'avg_ndcg': avg_ndcg if avg_ndcg is not None else float('nan')
        }
        
        # 添加各语言的NDCG分数
        for lang in all_langs:
            result_data[lang] = lang_scores.get(lang, float('nan'))
        
        all_results.append(result_data)
    
    # 写入CSV文件
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for result in all_results:
            writer.writerow(result)
    
    print(f"统计报告已生成: {output_file}")

if __name__ == "__main__":
    generate_csv_report() 