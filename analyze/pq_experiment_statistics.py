#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import csv
import glob
# from collections import defaultdict
import math

def parse_experiment_params(log_filename):
    """从日志文件名中提取实验参数设置"""
    filename = os.path.basename(log_filename)
    match = re.search(r'sv=(\d+)\+cs=(\d+)\+lr=([^+]+)\+l2=([^\.]+)', filename)
    if match:
        return {
            'subvectors': match.group(1),
            'code_size': match.group(2),
            'learning_rate': match.group(3),
            'l2_reg': match.group(4)
        }
    return None

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
            final_pattern = r'最终NDCG@10评估: \{([^\}]+)\}'
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

def generate_csv_report(output_file='analyze/pq_experiment_statistics.csv'):
    """生成CSV报告，包含所有实验的参数设置和NDCG@10分数"""
    log_files = glob.glob('logs/pq_experiments/*.log')
    
    if not log_files:
        print("未找到日志文件")
        return
    
    # 获取所有出现的语言代码
    all_langs = set()
    for log_file in log_files:
        lang_scores, _ = extract_ndcg_scores(log_file)
        all_langs.update(lang_scores.keys())
    
    # 按字母顺序排序语言代码
    all_langs = sorted(list(all_langs))
    
    # 准备CSV头部
    headers = ['subvectors', 'code_size', 'learning_rate', 'l2_reg'] + all_langs + ['avg_ndcg']
    
    # 收集所有实验数据
    all_experiments = []
    
    for log_file in log_files:
        params = parse_experiment_params(log_file)
        if not params:
            continue
            
        lang_scores, avg_ndcg = extract_ndcg_scores(log_file)
        if not lang_scores:
            continue
            
        if avg_ndcg is None and lang_scores:
            avg_ndcg = sum(lang_scores.values()) / len(lang_scores)
            
        experiment_data = {
            'subvectors': params['subvectors'],
            'code_size': params['code_size'],
            'learning_rate': params['learning_rate'],
            'l2_reg': params['l2_reg'],
            'avg_ndcg': avg_ndcg if avg_ndcg is not None else float('nan')
        }
        
        # 添加各语言的NDCG分数
        for lang in all_langs:
            experiment_data[lang] = lang_scores.get(lang, float('nan'))
        
        all_experiments.append(experiment_data)
    
    # 先按learning_rate数值升序，然后在每组内按l2_reg升序排序
    all_experiments.sort(key=lambda x: (float(x['learning_rate']), float(x['l2_reg'])))
    
    # 写入CSV文件
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for experiment in all_experiments:
            writer.writerow(experiment)
    
    print(f"统计报告已生成: {output_file}")
    print(f"总共分析了 {len(all_experiments)} 个实验")

if __name__ == "__main__":
    generate_csv_report() 