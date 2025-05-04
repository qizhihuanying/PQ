#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import csv
import glob
from collections import defaultdict
import math
import pandas as pd

def parse_experiment_params(log_filename):
    """从日志文件名中提取实验参数设置"""
    filename = os.path.basename(log_filename)
    match = re.search(r'sv=(\d+)\+cs=(\d+)\+attn_lr=([^+]+)\+heads=(\d+)', filename)
    if match:
        return {
            'subvectors': match.group(1),
            'code_size': match.group(2),
            'attention_lr': match.group(3),
            'attention_heads': match.group(4)
        }
    return None

def extract_ndcg_and_bias(log_file):
    """提取日志文件中各语言的NDCG@10分数、平均NDCG@10和最后一轮bias比例"""
    results = {}
    avg_ndcg = None
    bias_ratio = None
    
    try:
        with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
            
            # 检查日志是否完整
            if "最终NDCG@10评估:" not in content and "所有语言平均NDCG@10:" not in content:
                print(f"跳过未完成的日志文件: {log_file}")
                return None, None, None
            
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
            
            # 提取最后一轮训练结束后的bias比例
            bias_pattern = r'平均原始范数: [\d\.]+, 平均Bias范数: [\d\.]+, 平均比例: ([\d\.]+)'
            bias_matches = re.findall(bias_pattern, content)
            if bias_matches:
                bias_ratio = float(bias_matches[-1])  # 获取最后一个匹配项，即最后一轮训练
            
            # 如果没找到平均值但有各语言的值，则计算平均值
            if avg_ndcg is None and results:
                avg_ndcg = sum(results.values()) / len(results)
                
            # 检查是否有足够的数据
            if len(results) < 3:  # 至少应该有3种语言
                print(f"跳过数据不足的日志文件: {log_file} (只有{len(results)}种语言)")
                return None, None, None
                
    except Exception as e:
        print(f"处理文件 {log_file} 时发生错误: {e}")
        return None, None, None
    
    return results, avg_ndcg, bias_ratio

def generate_csv_report(output_file='analyze/pq_experiment_attn_statistics.csv'):
    """生成CSV报告，包含所有实验的参数设置、NDCG@10分数和bias比例"""
    log_files = glob.glob('logs/pq_experiments_attn/*.log')
    
    if not log_files:
        print("未找到日志文件")
        return
    
    print(f"找到 {len(log_files)} 个日志文件")
    
    # 获取所有出现的语言代码
    all_langs = set()
    valid_files = []
    
    for log_file in log_files:
        lang_scores, _, _ = extract_ndcg_and_bias(log_file)
        if lang_scores is not None:
            all_langs.update(lang_scores.keys())
            valid_files.append(log_file)
    
    print(f"有效日志文件: {len(valid_files)}/{len(log_files)}")
    
    # 按字母顺序排序语言代码
    all_langs = sorted(list(all_langs))
    print(f"检测到的语言: {', '.join(all_langs)}")
    
    # 收集所有实验数据
    all_experiments = []
    
    for log_file in valid_files:
        params = parse_experiment_params(log_file)
        if not params:
            continue
            
        lang_scores, avg_ndcg, bias_ratio = extract_ndcg_and_bias(log_file)
        if not lang_scores:
            continue
            
        if avg_ndcg is None and lang_scores:
            avg_ndcg = sum(lang_scores.values()) / len(lang_scores)
            
        if bias_ratio is None:
            bias_ratio = float('nan')
            
        experiment_data = {
            'subvectors': params['subvectors'],
            'code_size': params['code_size'],
            'attention_lr': params['attention_lr'],
            'attention_heads': int(params['attention_heads']),  # 转为整数便于排序
            'bias_ratio': round(bias_ratio, 4),
            'avg_ndcg': round(avg_ndcg, 4) if avg_ndcg is not None else float('nan')
        }
        
        # 添加各语言的NDCG分数
        for lang in all_langs:
            if lang in lang_scores:
                experiment_data[lang] = round(lang_scores[lang], 4)
            else:
                experiment_data[lang] = float('nan')
        
        all_experiments.append(experiment_data)
    
    # 使用pandas处理数据框并导出为CSV
    if all_experiments:
        # 创建DataFrame
        df = pd.DataFrame(all_experiments)
        
        # 添加数值形式的学习率列用于排序
        df['attn_lr_float'] = df['attention_lr'].apply(lambda x: float(x))
        
        # 先按attention_lr升序排序，再按attention_heads升序排序
        df = df.sort_values(by=['attn_lr_float', 'attention_heads'])
        
        # 删除临时用于排序的列
        df = df.drop(columns=['attn_lr_float'])
        
        # 准备列顺序
        columns = ['subvectors', 'code_size', 'attention_lr', 'attention_heads'] + all_langs + ['avg_ndcg', 'bias_ratio']
        
        # 重排列
        df = df[columns]
        
        # 保存为CSV，不带行索引
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False)
        
        print(f"统计报告已生成: {output_file}")
        print(f"总共分析了 {len(all_experiments)} 个实验")
    else:
        print("没有有效的实验数据可分析")

if __name__ == "__main__":
    generate_csv_report() 