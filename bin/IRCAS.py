    #!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from Bio import SeqIO
import subprocess
import argparse
from pathlib import Path
import shutil
import tempfile
import json

class SpliceCNNAttention(nn.Module):
    def __init__(self, input_channels=6, seq_len=502):
        super(SpliceCNNAttention, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=7, padding=3)
        self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(512)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
        conv_len = seq_len // 2
        conv_len = conv_len // 2
        self.conv_out_dim = 512 * conv_len
        
        self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=8, dropout=0.1)
        self.ln = nn.LayerNorm(512)
        
        self.fc1 = nn.Linear(self.conv_out_dim, 128)
        self.fc2 = nn.Linear(128, 2)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        
        x = x.permute(2, 0, 1)
        x = self.ln(x)
        
        attn_output, _ = self.attention(x, x, x)
        
        x = attn_output.permute(1, 0, 2).contiguous().view(-1, self.conv_out_dim)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def encode_base(base, is_start=False, is_end=False):
    """编码单个碱基或特殊位点为 6 维向量"""
    if is_start:
        return np.array([0, 0, 0, 0, 1, 0], dtype=np.float32)
    if is_end:
        return np.array([0, 0, 0, 0, 0, 1], dtype=np.float32)
    base_to_idx = {
        'A': [1, 0, 0, 0, 0, 0],
        'C': [0, 1, 0, 0, 0, 0],
        'G': [0, 0, 1, 0, 0, 0],
        'T': [0, 0, 0, 1, 0, 0]
    }
    return np.array(base_to_idx.get(base, [0, 0, 0, 0, 0, 0]), dtype=np.float32)

def encode_sequence(newup, newaa, newdown, max_newaa_len=400):
    """合并并编码序列"""
    if not isinstance(newup, str) or not isinstance(newaa, str) or not isinstance(newdown, str):
        raise ValueError(f"Invalid sequence input")
    
    if len(newup) < 50:
        newup = newup.ljust(50, 'A')
    else:
        newup = newup[:50]
    
    if len(newdown) < 50:
        newdown = newdown.ljust(50, 'A')
    else:
        newdown = newdown[:50]
    
    if len(newaa) < max_newaa_len:
        newaa = newaa.ljust(max_newaa_len, 'A')
    else:
        newaa = newaa[:max_newaa_len]
    
    sequence = newup + 'S' + newaa + 'E' + newdown
    encoded = [encode_base(base, is_start=(base == 'S'), is_end=(base == 'E')) for base in sequence]
    return np.array(encoded)

def run_initial_identification(fasta_file, model_type, threads):
    """运行identifier.sh的前三步：makeblastdb, blastn, 和 cDBG construction"""
    print("Step 1: Creating BLAST database...")
    subprocess.run(['makeblastdb', '-in', fasta_file, '-dbtype', 'nucl'], check=True)
    
    print("Step 2: Running BLAST alignment...")
    blast_output = f"{fasta_file}_blastout.txt"
    subprocess.run([
        'blastn', '-query', fasta_file, '-db', fasta_file, '-strand', 'plus',
        '-evalue', '1E-10', '-outfmt', '6', '-ungapped', '-num_threads', str(threads),
        '-out', blast_output
    ], check=True)
    
    print("Step 3: Predicting AS transcript pairs...")
    # 清理之前的split文件
    for split_file in Path('.').glob(f"{fasta_file}*split*"):
        split_file.unlink()
    
    # 运行unique.py
    subprocess.run(['python3', 'unique.py', blast_output, str(threads)], check=True)
    
    # 运行cDBG构建
    split_files = list(Path('.').glob(f"{fasta_file}*split*"))
    processes = []
    for split_file in split_files:
        cmd = ['python3', 'cdbg.py', str(split_file), fasta_file]
        with open(f"{split_file}_four_AS.seq", 'w') as outfile:
            process = subprocess.Popen(cmd, stdout=outfile)
            processes.append(process)
    
    # 等待所有进程完成
    for process in processes:
        process.wait()
    
    # 合并结果文件
    four_as_file = f"{fasta_file}_four_AS.seq"
    with open(four_as_file, 'w') as outfile:
        for split_file in split_files:
            seq_file = f"{split_file}_four_AS.seq"
            if os.path.exists(seq_file):
                with open(seq_file, 'r') as infile:
                    outfile.write(infile.read())
    
    # 清理所有split相关文件
    print("Cleaning up temporary files...")
    cleanup_patterns = [
        f"{fasta_file}*split*",
        f"{fasta_file}*.unique*", 
        f"{fasta_file}*done*"
    ]
    
    for pattern in cleanup_patterns:
        for temp_file in Path('.').glob(pattern):
            try:
                temp_file.unlink()
                print(f"Removed: {temp_file}")
            except Exception as e:
                print(f"Warning: Could not remove {temp_file}: {e}")
    
    return four_as_file

def parse_four_as_sequences(four_as_file, fasta_file):
    """解析four_AS.seq文件，提取序列信息"""
    seqs = {}
    for seq_record in SeqIO.parse(fasta_file, "fasta"):
        seqs[seq_record.id] = str(seq_record.seq)
    
    as_events = []
    if not os.path.exists(four_as_file):
        print(f"Warning: {four_as_file} does not exist, no AS events found.")
        return as_events
    
    print(f"Parsing {four_as_file}...")
    line_count = 0
    valid_count = 0
    
    with open(four_as_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            line_count += 1
            parts = line.split(',')
            
            # 检查格式: transcript1,transcript2,upstream,as_region,downstream,pos1,pos2,pos3,pos4,len1,len2
            if len(parts) != 11:
                if line_count <= 5:
                    print(f"Line {line_count}: Expected 11 parts, got {len(parts)}")
                continue
                
            try:
                seq1_name = parts[0]
                seq2_name = parts[1] 
                upstream_seq = parts[2]
                as_region = parts[3]
                downstream_seq = parts[4]
                start_pos1 = int(parts[5])
                end_pos1 = int(parts[6])
                start_pos2 = int(parts[7])
                end_pos2 = int(parts[8])
                len1 = int(parts[9])
                len2 = int(parts[10])
                
                # 检查转录本是否存在
                if seq1_name not in seqs or seq2_name not in seqs:
                    continue
                
                # 验证位置的合理性
                seq1_len = len(seqs[seq1_name])
                seq2_len = len(seqs[seq2_name])
                
                if (start_pos1 < 0 or end_pos1 > seq1_len or start_pos2 < 0 or end_pos2 > seq2_len):
                    continue
                
                # 使用提供的序列，但确保长度正确
                if len(upstream_seq) < 50:
                    if start_pos1 >= 50:
                        upstream_seq = seqs[seq1_name][start_pos1-50:start_pos1]
                    else:
                        upstream_seq = seqs[seq1_name][:start_pos1].ljust(50, 'N')
                elif len(upstream_seq) > 50:
                    upstream_seq = upstream_seq[-50:]
                
                if len(downstream_seq) < 50:
                    if end_pos1 + 50 <= seq1_len:
                        downstream_seq = seqs[seq1_name][end_pos1:end_pos1+50]
                    else:
                        downstream_seq = seqs[seq1_name][end_pos1:].ljust(50, 'N')
                elif len(downstream_seq) > 50:
                    downstream_seq = downstream_seq[:50]
                
                if len(as_region) == 0:
                    as_region = "N"
                
                transcript_pair = f"{seq1_name},{seq2_name}"
                
                as_events.append({
                    'transcript_pair': transcript_pair,
                    'upstream_seq': upstream_seq,
                    'as_region': as_region,
                    'downstream_seq': downstream_seq,
                    'start_pos1': start_pos1,
                    'end_pos1': end_pos1,
                    'start_pos2': start_pos2,
                    'end_pos2': end_pos2,
                    'len1': len1,
                    'len2': len2
                })
                valid_count += 1
                
                if valid_count <= 3:
                    print(f"Successfully parsed event {valid_count}: {transcript_pair}")
                
            except Exception as e:
                if line_count <= 10:
                    print(f"Error parsing line {line_count}: {e}")
                continue
    
    print(f"Successfully parsed {valid_count} AS events from {line_count} total lines.")
    return as_events

def predict_offsets(as_events, model, scaler, device):
    """使用rectification模型预测位置偏移"""
    if not as_events:
        print("No AS events to process.")
        return []
    
    print(f"Predicting offsets for {len(as_events)} AS events...")
    
    X = []
    valid_events = []
    
    for event in as_events:
        try:
            encoded_seq = encode_sequence(
                event['upstream_seq'], 
                event['as_region'], 
                event['downstream_seq']
            )
            X.append(encoded_seq)
            valid_events.append(event)
        except Exception as e:
            print(f"Error encoding sequence for {event['transcript_pair']}: {e}")
            continue
    
    if not X:
        print("No valid sequences for prediction.")
        return []
    
    X = np.array(X)
    
    # 预测
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    
    predictions = []
    with torch.no_grad():
        pred_y = model(X_tensor).cpu().numpy()
        pred_y = scaler.inverse_transform(pred_y)
        predictions = pred_y
    
    # 添加预测结果到事件中
    for i, event in enumerate(valid_events):
        event['pred_start_offset'] = predictions[i][0]
        event['pred_end_offset'] = predictions[i][1]
    
    return valid_events

def reconstruct_as_positions(as_events_with_offsets, fasta_file):
    """根据预测的偏移重新构建AS位置和区域"""
    seqs = {}
    for seq_record in SeqIO.parse(fasta_file, "fasta"):
        seqs[seq_record.id] = str(seq_record.seq)
    
    reconstructed_events = []
    
    for event in as_events_with_offsets:
        transcript_names = event['transcript_pair'].split(',')
        if len(transcript_names) != 2:
            continue
        
        seq1_name, seq2_name = transcript_names
        if seq1_name not in seqs or seq2_name not in seqs:
            continue
        
        # 应用偏移校正
        corrected_start1 = int(event['start_pos1'] + event['pred_start_offset'])
        corrected_end1 = int(event['end_pos1'] + event['pred_end_offset'])
        corrected_start2 = int(event['start_pos2'] + event['pred_start_offset'])
        corrected_end2 = int(event['end_pos2'] + event['pred_end_offset'])
        
        # 确保位置在序列范围内
        corrected_start1 = max(0, min(corrected_start1, len(seqs[seq1_name])))
        corrected_end1 = max(0, min(corrected_end1, len(seqs[seq1_name])))
        corrected_start2 = max(0, min(corrected_start2, len(seqs[seq2_name])))
        corrected_end2 = max(0, min(corrected_end2, len(seqs[seq2_name])))
        
        # 重新提取序列区域
        new_upstream = seqs[seq1_name][max(0, corrected_start1-50):corrected_start1]
        new_as_region = seqs[seq1_name][corrected_start1:corrected_end1] if corrected_end1 > corrected_start1 else seqs[seq2_name][corrected_start2:corrected_end2]
        new_downstream = seqs[seq1_name][corrected_end1:corrected_end1+50]
        
        # 确保序列长度
        if len(new_upstream) < 50:
            new_upstream = new_upstream.ljust(50, 'N')
        if len(new_downstream) < 50:
            new_downstream = new_downstream.ljust(50, 'N')
        
        reconstructed_event = {
            'transcript_pair': event['transcript_pair'],
            'original_start1': event['start_pos1'],
            'original_end1': event['end_pos1'],
            'original_start2': event['start_pos2'],
            'original_end2': event['end_pos2'],
            'corrected_start1': corrected_start1,
            'corrected_end1': corrected_end1,
            'corrected_start2': corrected_start2,
            'corrected_end2': corrected_end2,
            'pred_start_offset': event['pred_start_offset'],
            'pred_end_offset': event['pred_end_offset'],
            'new_upstream': new_upstream,
            'new_as_region': new_as_region,
            'new_downstream': new_downstream
        }
        
        reconstructed_events.append(reconstructed_event)
    
    return reconstructed_events

def extract_features(reconstructed_events, model_type):
    """提取特征"""
    print("Extracting features...")
    
    # 创建临时序列文件
    temp_seq_file = f"temp_sequences_{model_type}.csv"
    
    with open(temp_seq_file, 'w') as f:
        for i, event in enumerate(reconstructed_events):
            # 使用转录本对名称作为标识符，用分号连接
            transcript_names = event['transcript_pair'].replace(',', ';')
            f.write(f"{transcript_names},{event['new_upstream']},{event['new_as_region']},{event['new_downstream']}\n")
    
    print(f"Created temporary sequence file: {temp_seq_file}")
    
    # 运行特征提取脚本
    if model_type == "animal":
        feature_script = "featurehuman.py"
    else:  # plant
        feature_script = "featureplant.py"
    
    # 修改特征提取脚本的输入输出文件名
    temp_feature_script = f"temp_{feature_script}"
    
    # 读取原始特征脚本并修改输入输出文件名
    with open(feature_script, 'r') as f:
        script_content = f.read()
    
    # 替换输入输出文件名
    if model_type == "animal":
        script_content = script_content.replace(
            'input_file = "mousesequence1.csv"',
            f'input_file = "{temp_seq_file}"'
        )
        script_content = script_content.replace(
            'output_file = "mousesequencefeature.txt"',
            f'output_file = "temp_features_{model_type}.txt"'
        )
    else:  # plant
        script_content = script_content.replace(
            'input_file = "ricesequence1.csv"',
            f'input_file = "{temp_seq_file}"'
        )
        script_content = script_content.replace(
            'output_file = "ricesequencefeature.txt"',
            f'output_file = "temp_features_{model_type}.txt"'
        )
    
    # 写入临时脚本文件
    with open(temp_feature_script, 'w') as f:
        f.write(script_content)
    
    # 运行特征提取
    subprocess.run(['python3', temp_feature_script], check=True)
    
    feature_file = f"temp_features_{model_type}.txt"
    print(f"Features extracted to: {feature_file}")
    
    # 清理临时文件
    os.remove(temp_seq_file)
    os.remove(temp_feature_script)
    
    return feature_file

def process_features_to_dataset(feature_file, model_type):
    """将特征文件转换为PyG数据集"""
    print("Converting features to PyG dataset...")
    
    # 创建临时数据处理脚本
    temp_processing_script = f"temp_dataprocessing_{model_type}.py"
    
    # 读取原始数据处理脚本
    with open("dataprocessing_lite.py", 'r') as f:
        script_content = f.read()
    
    # 添加MKL线程层修复
    mkl_fix = '''import os
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MKL_THREADING_LAYER"] = "GNU"
import numpy as np
'''
    
    # 在import语句之前添加MKL修复
    if "import torch" in script_content:
        script_content = mkl_fix + script_content
    
    # 修改输入输出文件名
    script_content = script_content.replace(
        'df = pd.read_csv("mousesequencefeature.txt", header=None, engine=\'c\')',
        f'df = pd.read_csv("{feature_file}", header=None, engine=\'c\')'
    )
    script_content = script_content.replace(
        'torch.save(dataset, \'mousedataset.pt\')',
        f'torch.save(dataset, \'temp_dataset_{model_type}.pt\')'
    )
    
    # 写入临时脚本
    with open(temp_processing_script, 'w') as f:
        f.write(script_content)
    
    # 运行数据处理
    subprocess.run(['python3', temp_processing_script], check=True)
    
    dataset_file = f"temp_dataset_{model_type}.pt"
    print(f"Dataset created: {dataset_file}")
    
    # 清理临时文件
    os.remove(temp_processing_script)
    os.remove(feature_file)
    
    return dataset_file

def classify_as_events(dataset_file, model_type):
    """分类AS事件"""
    print("Classifying AS events...")
    
    # 创建临时分类脚本
    temp_classify_script = f"temp_classify_{model_type}.py"
    
    # 读取分类脚本
    with open("classifyAS.py", 'r') as f:
        script_content = f.read()
    
    # 添加MKL线程层修复到脚本开头
    mkl_fix = '''import os
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MKL_THREADING_LAYER"] = "GNU"
import numpy as np
'''
    
    # 在第一个import语句之前添加MKL修复
    if "import torch" in script_content or "import numpy" in script_content:
        lines = script_content.split('\n')
        new_lines = []
        mkl_added = False
        
        for line in lines:
            if not mkl_added and ('import ' in line and not line.strip().startswith('#')):
                new_lines.extend(mkl_fix.strip().split('\n'))
                new_lines.append('')
                mkl_added = True
            new_lines.append(line)
        
        script_content = '\n'.join(new_lines)
    
    # 修改配置
    if model_type == "animal":
        model_path = 'best_modela.pth'  # animal模型
    else:  # plant
        model_path = 'best_modelp.pth'  # plant模型
    
    script_content = script_content.replace(
        "model_path='best_model2.pth'",
        f"model_path='{model_path}'"
    )
    script_content = script_content.replace(
        "dataset='mousedataset.pt'",
        f"dataset='{dataset_file}'"
    )
    
    # 添加函数来提取转录本名称并保存结果
    additional_functions = '''
def extract_transcript_pairs_and_predict():
    """提取转录本对名称并进行预测"""
    predictor = ModelPredictor(model_path, device=device)
    
    # 加载数据集并提取转录本对名称
    dataset_obj = torch.load(dataset, map_location='cpu')
    transcript_pairs = []
    
    for data in dataset_obj:
        if hasattr(data, 'transcript_pair'):
            transcript_pairs.append(data.transcript_pair)
        else:
            transcript_pairs.append("Unknown")
    
    # 进行预测
    results = predictor.predict(dataset, return_probs=True)
    
    # 整合结果
    final_results = {
        'transcript_pairs': transcript_pairs,
        'predictions': results['predictions'].tolist(),
        'labels': results['labels'],
        'statistics': {
            'total_samples': len(results['predictions']),
            'class_distribution': {}
        }
    }
    
    # 计算类别分布
    unique, counts = np.unique(results['predictions'], return_counts=True)
    for cls, count in zip(unique, counts):
        final_results['statistics']['class_distribution'][str(int(cls))] = int(count)
    
    if 'confidence' in results:
        final_results['confidence'] = results['confidence'].tolist()
        final_results['mean_confidence'] = float(np.mean(results['confidence']))
    
    # 保存结果
    with open('predictions.json', 'w') as f:
        json.dump(final_results, f, indent=4)
    
    # 创建详细的CSV结果
    import pandas as pd
    df = pd.DataFrame({
        'transcript_pair': transcript_pairs,
        'predicted_class': results['predictions'],
        'predicted_label': results['labels']
    })
    
    if 'confidence' in results:
        df['confidence'] = results['confidence']
    
    df.to_csv('predictions_detailed.csv', index=False)
    
    print("Predictions with transcript pairs saved to predictions.json and predictions_detailed.csv")
    return final_results

# 替换simple_prediction函数
def simple_prediction():
    """使用转录本对名称的预测"""
    return extract_transcript_pairs_and_predict()
'''
    
    # 添加额外函数到脚本中
    script_content = script_content.replace(
        'if __name__ == "__main__":',
        additional_functions + '\nif __name__ == "__main__":'
    )
    
    # 写入临时脚本
    with open(temp_classify_script, 'w') as f:
        f.write(script_content)
    
    # 运行分类
    subprocess.run(['python3', temp_classify_script, 'simple'], check=True)
    
    print("Classification completed. Results saved to predictions.json")
    
    # 清理临时文件
    os.remove(temp_classify_script)
    os.remove(dataset_file)
    
    return "predictions.json"

def test_reconstruction_to_features(reconstructed_events, model_type, sample_size=5):
    """测试模式：验证reconstruction到feature提取这一步"""
    print(f"\n{'='*60}")
    print("TEST MODE: Reconstruction to Feature Extraction Validation")
    print(f"{'='*60}")
    
    # 只取前几个样本进行测试
    test_events = reconstructed_events[:sample_size]
    print(f"Testing with {len(test_events)} samples...")
    
    # 1. 显示重构的序列信息
    print("\n1. Reconstructed Sequences:")
    print("-" * 40)
    for i, event in enumerate(test_events):
        print(f"Sample {i+1}: {event['transcript_pair']}")
        print(f"  Upstream  (50bp): {event['new_upstream'][:50]}...")
        print(f"  AS region ({len(event['new_as_region'])}bp): {event['new_as_region'][:50]}...")
        print(f"  Downstream(50bp): {event['new_downstream'][:50]}...")
        print(f"  Offset: start={event['pred_start_offset']:.2f}, end={event['pred_end_offset']:.2f}")
        print()
    
    # 2. 生成特征
    print("2. Extracting features for test samples...")
    feature_file = extract_features(test_events, model_type)
    
    # 3. 检查特征文件
    print("3. Feature file validation:")
    print("-" * 40)
    
    with open(feature_file, 'r') as f:
        lines = f.readlines()
    
    print(f"Generated {len(lines)} feature lines")
    
    # 显示第一个样本的特征
    if lines:
        first_line = lines[0].strip()
        parts = first_line.split(',')
        print(f"First sample features: {len(parts)-4} features")  # 减去前4个字段（label,up,as,down）
        print(f"Sample format: {parts[0]},{parts[1][:20]}...,{parts[2][:20]}...,{parts[3][:20]}...,<{len(parts)-4} features>")
    
    # 4. 验证特征数量是否正确
    expected_features = 376 if model_type == "animal" else 297
    if lines:
        actual_features = len(lines[0].strip().split(',')) - 4
        print(f"Expected features: {expected_features}, Actual: {actual_features}")
        if actual_features == expected_features:
            print("✓ Feature count is correct!")
        else:
            print("✗ Feature count mismatch!")
    
    print(f"\nTest completed. Feature file saved as: {feature_file}")
    print("You can manually inspect this file to verify the feature extraction.")
    
    return feature_file

def main():
    parser = argparse.ArgumentParser(description="IRCAS: Integrated Reconstruction, Classification, and Analysis System")
    parser.add_argument('fasta_file', help="Input FASTA file with full-length transcripts")
    parser.add_argument('model_type', choices=['animal', 'plant'], 
                        help="Model type: 'animal' for animal species, 'plant' for plant species")
    parser.add_argument('--threads', type=int, default=20, help="Number of threads for BLAST and cDBG construction")
    parser.add_argument('--output', help="Output file for reconstructed AS events")
    parser.add_argument('--mode', choices=['full', 'test'], default='full',
                        help="Run mode: 'full' for complete pipeline, 'test' for reconstruction validation only")
    
    args = parser.parse_args()
    
    # 根据模型类型设置相应的文件路径
    if args.model_type == "plant":
        model_path = "Ararectification.pt"
        scaler_path = "scalerp.npy"
        classification_model = "best_modelp.pth"
    else:  # animal
        model_path = "humanrectification.pt"
        scaler_path = "scaler.npy"
        classification_model = "best_modela.pth"
    
    if not args.output:
        args.output = f"{args.fasta_file}_reconstructed_AS.tsv"
    
    # 检查文件是否存在
    if not os.path.exists(args.fasta_file):
        print(f"Error: FASTA file {args.fasta_file} does not exist!")
        return
    
    if not os.path.exists(model_path):
        print(f"Error: Rectification model file {model_path} does not exist!")
        return
    
    if not os.path.exists(scaler_path):
        print(f"Error: Scaler file {scaler_path} does not exist!")
        return
    
    if args.mode == 'full' and not os.path.exists(classification_model):
        print(f"Error: Classification model file {classification_model} does not exist!")
        return
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Model configuration: {args.model_type}")
    print(f"  - Rectification model: {model_path}")
    print(f"  - Scaler: {scaler_path}")
    if args.mode == 'full':
        print(f"  - Classification model: {classification_model}")
    
    # 加载模型和scaler
    print("Loading rectification model and scaler...")
    model = SpliceCNNAttention(input_channels=6, seq_len=502)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    
    scaler = np.load(scaler_path, allow_pickle=True).item()
    
    # 步骤1-3：初始AS识别
    print("Running initial AS identification...")
    four_as_file = run_initial_identification(args.fasta_file, args.model_type, args.threads)
    
    # 解析AS事件
    print("Parsing AS events...")
    as_events = parse_four_as_sequences(four_as_file, args.fasta_file)
    print(f"Found {len(as_events)} AS events.")
    
    if len(as_events) == 0:
        print("No AS events found. Pipeline terminated.")
        return
    
    # 预测偏移
    print("Predicting position offsets...")
    as_events_with_offsets = predict_offsets(as_events, model, scaler, device)
    
    # 重构AS位置
    print("Reconstructing AS positions...")
    reconstructed_events = reconstruct_as_positions(as_events_with_offsets, args.fasta_file)
    
    # 保存重构结果
    df = pd.DataFrame(reconstructed_events)
    df.to_csv(args.output, index=False, sep='\t')
    print(f"Reconstructed AS events saved to {args.output}")
    
    if args.mode == 'test':
        # 测试模式：只验证到特征提取
        test_reconstruction_to_features(reconstructed_events, args.model_type)
        print("Test mode completed. Check the generated feature file for validation.")
        return
    
    # 完整模式：继续执行特征提取、数据处理和分类
    print("\n" + "="*60)
    print("FULL PIPELINE MODE")
    print("="*60)
    
    # 步骤4：特征提取
    feature_file = extract_features(reconstructed_events, args.model_type)
    
    # 步骤5：数据处理（转换为PyG数据集）
    dataset_file = process_features_to_dataset(feature_file, args.model_type)
    
    # 步骤6：AS事件分类
    prediction_file = classify_as_events(dataset_file, args.model_type)
    
    # 步骤7：整合结果
    print("\nIntegrating results...")
    
    # 读取预测结果
    import json
    with open(prediction_file, 'r') as f:
        predictions = json.load(f)
    
    # 将预测结果添加到重构事件中，按转录本对匹配
    final_results = []
    transcript_to_prediction = {}
    
    # 创建转录本对到预测结果的映射
    if 'transcript_pairs' in predictions:
        for i, transcript_pair in enumerate(predictions['transcript_pairs']):
            if i < len(predictions['predictions']):
                transcript_to_prediction[transcript_pair] = {
                    'predicted_class': predictions['predictions'][i],
                    'predicted_label': predictions['labels'][i],
                    'confidence': predictions['confidence'][i] if 'confidence' in predictions else None
                }
    
    # 匹配重构事件和预测结果
    for event in reconstructed_events:
        # 将转录本对格式统一（逗号 -> 分号）
        transcript_pair_key = event['transcript_pair'].replace(',', ';')
        
        if transcript_pair_key in transcript_to_prediction:
            pred = transcript_to_prediction[transcript_pair_key]
            event['predicted_class'] = pred['predicted_class']
            event['predicted_label'] = pred['predicted_label']
            if pred['confidence'] is not None:
                event['confidence'] = pred['confidence']
        else:
            event['predicted_class'] = -1
            event['predicted_label'] = "Unknown"
            event['confidence'] = 0.0
        
        final_results.append(event)
    
    # 保存最终结果
    final_output = args.output.replace('.tsv', '_final_results.tsv')
    final_df = pd.DataFrame(final_results)
    final_df.to_csv(final_output, index=False, sep='\t')
    
    # 生成统计报告
    print("\n" + "="*60)
    print("PIPELINE COMPLETION SUMMARY")
    print("="*60)
    print(f"Total AS events processed: {len(final_results)}")
    
    if 'predictions' in locals():
        pred_counts = {}
        for label in predictions['labels']:
            pred_counts[label] = pred_counts.get(label, 0) + 1
        
        print("\nAS Event Classification Results:")
        for label, count in pred_counts.items():
            percentage = count / len(predictions['labels']) * 100
            print(f"  {label}: {count} events ({percentage:.2f}%)")
    
    print(f"\nOutput files:")
    print(f"  - Reconstructed events: {args.output}")
    print(f"  - Final results: {final_output}")
    print(f"  - Predictions: {prediction_file}")
    
    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main()