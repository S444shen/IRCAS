#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import sys
import numpy as np
from collections import Counter

def read_relativesuppa(file_path):
    """读取 relativesuppa.tsv，不过滤任何长度的序列"""
    try:
        df = pd.read_csv(file_path, sep='\t', header=None, encoding='utf-8-sig')
        print(f"Read {len(df)} rows from {file_path}, columns: {df.shape[1]}")
        
        if df.shape[1] != 13:
            print(f"Error: Expected 13 columns in {file_path}, found {df.shape[1]}")
            sys.exit(1)

        relative_positions = {}
        row_count = 0
        valid_pairs = 0

        for _, row in df.iterrows():
            row_count += 1
            if len(row) < 13:
                print(f"Row {row_count}: Skipping invalid row with {len(row)} columns: {row.tolist()}")
                continue
            t1, t2 = str(row[2]), str(row[3])  # QueryName, SubjectName
            try:
                newaa_start_raw = int(row[4])    # newaa_start
                newaa_end_raw = int(row[5])      # newaa_end
                relative_start_raw = int(row[6])  # relative_start
                relative_end_raw = int(row[7])    # relative_end
                
                # 确保start < end，如果不是则交换
                newaa_start = min(newaa_start_raw, newaa_end_raw)
                newaa_end = max(newaa_start_raw, newaa_end_raw)
                relative_start = min(relative_start_raw, relative_end_raw)
                relative_end = max(relative_start_raw, relative_end_raw)
                
                key1 = (t1, t2)
                key2 = (t2, t1)
                if key1 in relative_positions or key2 in relative_positions:
                    print(f"Row {row_count}: Duplicate transpair found: {t1}, {t2}")
                    continue
                    
                # 存储所有数据，不管长度如何
                relative_positions[key1] = (newaa_start, newaa_end, relative_start, relative_end)
                relative_positions[key2] = (relative_start, relative_end, newaa_start, newaa_end)
               
                valid_pairs += 2
            except ValueError as e:
                print(f"Row {row_count}: Invalid data in row: {row.tolist()}, error: {e}")
                continue
                
        print(f"Loaded {valid_pairs} pairs from {file_path}")
        return relative_positions
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        sys.exit(1)

def read_bubble(file_path):
    """读取 cdbgresult1.tsv（无表头，逗号分隔，11 列），返回 DataFrame"""
    try:
        df = pd.read_csv(file_path, sep=',', header=None, encoding='utf-8-sig')
        print(f"Read {len(df)} rows from {file_path}, columns: {df.shape[1]}")
        
        if df.shape[1] != 11:
            print(f"Error: Expected 11 columns in {file_path}, found {df.shape[1]}")
            sys.exit(1)
        return df
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        sys.exit(1)
        
def calculate_statistics(valid_rows, output_file_base):
    """计算start_diff和end_diff的统计量并输出到.out文件"""
    if not valid_rows:
        print("No valid rows for statistics calculation")
        return
    
    # 提取差值数据
    start_diffs = []
    end_diffs = []
    abs_start_diffs = []
    abs_end_diffs = []
    
    for row in valid_rows:
        try:
            start_diff = float(row[9])  # start_diff列
            end_diff = float(row[10])   # end_diff列
            start_diffs.append(start_diff)
            end_diffs.append(end_diff)
            abs_start_diffs.append(abs(start_diff))
            abs_end_diffs.append(abs(end_diff))
        except (ValueError, IndexError):
            continue
    
    if not start_diffs:
        print("No valid diff values for statistics calculation")
        return
    
    # 计算统计值
    mean_start_diff = np.mean(start_diffs)
    mean_end_diff = np.mean(end_diffs)
    mean_abs_start_diff = np.mean(abs_start_diffs)
    mean_abs_end_diff = np.mean(abs_end_diffs)
    
    # 计算众数
    mode_start_diff = Counter(start_diffs).most_common(1)[0][0] if start_diffs else float('nan')
    mode_end_diff = Counter(end_diffs).most_common(1)[0][0] if end_diffs else float('nan')
    
    # 计算四分位数
    q1_start_diff, q2_start_diff, q3_start_diff = np.percentile(start_diffs, [25, 50, 75])
    q1_end_diff, q2_end_diff, q3_end_diff = np.percentile(end_diffs, [25, 50, 75])
    
    # 计算标准差
    std_start_diff = np.std(start_diffs)
    std_end_diff = np.std(end_diffs)
    
    # 计算最大值和最小值
    min_start_diff, max_start_diff = np.min(start_diffs), np.max(start_diffs)
    min_end_diff, max_end_diff = np.min(end_diffs), np.max(end_diffs)
    
    # 输出统计结果到.out文件
    stats_output_file = f"{output_file_base}_statistics.out"
    with open(stats_output_file, 'w') as f_out:
        f_out.write("=== Splice Position Difference Statistics ===\n\n")
        
        f_out.write(f"Total valid rows analyzed: {len(start_diffs)}\n\n")
        
        f_out.write("START_DIFF Statistics:\n")
        f_out.write(f"  Mean: {mean_start_diff:.4f}\n")
        f_out.write(f"  Standard Deviation: {std_start_diff:.4f}\n")
        f_out.write(f"  Mode: {mode_start_diff:.4f}\n")
        f_out.write(f"  Minimum: {min_start_diff:.4f}\n")
        f_out.write(f"  Q1 (25th percentile): {q1_start_diff:.4f}\n")
        f_out.write(f"  Q2 (50th percentile/Median): {q2_start_diff:.4f}\n")
        f_out.write(f"  Q3 (75th percentile): {q3_start_diff:.4f}\n")
        f_out.write(f"  Maximum: {max_start_diff:.4f}\n")
        f_out.write(f"  Mean Absolute Difference: {mean_abs_start_diff:.4f}\n\n")
        
        f_out.write("END_DIFF Statistics:\n")
        f_out.write(f"  Mean: {mean_end_diff:.4f}\n")
        f_out.write(f"  Standard Deviation: {std_end_diff:.4f}\n")
        f_out.write(f"  Mode: {mode_end_diff:.4f}\n")
        f_out.write(f"  Minimum: {min_end_diff:.4f}\n")
        f_out.write(f"  Q1 (25th percentile): {q1_end_diff:.4f}\n")
        f_out.write(f"  Q2 (50th percentile/Median): {q2_end_diff:.4f}\n")
        f_out.write(f"  Q3 (75th percentile): {q3_end_diff:.4f}\n")
        f_out.write(f"  Maximum: {max_end_diff:.4f}\n")
        f_out.write(f"  Mean Absolute Difference: {mean_abs_end_diff:.4f}\n\n")
        
        f_out.write("=== Summary ===\n")
        f_out.write(f"Overall Mean Absolute Start Difference: {mean_abs_start_diff:.4f}\n")
        f_out.write(f"Overall Mean Absolute End Difference: {mean_abs_end_diff:.4f}\n")
        f_out.write(f"Combined Mean Absolute Difference: {(mean_abs_start_diff + mean_abs_end_diff) / 2:.4f}\n")
    
    # 在控制台也输出简要统计信息
    print(f"\n=== Statistics Summary ===")
    print(f"Valid rows analyzed: {len(start_diffs)}")
    print(f"Mean start_diff: {mean_start_diff:.4f}")
    print(f"Mean end_diff: {mean_end_diff:.4f}")
    print(f"Mean absolute start_diff: {mean_abs_start_diff:.4f}")
    print(f"Mean absolute end_diff: {mean_abs_end_diff:.4f}")
    print(f"Statistics saved to: {stats_output_file}")
    
    return stats_output_file
    """读取 cdbgresult1.tsv（无表头，逗号分隔，11 列），返回 DataFrame"""
    try:
        df = pd.read_csv(file_path, sep=',', header=None, encoding='utf-8-sig')
        print(f"Read {len(df)} rows from {file_path}, columns: {df.shape[1]}")
        
        if df.shape[1] != 11:
            print(f"Error: Expected 11 columns in {file_path}, found {df.shape[1]}")
            sys.exit(1)
        return df
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        sys.exit(1)

def compare_splice_positions(relativesuppa_file, bubble_file, output_file="bubble_relativesuppa_comparison.tsv", na_file="na_transpairs_bubble.tsv"):
    """比较 cdbgresult1.tsv 和 relativesuppa.tsv 的剪接位点，输出结果"""
    # 读取文件
    relative_positions = read_relativesuppa(relativesuppa_file)
    bubble_df = read_bubble(bubble_file)

    # 打开输出文件
    with open(output_file, 'w') as f_out, \
         open(na_file, 'w') as f_na:
        # 输出文件头
        header = ['QueryName', 'SubjectName', 'newup', 'newaa', 'newdown',
                  'newaa_start', 'newaa_end', 'relative_start', 'relative_end',
                  'start_diff', 'end_diff', 'reason']
        f_out.write('\t'.join(header) + '\n')
        
        # NA文件头，包含原因列
        na_header = ['QueryName', 'SubjectName', 'newaa_start', 'newaa_end', 
                    'relative_start', 'relative_end', 'reason']
        f_na.write('\t'.join(na_header) + '\n')

        # 处理每行
        skipped = 0
        processed = 0
        filtered_large_diff = 0
        valid_rows = []
        na_transpairs = []

        for _, row in bubble_df.iterrows():
            if len(row) < 11:
                print(f"Skipping invalid row in {bubble_file}: {row.tolist()}")
                na_transpairs.append([str(row[0]) if len(row) >= 1 else 'Unknown', 
                                    str(row[1]) if len(row) >= 2 else 'Unknown', 
                                    'NA', 'NA', 'NA', 'NA', 'invalid_row'])
                skipped += 1
                continue

            qname, sname, newup, newaa, newdown, start_pos1, end_pos1, start_pos2, end_pos2, length1, length2 = row

            # 确定 newaa 来源
            try:
                dis1 = int(end_pos1) - int(start_pos1)
                dis2 = int(end_pos2) - int(start_pos2)
                start_pos1 = int(start_pos1)
                end_pos1 = int(end_pos1)
                start_pos2 = int(start_pos2)
                end_pos2 = int(end_pos2)
                
                # 确保bubble数据中的坐标也是start < end
                if start_pos1 > end_pos1:
                    start_pos1, end_pos1 = end_pos1, start_pos1
                if start_pos2 > end_pos2:
                    start_pos2, end_pos2 = end_pos2, start_pos2
                    
                # 重新计算距离
                dis1 = end_pos1 - start_pos1
                dis2 = end_pos2 - start_pos2
            except ValueError:
                print(f"Invalid position values in {bubble_file}: qname={qname}, sname={sname}")
                na_transpairs.append([str(qname), str(sname), 'NA', 'NA', 'NA', 'NA', 'invalid_positions'])
                skipped += 1
                continue

            if dis1 == 0 and dis2 > 2:  # newaa 来自 SubjectName
                source = 'subject'
                newaa_start = start_pos2
                newaa_end = end_pos2
            elif dis2 == 0 and dis1 > 2:  # newaa 来自 QueryName
                source = 'query'
                newaa_start = start_pos1
                newaa_end = end_pos1
            else:
                print(f"Skipping: qname={qname}, sname={sname}, dis1={dis1}, dis2={dis2} (not a clear splice event)")
                na_transpairs.append([str(qname), str(sname), 'NA', 'NA', 'NA', 'NA', 'not_clear_splice_event'])
                skipped += 1
                continue

            # 查找 relativesuppa.tsv 中的相对位置
            key1 = (str(qname), str(sname))
            key2 = (str(sname), str(qname))
            
            relative_start = None
            relative_end = None
            rel_newaa_start = None
            rel_newaa_end = None
            used_key = None
            
            # 尝试第一个key
            if key1 in relative_positions:
                if source == 'query':
                    temp_relative_start, temp_relative_end, temp_rel_newaa_start, temp_rel_newaa_end = relative_positions[key1]
                else:
                    temp_rel_newaa_start, temp_rel_newaa_end, temp_relative_start, temp_relative_end = relative_positions[key1]
                
                # 检查relative序列长度
                temp_relative_length = abs(temp_relative_end - temp_relative_start)
                if temp_relative_length > 1:  # 长度大于1，使用这个结果
                    relative_start, relative_end = temp_relative_start, temp_relative_end
                    rel_newaa_start, rel_newaa_end = temp_rel_newaa_start, temp_rel_newaa_end
                    used_key = key1
                    print(f"Using key1 {key1} with relative_length={temp_relative_length}")
                else:
                    print(f"Key1 {key1} has short relative_length={temp_relative_length}, trying key2...")
            
            # 如果第一个key不存在或者长度为1，尝试第二个key
            if relative_start is None and key2 in relative_positions:
                if source == 'subject':  # 注意这里source判断是相反的
                    temp_relative_start, temp_relative_end, temp_rel_newaa_start, temp_rel_newaa_end = relative_positions[key2]
                else:
                    temp_rel_newaa_start, temp_rel_newaa_end, temp_relative_start, temp_relative_end = relative_positions[key2]
                
                # 检查relative序列长度
                temp_relative_length = abs(temp_relative_end - temp_relative_start)
                if temp_relative_length > 1:  # 长度大于1，使用这个结果
                    relative_start, relative_end = temp_relative_start, temp_relative_end
                    rel_newaa_start, rel_newaa_end = temp_rel_newaa_start, temp_rel_newaa_end
                    used_key = key2
                    print(f"Using key2 {key2} with relative_length={temp_relative_length}")
                else:
                    print(f"Key2 {key2} also has short relative_length={temp_relative_length}")
            
            # 如果两个key都不存在或都是短序列
            if relative_start is None:
                if key1 not in relative_positions and key2 not in relative_positions:
                    print(f"No match in {relativesuppa_file}: qname={qname}, sname={sname}, source={source}")
                    na_transpairs.append([str(qname), str(sname), str(newaa_start), str(newaa_end), 'NA', 'NA', 'no_match_in_relativesuppa'])
                else:
                    print(f"Both keys have short sequences: qname={qname}, sname={sname}, source={source}")
                    na_transpairs.append([str(qname), str(sname), str(newaa_start), str(newaa_end), 'NA', 'NA', 'both_keys_short_sequences'])
                skipped += 1
                continue

            # 计算位置差值 (移除了减1的操作)
            start_diff = newaa_start - relative_start
            end_diff = newaa_end - relative_end

            # 最终检查relative序列长度（这里应该已经>1了，但保险起见再检查一次）
            relative_length = relative_end - relative_start if relative_end >= relative_start else relative_start - relative_end
            if relative_length <= 1:
                print(f"Final check: Filtering row with short relative sequence (length={relative_length}): qname={qname}, sname={sname}")
                na_transpairs.append([str(qname), str(sname), str(newaa_start), str(newaa_end), 
                                    str(relative_start), str(relative_end), f'final_check_short_relative_sequence_length_{relative_length}'])
                filtered_large_diff += 1
                continue

            # 过滤 |start_diff| > 50 或 |end_diff| > 50
            if abs(start_diff) > 50 or abs(end_diff) > 50:
                print(f"Filtering row with large diff (|start_diff|={abs(start_diff)} or |end_diff|={abs(end_diff)}): qname={qname}, sname={sname}")
                na_transpairs.append([str(qname), str(sname), str(newaa_start), str(newaa_end), 
                                    str(relative_start), str(relative_end), f'large_diff_start_{start_diff}_end_{end_diff}'])
                filtered_large_diff += 1
                continue

            # 存储有效行，包含使用的key信息
            valid_rows.append([
                str(qname), str(sname), str(newup), str(newaa), str(newdown),
                str(newaa_start), str(newaa_end),
                str(relative_start), str(relative_end),
                f"{start_diff:.2f}", f"{end_diff:.2f}", f'valid_used_{used_key}'
            ])
            processed += 1

        # 写入有效行
        for row in valid_rows:
            f_out.write('\t'.join(row) + '\n')

        # 写入无效转录本对
        for pair in na_transpairs:
            f_na.write('\t'.join(pair) + '\n')

        # 输出统计信息
        print(f"Processed {processed} valid rows (|start_diff| ≤ 50 and |end_diff| ≤ 50), "
              f"skipped {skipped} rows (invalid or no match), "
              f"filtered {filtered_large_diff} rows with |start_diff| or |end_diff| > 50 or short sequences")
        print(f"Wrote {processed} valid rows to {output_file}")
        print(f"Wrote {len(na_transpairs)} NA or filtered transpairs to {na_file}")

        if processed == 0:
            print("Warning: No valid rows found (|start_diff| ≤ 50 and |end_diff| ≤ 50).")
        else:
            # 计算并输出统计量
            output_base = output_file.rsplit('.', 1)[0]  # 移除扩展名
            stats_file = calculate_statistics(valid_rows, output_base)
            print(f"Statistical analysis completed, results saved to {stats_file}")

def main():
    if len(sys.argv) != 3:
        print("Usage: ./compare_splice_positions.py relativesuppa.tsv cdbgresult1.tsv")
        sys.exit(1)
    compare_splice_positions(sys.argv[1], sys.argv[2])

if __name__ == "__main__":
    main()