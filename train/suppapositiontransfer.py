#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import sys
import os
import logging
from Bio import SeqIO

# 设置日志文件
logging.basicConfig(filename='skipped_events.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 检查命令行参数
if len(sys.argv) < 4:
    print("Usage: python suppapositiontransfer.py <cdna_fasta> <gtf_file> <output_file>")
    print("Example: python suppapositiontransfer.py rice_all.transcripts.fa rice_all.gtf riceposition.tsv")
    sys.exit(1)

# 从命令行参数获取文件路径
cdna_file = sys.argv[1]
gtf_file = sys.argv[2]
output_file = sys.argv[3]

# 打开输出文件
output_handle = open(output_file, 'w')

# 写入表头
print("event_type\tchromosome\ttranscript1\ttranscript2\tt1_start\tt1_end\tt2_start\tt2_end\tgenomic_start1\tgenomic_end1\tgenomic_start2\tgenomic_end2\tas_length", file=output_handle)

print(f"Loading transcripts from {cdna_file}...", file=sys.stderr)
position = {}
tras = {}

# 加载 cDNA 序列，去掉版本号
for seq_record in SeqIO.parse(cdna_file, "fasta"):
    transcript_id = seq_record.id.split(".")[0]  # 去掉版本号
    tras[transcript_id] = str(seq_record.seq).upper()
    logging.info(f"Loaded transcript: {transcript_id}")

print(f"Loaded {len(tras)} transcripts", file=sys.stderr)
# Print first 5 transcript IDs for debugging
print("First 5 transcript IDs in tras:", list(tras.keys())[:5], file=sys.stderr)

print(f"Processing GTF file {gtf_file}...", file=sys.stderr)
ioe = open(gtf_file, "r")
trans = "trans"
pos = 1
line_count = 0
exon_count = 0

# Debugging: Print first 5 non-comment lines of GTF
print("Inspecting first 5 non-comment lines of GTF file...", file=sys.stderr)
non_comment_lines = 0
for line in ioe:
    if not line.startswith('#'):
        print(f"GTF line: {line.strip()}", file=sys.stderr)
        non_comment_lines += 1
        if non_comment_lines >= 5:
            break
ioe.seek(0)  # Reset file pointer

while True:
    lines = ioe.readlines(100000)
    if not lines:
        break
    for line in lines:
        line = line.strip('\n')
        line_count += 1
        
        if line.startswith("#"):
            continue
            
        if '\texon\t' not in line:
            continue
            
        # 更灵活的正则表达式，适配多种 GTF 格式（包括 MSU 格式）
        searchObj = re.search(r'^(\S+)\t([^\t]*)\texon\t(\d+)\t(\d+)\t([^\t]*)\t([+-])\t([^\t]*)\t(.*)', line)
        
        if searchObj:
            chrom = searchObj.group(1)
            start = searchObj.group(3)
            end = searchObj.group(4)
            strand = searchObj.group(6)
            attributes = searchObj.group(8)
            
            # 提取 gene_id 和 transcript_id，适配 ID 和 Parent 字段
            gene_id_match = re.search(r'(?:gene_id|ID)\s*["\']?([^"\';]+)["\']?', attributes)
            transcript_id_match = re.search(r'(?:transcript_id|Parent)\s*["\']?([^"\';]+)["\']?', attributes)
            
            if gene_id_match and transcript_id_match:
                gene_id = gene_id_match.group(1).split(".")[0]  # 去掉版本号
                transcript_id = transcript_id_match.group(1).split(".")[0]  # 去掉版本号
                
                exon_count += 1
                if exon_count % 10000 == 0:
                    print(f"Processed {exon_count} exons...", file=sys.stderr)
                
                if transcript_id in tras:
                    if transcript_id != trans:
                        trans = transcript_id
                        length = len(tras[trans])
                        pos = 1
                        if strand == "+":
                            position[trans + start] = pos
                            pos = pos + int(end) - int(start)
                            position[trans + end] = pos
                        elif strand == "-":
                            position[trans + start] = length - pos + 1
                            pos = pos + int(end) - int(start)
                            position[trans + end] = length - pos + 1
                        pos = pos + 1
                    else:
                        if strand == "+":
                            position[trans + start] = pos
                            pos = pos + int(end) - int(start)
                            position[trans + end] = pos
                        elif strand == "-":
                            position[trans + start] = length - pos + 1
                            pos = pos + int(end) - int(start)
                            position[trans + end] = length - pos + 1
                        pos = pos + 1
                else:
                    logging.warning(f"Transcript {transcript_id} not found in FASTA file")
            else:
                logging.warning(f"Could not parse gene_id or transcript_id from line: {line}")
        else:
            logging.warning(f"Could not parse exon line: {line}")

ioe.close()
print(f"Finished processing GTF. Total lines: {line_count}, Total exons: {exon_count}", file=sys.stderr)
print(f"Position mappings created: {len(position)}", file=sys.stderr)
# Print first 5 position keys for debugging
print("First 5 position keys:", list(position.keys())[:5], file=sys.stderr)

def inspect_ioe_file(filename):
    """Inspect the first 5 non-header lines of an IOE file for debugging"""
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found", file=sys.stderr)
        logging.warning(f"IOE file {filename} not found")
        return False, 0
    with open(filename, "r") as ioe:
        lines = ioe.readlines()
        if not lines:
            print(f"Warning: {filename} is empty", file=sys.stderr)
            logging.warning(f"IOE file {filename} is empty")
            return False, 0
        non_header_lines = [line.strip() for line in lines if not line.startswith('seqname\t')]
        total_lines = len(lines)
        print(f"Inspecting up to 5 non-header lines of {filename} (total lines: {total_lines})...", file=sys.stderr)
        for i, line in enumerate(non_header_lines[:5], 1):
            print(f"{filename} line {i}: {line}", file=sys.stderr)
        return bool(non_header_lines), total_lines

def a3():
    """Process A3 (Alternative 3' splice site) events"""
    print("Processing A3 events...", file=sys.stderr)
    filename = "events_A3_strict.ioe"
    has_data, total_lines = inspect_ioe_file(filename)
    if not has_data:
        print(f"No data lines in {filename}, skipping A3 events", file=sys.stderr)
        return
        
    ioe = open(filename, "r")
    event_count = 0
    line_count = 0
    
    while True:
        lines = ioe.readlines(100000)
        if not lines:
            break
        for line in lines:
            line = line.strip('\n')
            line_count += 1
            if line.startswith('seqname\t'):
                continue
                
            parts = line.split("\t")
            if len(parts) < 5:
                logging.warning(f"Malformed A3 line (line {line_count}): {line}")
                continue
                
            chrom = parts[0].lower().replace('chr', '')  # Normalize chromosome
            gene_id = parts[1].split(".")[0]
            event_info = parts[2]
            alter_trans = [tid.split(".")[0] for tid in parts[3].split(",")]
            all_trans = [tid.split(".")[0] for tid in parts[4].split(",")]
            
            # Log transcript IDs for debugging
            logging.info(f"A3 event (line {line_count}): alter_trans={alter_trans}, all_trans={all_trans}")
            
            searchObj = re.search(r'(\S+);A3:(\S+):(\d+)-(\d+):(\d+)-(\d+):([+-])', event_info)
            
            if searchObj:
                event_gene_id = searchObj.group(1).split(".")[0]
                chrom_event = searchObj.group(2).lower().replace('chr', '')
                start1, end1 = int(searchObj.group(3)), int(searchObj.group(4))
                start2, end2 = int(searchObj.group(5)), int(searchObj.group(6))
                strand = searchObj.group(7)
                
                if chrom != chrom_event:
                    logging.warning(f"Chromosome mismatch in A3 event (line {line_count}): {line} (chrom: {chrom}, chrom_event: {chrom_event})")
                    continue
                
                if strand == "+":
                    for i in alter_trans:
                        for j in all_trans:
                            if j in alter_trans:
                                continue
                            if i in tras and j in tras:
                                asregion = end2 - end1
                                key1 = i + str(start1)
                                key2 = i + str(end1)
                                key3 = j + str(start2)
                                key4 = j + str(end2)
                                
                                # Log keys for debugging
                                logging.info(f"A3 event (line {line_count}): checking keys {key1}, {key2}, {key3}, {key4}")
                                
                                if key1 in position and key2 in position and key3 in position and key4 in position:
                                    print(f"A3+\t{chrom}\t{i}\t{j}\t{position[key1]}\t{position[key2]+asregion}\t{position[key3]}\t{position[key4]}\t{start1}\t{end1}\t{start2}\t{end2}\t{asregion}", file=output_handle)
                                    event_count += 1
                                else:
                                    logging.warning(f"Skipping A3 event due to missing position keys (line {line_count}): {line} (keys: {key1}, {key2}, {key3}, {key4})")
                            else:
                                logging.warning(f"Skipping A3 event due to missing transcripts {i} or {j} (line {line_count}): {line}")
                                    
                elif strand == "-":
                    for i in alter_trans:
                        for j in all_trans:
                            if j in alter_trans:
                                continue
                            if i in tras and j in tras:
                                asregion = start1 - start2
                                key1 = i + str(end1)
                                key2 = i + str(start1)
                                key3 = j + str(end2)
                                key4 = j + str(start2)
                                
                                logging.info(f"A3 event (line {line_count}): checking keys {key1}, {key2}, {key3}, {key4}")
                                
                                if key1 in position and key2 in position and key3 in position and key4 in position:
                                    print(f"A3-\t{chrom}\t{i}\t{j}\t{position[key1]}\t{position[key2]+asregion}\t{position[key3]}\t{position[key4]}\t{end1}\t{start1}\t{end2}\t{start2}\t{asregion}", file=output_handle)
                                    event_count += 1
                                else:
                                    logging.warning(f"Skipping A3 event due to missing position keys (line {line_count}): {line} (keys: {key1}, {key2}, {key3}, {key4})")
                            else:
                                logging.warning(f"Skipping A3 event due to missing transcripts {i} or {j} (line {line_count}): {line}")
            else:
                logging.warning(f"Could not parse A3 event info (line {line_count}): {event_info}")
    
    ioe.close()
    print(f"Processed {event_count} A3 events (total lines: {line_count})", file=sys.stderr)

def a5():
    """Process A5 (Alternative 5' splice site) events"""
    print("Processing A5 events...", file=sys.stderr)
    filename = "events_A5_strict.ioe"
    has_data, total_lines = inspect_ioe_file(filename)
    if not has_data:
        print(f"No data lines in {filename}, skipping A5 events", file=sys.stderr)
        return
        
    ioe = open(filename, "r")
    event_count = 0
    line_count = 0
    
    while True:
        lines = ioe.readlines(100000)
        if not lines:
            break
        for line in lines:
            line = line.strip('\n')
            line_count += 1
            if line.startswith('seqname\t'):
                continue
                
            parts = line.split("\t")
            if len(parts) < 5:
                logging.warning(f"Malformed A5 line (line {line_count}): {line}")
                continue
                
            chrom = parts[0].lower().replace('chr', '')
            gene_id = parts[1].split(".")[0]
            event_info = parts[2]
            alter_trans = [tid.split(".")[0] for tid in parts[3].split(",")]
            all_trans = [tid.split(".")[0] for tid in parts[4].split(",")]
            
            logging.info(f"A5 event (line {line_count}): alter_trans={alter_trans}, all_trans={all_trans}")
            
            searchObj = re.search(r'(\S+);A5:(\S+):(\d+)-(\d+):(\d+)-(\d+):([+-])', event_info)
            
            if searchObj:
                event_gene_id = searchObj.group(1).split(".")[0]
                chrom_event = searchObj.group(2).lower().replace('chr', '')
                start1, end1 = int(searchObj.group(3)), int(searchObj.group(4))
                start2, end2 = int(searchObj.group(5)), int(searchObj.group(6))
                strand = searchObj.group(7)
                
                if chrom != chrom_event:
                    logging.warning(f"Chromosome mismatch in A5 event (line {line_count}): {line} (chrom: {chrom}, chrom_event: {chrom_event})")
                    continue
                
                if strand == "+":
                    for i in alter_trans:
                        for j in all_trans:
                            if j in alter_trans:
                                continue
                            if i in tras and j in tras:
                                asregion = start1 - start2
                                key1 = i + str(start1)
                                key2 = i + str(end1)
                                key3 = j + str(start2)
                                key4 = j + str(end2)
                                
                                logging.info(f"A5 event (line {line_count}): checking keys {key1}, {key2}, {key3}, {key4}")
                                
                                if key1 in position and key2 in position and key3 in position and key4 in position:
                                    print(f"A5+\t{chrom}\t{i}\t{j}\t{position[key1]-asregion}\t{position[key2]}\t{position[key3]}\t{position[key4]}\t{end1}\t{start1}\t{end2}\t{start2}\t{asregion}", file=output_handle)
                                    event_count += 1
                                else:
                                    logging.warning(f"Skipping A5 event due to missing position keys (line {line_count}): {line} (keys: {key1}, {key2}, {key3}, {key4})")
                            else:
                                logging.warning(f"Skipping A5 event due to missing transcripts {i} or {j} (line {line_count}): {line}")
                                    
                elif strand == "-":
                    for i in alter_trans:
                        for j in all_trans:
                            if j in alter_trans:
                                continue
                            if i in tras and j in tras:
                                asregion = end2 - end1
                                key1 = i + str(end1)
                                key2 = i + str(start1)
                                key3 = j + str(end2)
                                key4 = j + str(start2)
                                
                                logging.info(f"A5 event (line {line_count}): checking keys {key1}, {key2}, {key3}, {key4}")
                                
                                if key1 in position and key2 in position and key3 in position and key4 in position:
                                    print(f"A5-\t{chrom}\t{i}\t{j}\t{position[key1]-asregion}\t{position[key2]}\t{position[key3]}\t{position[key4]}\t{start1}\t{end1}\t{start2}\t{end2}\t{asregion}", file=output_handle)
                                    event_count += 1
                                else:
                                    logging.warning(f"Skipping A5 event due to missing position keys (line {line_count}): {line} (keys: {key1}, {key2}, {key3}, {key4})")
                            else:
                                logging.warning(f"Skipping A5 event due to missing transcripts {i} or {j} (line {line_count}): {line}")
            else:
                logging.warning(f"Could not parse A5 event info (line {line_count}): {event_info}")
    
    ioe.close()
    print(f"Processed {event_count} A5 events (total lines: {line_count})", file=sys.stderr)

def se():
    """Process SE (Skipped Exon) events"""
    print("Processing SE events...", file=sys.stderr)
    filename = "events_SE_strict.ioe"
    has_data, total_lines = inspect_ioe_file(filename)
    if not has_data:
        print(f"No data lines in {filename}, skipping SE events", file=sys.stderr)
        return
        
    ioe = open(filename, "r")
    event_count = 0
    line_count = 0
    
    while True:
        lines = ioe.readlines(100000)
        if not lines:
            break
        for line in lines:
            line = line.strip('\n')
            line_count += 1
            if line.startswith('seqname\t'):
                continue
                
            parts = line.split("\t")
            if len(parts) < 5:
                logging.warning(f"Malformed SE line (line {line_count}): {line}")
                continue
                
            chrom = parts[0].lower().replace('chr', '')
            gene_id = parts[1].split(".")[0]
            event_info = parts[2]
            alter_trans = [tid.split(".")[0] for tid in parts[3].split(",")]
            all_trans = [tid.split(".")[0] for tid in parts[4].split(",")]
            
            logging.info(f"SE event (line {line_count}): alter_trans={alter_trans}, all_trans={all_trans}")
            
            searchObj = re.search(r'(\S+);SE:(\S+):(\d+)-(\d+):(\d+)-(\d+):([+-])', event_info)
            
            if searchObj:
                event_gene_id = searchObj.group(1).split(".")[0]
                chrom_event = searchObj.group(2).lower().replace('chr', '')
                start1, end1 = int(searchObj.group(3)), int(searchObj.group(4))
                start2, end2 = int(searchObj.group(5)), int(searchObj.group(6))
                strand = searchObj.group(7)
                
                if chrom != chrom_event:
                    logging.warning(f"Chromosome mismatch in SE event (line {line_count}): {line} (chrom: {chrom}, chrom_event: {chrom_event})")
                    continue
                
                if strand == "+":
                    for i in alter_trans:
                        for j in all_trans:
                            if j in alter_trans:
                                continue
                            if i in tras and j in tras:
                                se_length = end1 - start1
                                key1 = i + str(start1)
                                key2 = i + str(end1)
                                key3 = j + str(start2)
                                key4 = j + str(end2)
                                
                                logging.info(f"SE event (line {line_count}): checking keys {key1}, {key2}, {key3}, {key4}")
                                
                                if key1 in position and key2 in position and key3 in position and key4 in position:
                                    print(f"SE+\t{chrom}\t{i}\t{j}\t{position[key1]}\t{position[key2]}\t{position[key3]}\t{position[key4]}\t{start1}\t{end1}\t{start2}\t{end2}\t{se_length}", file=output_handle)
                                    event_count += 1
                                else:
                                    logging.warning(f"Skipping SE event due to missing position keys (line {line_count}): {line} (keys: {key1}, {key2}, {key3}, {key4})")
                            else:
                                logging.warning(f"Skipping SE event due to missing transcripts {i} or {j} (line {line_count}): {line}")
                                    
                elif strand == "-":
                    for i in alter_trans:
                        for j in all_trans:
                            if j in alter_trans:
                                continue
                            if i in tras and j in tras:
                                se_length = end1 - start1
                                key1 = i + str(end1)
                                key2 = i + str(start1)
                                key3 = j + str(end2)
                                key4 = j + str(start2)
                                
                                logging.info(f"SE event (line {line_count}): checking keys {key1}, {key2}, {key3}, {key4}")
                                
                                if key1 in position and key2 in position and key3 in position and key4 in position:
                                    print(f"SE-\t{chrom}\t{i}\t{j}\t{position[key1]}\t{position[key2]}\t{position[key3]}\t{position[key4]}\t{start1}\t{end1}\t{start2}\t{end2}\t{se_length}", file=output_handle)
                                    event_count += 1
                                else:
                                    logging.warning(f"Skipping SE event due to missing position keys (line {line_count}): {line} (keys: {key1}, {key2}, {key3}, {key4})")
                            else:
                                logging.warning(f"Skipping SE event due to missing transcripts {i} or {j} (line {line_count}): {line}")
            else:
                logging.warning(f"Could not parse SE event info (line {line_count}): {event_info}")
    
    ioe.close()
    print(f"Processed {event_count} SE events (total lines: {line_count})", file=sys.stderr)

def ri():
    """Process RI (Retained Intron) events"""
    print("Processing RI events...", file=sys.stderr)
    filename = "events_RI_strict.ioe"
    has_data, total_lines = inspect_ioe_file(filename)
    if not has_data:
        print(f"No data lines in {filename}, skipping RI events", file=sys.stderr)
        return
        
    ioe = open(filename, "r")
    event_count = 0
    line_count = 0
    
    while True:
        lines = ioe.readlines(100000)
        if not lines:
            break
        for line in lines:
            line = line.strip('\n')
            line_count += 1
            if line.startswith('seqname\t'):
                continue
                
            parts = line.split("\t")
            if len(parts) < 5:
                logging.warning(f"Malformed RI line (line {line_count}): {line}")
                continue
                
            chrom = parts[0].lower().replace('chr', '')
            gene_id = parts[1].split(".")[0]
            event_info = parts[2]
            alter_trans = [tid.split(".")[0] for tid in parts[3].split(",")]
            all_trans = [tid.split(".")[0] for tid in parts[4].split(",")]
            
            logging.info(f"RI event (line {line_count}): alter_trans={alter_trans}, all_trans={all_trans}")
            
            searchObj = re.search(r'(\S+);RI:(\S+):(\d+):(\d+)-(\d+):(\d+):([+-])', event_info)
            
            if searchObj:
                event_gene_id = searchObj.group(1).split(".")[0]
                chrom_event = searchObj.group(2).lower().replace('chr', '')
                start1, start2, end2 = int(searchObj.group(3)), int(searchObj.group(4)), int(searchObj.group(5))
                end1 = int(searchObj.group(6))
                strand = searchObj.group(7)
                
                if chrom != chrom_event:
                    logging.warning(f"Chromosome mismatch in RI event (line {line_count}): {line} (chrom: {chrom}, chrom_event: {chrom_event})")
                    continue
                
                if strand == "+":
                    for i in alter_trans:
                        for j in all_trans:
                            if j in alter_trans:
                                continue
                            if i in tras and j in tras:
                                ri_length = end2 - start2
                                key1 = i + str(start2)
                                key2 = i + str(end2)
                                key3 = j + str(start1)
                                key4 = j + str(end1)
                                
                                logging.info(f"RI event (line {line_count}): checking keys {key1}, {key2}, {key3}, {key4}")
                                
                                if key1 in position and key2 in position and key3 in position and key4 in position:
                                    print(f"RI+\t{chrom}\t{i}\t{j}\t{position[key1]}\t{position[key2]}\t{position[key3]}\t{position[key4]}\t{start2}\t{end2}\t{start1}\t{end1}\t{ri_length}", file=output_handle)
                                    event_count += 1
                                else:
                                    logging.warning(f"Skipping RI event due to missing position keys (line {line_count}): {line} (keys: {key1}, {key2}, {key3}, {key4})")
                            else:
                                logging.warning(f"Skipping RI event due to missing transcripts {i} or {j} (line {line_count}): {line}")
                                    
                elif strand == "-":
                    for i in alter_trans:
                        for j in all_trans:
                            if j in alter_trans:
                                continue
                            if i in tras and j in tras:
                                ri_length = end2 - start2
                                key1 = i + str(end2)
                                key2 = i + str(start2)
                                key3 = j + str(end1)
                                key4 = j + str(start1)
                                
                                logging.info(f"RI event (line {line_count}): checking keys {key1}, {key2}, {key3}, {key4}")
                                
                                if key1 in position and key2 in position and key3 in position and key4 in position:
                                    print(f"RI-\t{chrom}\t{i}\t{j}\t{position[key1]}\t{position[key2]}\t{position[key3]}\t{position[key4]}\t{start2}\t{end2}\t{start1}\t{end1}\t{ri_length}", file=output_handle)
                                    event_count += 1
                                else:
                                    logging一同ling.warning(f"Skipping RI event due to missing position keys (line {line_count}): {line} (keys: {key1}, {key2}, {key3}, {key4})")
                            else:
                                logging.warning(f"Skipping RI event due to missing transcripts {i} or {j} (line {line_count}): {line}")
            else:
                logging.warning(f"Could not parse RI event info (line {line_count}): {event_info}")
    
    ioe.close()
    print(f"Processed {event_count} RI events (total lines: {line_count})", file=sys.stderr)

def mx():
    """Process MX (Mutually Exclusive) events"""
    print("Processing MX events...", file=sys.stderr)
    filename = "events_MX_strict.ioe"
    has_data, total_lines = inspect_ioe_file(filename)
    if not has_data:
        print(f"No data lines in {filename}, skipping MX events", file=sys.stderr)
        return
        
    ioe = open(filename, "r")
    event_count = 0
    line_count = 0
    
    while True:
        lines = ioe.readlines(100000)
        if not lines:
            break
        for line in lines:
            line = line.strip('\n')
            line_count += 1
            if line.startswith('seqname\t'):
                continue
                
            parts = line.split("\t")
            if len(parts) < 5:
                logging.warning(f"Malformed MX line (line {line_count}): {line}")
                continue
                
            chrom = parts[0].lower().replace('chr', '')
            gene_id = parts[1].split(".")[0]
            event_info = parts[2]
            alter_trans = [tid.split(".")[0] for tid in parts[3].split(",")]
            all_trans = [tid.split(".")[0] for tid in parts[4].split(",")]
            
            logging.info(f"MX event (line {line_count}): alter_trans={alter_trans}, all_trans={all_trans}")
            
            searchObj = re.search(r'(\S+);MX:(\S+):(\d+)-(\d+):(\d+)-(\d+):(\d+)-(\d+):(\d+)-(\d+):([+-])', event_info)
            
            if searchObj:
                event_gene_id = searchObj.group(1).split(".")[0]
                chrom_event = searchObj.group(2).lower().replace('chr', '')
                start1, end1 = int(searchObj.group(3)), int(searchObj.group(4))
                start2, end2 = int(searchObj.group(5)), int(searchObj.group(6))
                start3, end3 = int(searchObj.group(7)), int(searchObj.group(8))
                start4, end4 = int(searchObj.group(9)), int(searchObj.group(10))
                strand = searchObj.group(11)
                
                if chrom != chrom_event:
                    logging.warning(f"Chromosome mismatch in MX event (line {line_count}): {line} (chrom: {chrom}, chrom_event: {chrom_event})")
                    continue
                
                if strand == "+":
                    for i in alter_trans:
                        for j in all_trans:
                            if j in alter_trans:
                                continue
                            if i in tras and j in tras:
                                mx_length = end1 - start1
                                key1 = i + str(start1)
                                key2 = i + str(end1)
                                key3 = j + str(start3)
                                key4 = j + str(end3)
                                
                                logging.info(f"MX event (line {line_count}): checking keys {key1}, {key2}, {key3}, {key4}")
                                
                                if key1 in position and key2 in position and key3 in position and key4 in position:
                                    print(f"MX+\t{chrom}\t{i}\t{j}\t{position[key1]}\t{position[key2]}\t{position[key3]}\t{position[key4]}\t{start1}\t{end1}\t{start3}\t{end3}\t{mx_length}", file=output_handle)
                                    event_count += 1
                                else:
                                    logging.warning(f"Skipping MX event due to missing position keys (line {line_count}): {line} (keys: {key1}, {key2}, {key3}, {key4})")
                            else:
                                logging.warning(f"Skipping MX event due to missing transcripts {i} or {j} (line {line_count}): {line}")
                                    
                elif strand == "-":
                    for i in alter_trans:
                        for j in all_trans:
                            if j in alter_trans:
                                continue
                            if i in tras and j in tras:
                                mx_length = end1 - start1
                                key1 = i + str(end1)
                                key2 = i + str(start1)
                                key3 = j + str(end3)
                                key4 = j + str(start3)
                                
                                logging.info(f"MX event (line {line_count}): checking keys {key1}, {key2}, {key3}, {key4}")
                                
                                if key1 in position and key2 in position and key3 in position and key4 in position:
                                    print(f"MX-\t{chrom}\t{i}\t{j}\t{position[key1]}\t{position[key2]}\t{position[key3]}\t{position[key4]}\t{start1}\t{end1}\t{start3}\t{end3}\t{mx_length}", file=output_handle)
                                    event_count += 1
                                else:
                                    logging.warning(f"Skipping MX event due to missing position keys (line {line_count}): {line} (keys: {key1}, {key2}, {key3}, {key4})")
                            else:
                                logging.warning(f"Skipping MX event due to missing transcripts {i} or {j} (line {line_count}): {line}")
            else:
                logging.warning(f"Could not parse MX event info (line {line_count}): {event_info}")
    
    ioe.close()
    print(f"Processed {event_count} MX events (total lines: {line_count})", file=sys.stderr)

def af():
    """Process AF (Alternative First) events"""
    print("Processing AF events...", file=sys.stderr)
    filename = "events_AF_strict.ioe"
    has_data, total_lines = inspect_ioe_file(filename)
    if not has_data:
        print(f"No data lines in {filename}, skipping AF events", file=sys.stderr)
        return
        
    ioe = open(filename, "r")
    event_count = 0
    line_count = 0
    
    while True:
        lines = ioe.readlines(100000)
        if not lines:
            break
        for line in lines:
            line = line.strip('\n')
            line_count += 1
            if line.startswith('seqname\t'):
                continue
                
            parts = line.split("\t")
            if len(parts) < 5:
                logging.warning(f"Malformed AF line (line {line_count}): {line}")
                continue
                
            chrom = parts[0].lower().replace('chr', '')
            gene_id = parts[1].split(".")[0]
            event_info = parts[2]
            alter_trans = [tid.split(".")[0] for tid in parts[3].split(",")]
            all_trans = [tid.split(".")[0] for tid in parts[4].split(",")]
            
            logging.info(f"AF event (line {line_count}): alter_trans={alter_trans}, all_trans={all_trans}")
            
            searchObj = re.search(r'(\S+);AF:(\S+):(\d+)-(\d+):(\d+):(\d+)-(\d+):(\d+):([+-])', event_info)
            
            if searchObj:
                event_gene_id = searchObj.group(1).split(".")[0]
                chrom_event = searchObj.group(2).lower().replace('chr', '')
                start1, end1 = int(searchObj.group(3)), int(searchObj.group(4))
                mid = int(searchObj.group(5))
                start2, end2 = int(searchObj.group(6)), int(searchObj.group(7))
                end3 = int(searchObj.group(8))
                strand = searchObj.group(9)
                
                if chrom != chrom_event:
                    logging.warning(f"Chromosome mismatch in AF event (line {line_count}): {line} (chrom: {chrom}, chrom_event: {chrom_event})")
                    continue
                
                if strand == "+":
                    for i in alter_trans:
                        for j in all_trans:
                            if j in alter_trans:
                                continue
                            if i in tras and j in tras:
                                af_length = end1 - start1
                                key1 = i + str(start1)
                                key2 = i + str(end1)
                                key3 = j + str(start2)
                                key4 = j + str(end2)
                                
                                logging.info(f"AF event (line {line_count}): checking keys {key1}, {key2}, {key3}, {key4}")
                                
                                if key1 in position and key2 in position and key3 in position and key4 in position:
                                    print(f"AF+\t{chrom}\t{i}\t{j}\t{position[key1]}\t{position[key2]}\t{position[key3]}\t{position[key4]}\t{start1}\t{end1}\t{start2}\t{end2}\t{af_length}", file=output_handle)
                                    event_count += 1
                                else:
                                    logging.warning(f"Skipping AF event due to missing position keys (line {line_count}): {line} (keys: {key1}, {key2}, {key3}, {key4})")
                            else:
                                logging.warning(f"Skipping AF event due to missing transcripts {i} or {j} (line {line_count}): {line}")
                                    
                elif strand == "-":
                    for i in alter_trans:
                        for j in all_trans:
                            if j in alter_trans:
                                continue
                            if i in tras and j in tras:
                                af_length = end1 - start1
                                key1 = i + str(end1)
                                key2 = i + str(start1)
                                key3 = j + str(end2)
                                key4 = j + str(start2)
                                
                                logging.info(f"AF event (line {line_count}): checking keys {key1}, {key2}, {key3}, {key4}")
                                
                                if key1 in position and key2 in position and key3 in position and key4 in position:
                                    print(f"AF-\t{chrom}\t{i}\t{j}\t{position[key1]}\t{position[key2]}\t{position[key3]}\t{position[key4]}\t{start1}\t{end1}\t{start2}\t{end2}\t{af_length}", file=output_handle)
                                    event_count += 1
                                else:
                                    logging.warning(f"Skipping AF event due to missing position keys (line {line_count}): {line} (keys: {key1}, {key2}, {key3}, {key4})")
                            else:
                                logging.warning(f"Skipping AF event due to missing transcripts {i} or {j} (line {line_count}): {line}")
            else:
                logging.warning(f"Could not parse AF event info (line {line_count}): {event_info}")
    
    ioe.close()
    print(f"Processed {event_count} AF events (total lines: {line_count})", file=sys.stderr)

def al():
    """Process AL (Alternative Last) events"""
    print("Processing AL events...", file=sys.stderr)
    filename = "events_AL_strict.ioe"
    has_data, total_lines = inspect_ioe_file(filename)
    if not has_data:
        print(f"No data lines in {filename}, skipping AL events", file=sys.stderr)
        return
        
    ioe = open(filename, "r")
    event_count = 0
    line_count = 0
    
    while True:
        lines = ioe.readlines(100000)
        if not lines:
            break
        for line in lines:
            line = line.strip('\n')
            line_count += 1
            if line.startswith('seqname\t'):
                continue
                
            parts = line.split("\t")
            if len(parts) < 5:
                logging.warning(f"Malformed AL line (line {line_count}): {line}")
                continue
                
            chrom = parts[0].lower().replace('chr', '')
            gene_id = parts[1].split(".")[0]
            event_info = parts[2]
            alter_trans = [tid.split(".")[0] for tid in parts[3].split(",")]
            all_trans = [tid.split(".")[0] for tid in parts[4].split(",")]
            
            logging.info(f"AL event (line {line_count}): alter_trans={alter_trans}, all_trans={all_trans}")
            
            searchObj = re.search(r'(\S+);AL:(\S+):(\d+)-(\d+):(\d+):(\d+)-(\d+):(\d+):([+-])', event_info)
            
            if searchObj:
                event_gene_id = searchObj.group(1).split(".")[0]
                chrom_event = searchObj.group(2).lower().replace('chr', '')
                start1, end1 = int(searchObj.group(3)), int(searchObj.group(4))
                mid = int(searchObj.group(5))
                start2, end2 = int(searchObj.group(6)), int(searchObj.group(7))
                end3 = int(searchObj.group(8))
                strand = searchObj.group(9)
                
                if chrom != chrom_event:
                    logging.warning(f"Chromosome mismatch in AL event (line {line_count}): {line} (chrom: {chrom}, chrom_event: {chrom_event})")
                    continue
                
                if strand == "+":
                    for i in alter_trans:
                        for j in all_trans:
                            if j in alter_trans:
                                continue
                            if i in tras and j in tras:
                                al_length = end1 - start1
                                key1 = i + str(start1)
                                key2 = i + str(end1)
                                key3 = j + str(start2)
                                key4 = j + str(end2)
                                
                                logging.info(f"AL event (line {line_count}): checking keys {key1}, {key2}, {key3}, {key4}")
                                
                                if key1 in position and key2 in position and key3 in position and key4 in position:
                                    print(f"AL+\t{chrom}\t{i}\t{j}\t{position[key1]}\t{position[key2]}\t{position[key3]}\t{position[key4]}\t{start1}\t{end1}\t{start2}\t{end2}\t{al_length}", file=output_handle)
                                    event_count += 1
                                else:
                                    logging.warning(f"Skipping AL event due to missing position keys (line {line_count}): {line} (keys: {key1}, {key2}, {key3}, {key4})")
                            else:
                                logging.warning(f"Skipping AL event due to missing transcripts {i} or {j} (line {line_count}): {line}")
                                    
                elif strand == "-":
                    for i in alter_trans:
                        for j in all_trans:
                            if j in alter_trans:
                                continue
                            if i in tras and j in tras:
                                al_length = end1 - start1
                                key1 = i + str(end1)
                                key2 = i + str(start1)
                                key3 = j + str(end2)
                                key4 = j + str(start2)
                                
                                logging.info(f"AL event (line {line_count}): checking keys {key1}, {key2}, {key3}, {key4}")
                                
                                if key1 in position and key2 in position and key3 in position and key4 in position:
                                    print(f"AL-\t{chrom}\t{i}\t{j}\t{position[key1]}\t{position[key2]}\t{position[key3]}\t{position[key4]}\t{start1}\t{end1}\t{start2}\t{end2}\t{al_length}", file=output_handle)
                                    event_count += 1
                                else:
                                    logging.warning(f"Skipping AL event due to missing position keys (line {line_count}): {line} (keys: {key1}, {key2}, {key3}, {key4})")
                            else:
                                logging.warning(f"Skipping AL event due to missing transcripts {i} or {j} (line {line_count}): {line}")
            else:
                logging.warning(f"Could not parse AL event info (line {line_count}): {event_info}")
    
    ioe.close()
    print(f"Processed {event_count} AL events (total lines: {line_count})", file=sys.stderr)

# 执行主要功能
print("\n" + "="*60, file=sys.stderr)
print("Starting to process splicing events...", file=sys.stderr)
print("="*60 + "\n", file=sys.stderr)

a3()
a5()
se()
ri()
mx()
af()
al()

# 关闭输出文件
output_handle.close()

print("\n" + "="*60, file=sys.stderr)
print(f"All done! Results saved to {output_file}", file=sys.stderr)
print("="*60, file=sys.stderr)