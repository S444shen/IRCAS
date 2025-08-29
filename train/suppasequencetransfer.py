#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import sys
import os
from Bio import SeqIO
from Bio.Seq import reverse_complement

# 检查命令行参数
if len(sys.argv) < 5:
    print("Usage: python suppasequencetransfer.py <genome_fasta> <cdna_fasta> <gtf_file> <output_file>")
    print("Example: python suppasequencetransfer.py rice_all.chrs.fa rice_all.transcripts.fa rice_all.gtf ricesequence.tsv")
    sys.exit(1)

# 从命令行参数获取文件路径
genome_file = sys.argv[1]
cdna_file = sys.argv[2] 
gtf_file = sys.argv[3]
output_file = sys.argv[4]

# 打开输出文件
output_handle = open(output_file, 'w')
print("event_type\tupstream\tevent\tdownstream\ttranscripts", file=output_handle)

print(f"Loading genome from {genome_file}...", file=sys.stderr)
seqs={}
for seq_record in SeqIO.parse(genome_file, "fasta"):
    seqs[seq_record.id] = str(seq_record.seq).upper()
print(f"Loaded {len(seqs)} chromosomes/contigs", file=sys.stderr)

position={}
tras={}

print(f"Loading transcripts from {cdna_file}...", file=sys.stderr)
for seq_record in SeqIO.parse(cdna_file, "fasta"):
    tras[seq_record.id] = str(seq_record.seq).upper()
print(f"Loaded {len(tras)} transcripts", file=sys.stderr)

zhengmin={}
zhengmax={}
fanmin={}
fanmax={}

print(f"Processing GTF file {gtf_file}...", file=sys.stderr)
ioe = open(gtf_file,"r")
trans = "trans"
pos = 1
line_count = 0
exon_count = 0

# Debugging: Print first few lines of GTF to inspect format
print("Inspecting first 5 non-comment lines of GTF file...", file=sys.stderr)
non_comment_lines = 0
for line in ioe:
    if not line.startswith('#'):
        print(f"GTF line: {line.strip()}", file=sys.stderr)
        non_comment_lines += 1
        if non_comment_lines >= 5:
            break
ioe.seek(0)  # Reset file pointer to start

while 1:
    lines = ioe.readlines(100000)
    if not lines:
        break
    for line in lines:
        line = line.strip('\n')
        line_count += 1
        
        if line.startswith('#'):
            continue
            
        if '\texon\t' not in line:
            continue
            
        # More flexible regex to handle various GTF formats
        searchObj = re.search(r'^(\S+)\t([^\t]*)\texon\t(\d+)\t(\d+)\t([^\t]*)\t([+-])\t([^\t]*)\t(.*)', line)
        
        if searchObj:
            chrom = searchObj.group(1)
            start = searchObj.group(3)
            end = searchObj.group(4)
            strand = searchObj.group(6)
            attributes = searchObj.group(8)
            
            # Extract gene_id and transcript_id from attributes
            gene_id_match = re.search(r'gene_id\s+["\']?([^"\';]+)["\']?', attributes)
            transcript_id_match = re.search(r'transcript_id\s+["\']?([^"\';]+)["\']?', attributes)
            
            if gene_id_match and transcript_id_match:
                gene_id = gene_id_match.group(1)
                transcript_id = transcript_id_match.group(1)
                exon_count += 1
                
                if exon_count % 10000 == 0:
                    print(f"Processed {exon_count} exons... Current gene: {gene_id}, transcript: {transcript_id}", file=sys.stderr)
                
                if transcript_id != trans:
                    trans = transcript_id
                    pos = 1
                    
                    if strand == "+":
                        position[trans + start] = pos
                        pos = pos + int(end) - int(start)
                        position[trans + end] = pos
                        
                        if gene_id in zhengmin:
                            zhengmin[gene_id] = min(zhengmin[gene_id], int(end), int(start))
                            zhengmax[gene_id] = max(zhengmax[gene_id], int(end), int(start))
                        else:
                            zhengmin[gene_id] = min(int(end), int(start))
                            zhengmax[gene_id] = max(int(end), int(start))
                            
                    elif strand == "-":
                        pos = pos + int(end) - int(start)
                        
                        if gene_id in fanmin:
                            fanmin[gene_id] = min(fanmin[gene_id], int(end), int(start))
                            fanmax[gene_id] = max(fanmax[gene_id], int(end), int(start))
                        else:
                            fanmin[gene_id] = min(int(end), int(start))
                            fanmax[gene_id] = max(int(end), int(start))
                            
                    pos = pos + 1
                else:
                    if strand == "+":
                        position[trans + start] = pos
                        pos = pos + int(end) - int(start)
                        position[trans + end] = pos
                        
                        if gene_id in zhengmin:
                            zhengmin[gene_id] = min(zhengmin[gene_id], int(end), int(start))
                            zhengmax[gene_id] = max(zhengmax[gene_id], int(end), int(start))
                        else:
                            zhengmin[gene_id] = min(int(end), int(start))
                            zhengmax[gene_id] = max(int(end), int(start))
                        
                    elif strand == "-":
                        pos = pos + int(end) - int(start)
                        
                        if gene_id in fanmin:
                            fanmin[gene_id] = min(fanmin[gene_id], int(end), int(start))
                            fanmax[gene_id] = max(fanmax[gene_id], int(end), int(start))
                        else:
                            fanmin[gene_id] = min(int(end), int(start))
                            fanmax[gene_id] = max(int(end), int(start))
                        
                    pos = pos + 1
            else:
                print(f"Warning: Could not parse gene_id or transcript_id from line: {line}", file=sys.stderr)

ioe.close()
print(f"Finished processing GTF. Total lines: {line_count}, Total exons: {exon_count}", file=sys.stderr)
print(f"Positive strand genes: {len(zhengmin)}", file=sys.stderr)
print(f"Negative strand genes: {len(fanmin)}", file=sys.stderr)

def a3():
    """Alternative 3' splice site"""
    print("Processing A3 events...", file=sys.stderr)
    if not os.path.exists("./events_A3_strict.ioe"):
        print("Warning: events_A3_strict.ioe not found, skipping A3 events", file=sys.stderr)
        return
        
    with open("./events_A3_strict.ioe", "r") as ioe:
        event_count = 0
        for line in ioe:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split("\t")
            if len(parts) < 5:
                print(f"Warning: Malformed A3 line: {line}", file=sys.stderr)
                continue
                
            gene_id = parts[1]
            event_info = parts[2]
            transcript_ids = parts[4].split(",")

            searchObj = re.search(
                r'(\S+);A3:(\S+):(\d+)-(\d+):(\d+)-(\d+):([+-])',
                event_info
            )
            if searchObj:
                event_gene_id = searchObj.group(1)
                chrom = searchObj.group(2)
                coord1, coord2 = int(searchObj.group(3)), int(searchObj.group(4))
                coord3, coord4 = int(searchObj.group(5)), int(searchObj.group(6))
                strand = searchObj.group(7)

                full_sequence = ";".join([tras.get(tid, "N/A") for tid in transcript_ids])

                if strand == "+" and gene_id in zhengmin:
                    start = max(zhengmin[gene_id], coord1 - 100)
                    end = min(zhengmax[gene_id], coord4 + 99)
                    up = seqs.get(chrom, "")[start:coord1]
                    a3 = seqs.get(chrom, "")[coord2 - 1:coord4 - 1]
                    down = seqs.get(chrom, "")[coord4 - 1:end]
                    print(f"A3\t{up}\t{a3}\t{down}\t{full_sequence}", file=output_handle)
                    event_count += 1
                elif strand == "-" and gene_id in fanmin:
                    start = max(fanmin[gene_id], coord3 - 100)
                    end = min(fanmax[gene_id], coord2 + 99)
                    up = reverse_complement(seqs.get(chrom, "")[coord2 - 1:end])
                    a3 = reverse_complement(seqs.get(chrom, "")[coord3:coord1])
                    down = reverse_complement(seqs.get(chrom, "")[start:coord3])
                    print(f"A3\t{up}\t{a3}\t{down}\t{full_sequence}", file=output_handle)
                    event_count += 1
            else:
                print(f"Warning: Could not parse A3 event info: {event_info}", file=sys.stderr)
                    
        print(f"Processed {event_count} A3 events", file=sys.stderr)

def a5():
    """Alternative 5' splice site"""
    print("Processing A5 events...", file=sys.stderr)
    if not os.path.exists("./events_A5_strict.ioe"):
        print("Warning: events_A5_strict.ioe not found, skipping A5 events", file=sys.stderr)
        return
        
    with open("./events_A5_strict.ioe", "r") as ioe:
        event_count = 0
        for line in ioe:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split("\t")
            if len(parts) >= 2:
                transcript_ids = parts[-2].split(",")
            else:
                continue

            searchObj = re.search(
                r'(\S+);A5:(\S+):(\d+)-(\d+):(\d+)-(\d+):([+-])',
                line
            )
            if searchObj:
                gene_id = searchObj.group(1)
                chrom = searchObj.group(2)
                coord1, coord2 = int(searchObj.group(3)), int(searchObj.group(4))
                coord3, coord4 = int(searchObj.group(5)), int(searchObj.group(6))
                strand = searchObj.group(7)

                full_sequence = ";".join([tras.get(tid, "N/A") for tid in transcript_ids])

                if strand == "+" and gene_id in zhengmin:
                    start = max(zhengmin[gene_id], coord3 - 100)
                    end = min(zhengmax[gene_id], coord2 + 99)
                    up = seqs.get(chrom, "")[start:coord3]
                    a5 = seqs.get(chrom, "")[coord3:coord1]
                    down = seqs.get(chrom, "")[coord2 - 1:end]
                    print(f"A5\t{up}\t{a5}\t{down}\t{full_sequence}", file=output_handle)
                    event_count += 1
                elif strand == "-" and gene_id in fanmin:
                    start = max(fanmin[gene_id], coord1 - 100)
                    end = min(fanmax[gene_id], coord4 + 99)
                    up = reverse_complement(seqs.get(chrom, "")[start:coord1])
                    a5 = reverse_complement(seqs.get(chrom, "")[coord2 - 1:coord4 - 1])
                    down = reverse_complement(seqs.get(chrom, "")[coord4 - 1:end])
                    print(f"A5\t{up}\t{a5}\t{down}\t{full_sequence}", file=output_handle)
                    event_count += 1
                    
        print(f"Processed {event_count} A5 events", file=sys.stderr)

def se():
    """Skipped Exon"""
    print("Processing SE events...", file=sys.stderr)
    if not os.path.exists("./events_SE_strict.ioe"):
        print("Warning: events_SE_strict.ioe not found, skipping SE events", file=sys.stderr)
        return
        
    with open("./events_SE_strict.ioe", "r") as ioe:
        event_count = 0
        for line in ioe:
            line = line.strip('\n')
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) >= 2:
                transcript_ids = parts[-2].split(",")
            else:
                continue

            searchObj = re.search(r'(\S+);SE:(\S+):(\d+)-(\d+):(\d+)-(\d+):([+-])', line)
            
            if searchObj:
                gene_id = searchObj.group(1)
                chrom = searchObj.group(2)
                coord1, coord2 = int(searchObj.group(3)), int(searchObj.group(4))
                coord3, coord4 = int(searchObj.group(5)), int(searchObj.group(6))
                strand = searchObj.group(7)

                full_sequence = ";".join([tras.get(tid, "N/A") for tid in transcript_ids])

                if strand == "+" and gene_id in zhengmin:
                    start = max(zhengmin[gene_id], coord1 - 100)
                    end = min(zhengmax[gene_id], coord4 + 99)
                    up = seqs.get(chrom, "")[start:coord1]
                    se = seqs.get(chrom, "")[coord2 - 1:coord3]
                    down = seqs.get(chrom, "")[coord4 - 1:end]
                    print(f"SE\t{up}\t{se}\t{down}\t{full_sequence}", file=output_handle)
                    event_count += 1
                elif strand == "-" and gene_id in fanmin:
                    start = max(fanmin[gene_id], coord1 - 100)
                    end = min(fanmax[gene_id], coord4 + 99)
                    down = reverse_complement(seqs.get(chrom, "")[start:coord1])
                    se = reverse_complement(seqs.get(chrom, "")[coord2 - 1:coord3])
                    up = reverse_complement(seqs.get(chrom, "")[coord4 - 1:end])
                    print(f"SE\t{up}\t{se}\t{down}\t{full_sequence}", file=output_handle)
                    event_count += 1
                    
        print(f"Processed {event_count} SE events", file=sys.stderr)

def ri():
    """Retained Intron"""
    print("Processing RI events...", file=sys.stderr)
    if not os.path.exists("./events_RI_strict.ioe"):
        print("Warning: events_RI_strict.ioe not found, skipping RI events", file=sys.stderr)
        return
        
    with open("./events_RI_strict.ioe", "r") as ioe:
        event_count = 0
        for line in ioe:
            line = line.strip('\n')
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) >= 2:
                transcript_ids = parts[-2].split(",")
            else:
                continue

            searchObj = re.search(r'(\S+);RI:(\S+):(\d+):(\d+)-(\d+):(\d+):([+-])', line)
            
            if searchObj:
                gene_id = searchObj.group(1)
                chrom = searchObj.group(2)
                coord1 = int(searchObj.group(3))
                coord2 = int(searchObj.group(4))
                coord3 = int(searchObj.group(5))
                coord4 = int(searchObj.group(6))
                strand = searchObj.group(7)

                full_sequence = ";".join([tras.get(tid, "N/A") for tid in transcript_ids])

                if strand == "+" and gene_id in zhengmin:
                    start = max(zhengmin[gene_id], coord2 - 100)
                    end = min(zhengmax[gene_id], coord3 + 99)
                    up = seqs.get(chrom, "")[start:coord2]
                    ri = seqs.get(chrom, "")[coord2:coord3 - 1]
                    down = seqs.get(chrom, "")[coord3 - 1:end]
                    print(f"RI\t{up}\t{ri}\t{down}\t{full_sequence}", file=output_handle)
                    event_count += 1
                elif strand == "-" and gene_id in fanmin:
                    start = max(fanmin[gene_id], coord2 - 100)
                    end = min(fanmax[gene_id], coord3 + 99)
                    down = reverse_complement(seqs.get(chrom, "")[start:coord2])
                    ri = reverse_complement(seqs.get(chrom, "")[coord2:coord3 - 1])
                    up = reverse_complement(seqs.get(chrom, "")[coord3 - 1:end])
                    print(f"RI\t{up}\t{ri}\t{down}\t{full_sequence}", file=output_handle)
                    event_count += 1
                    
        print(f"Processed {event_count} RI events", file=sys.stderr)

def mx():
    """Mutually Exclusive Exons"""
    print("Processing MX events...", file=sys.stderr)
    if not os.path.exists("./events_MX_strict.ioe"):
        print("Warning: events_MX_strict.ioe not found, skipping MX events", file=sys.stderr)
        return
        
    with open("./events_MX_strict.ioe", "r") as ioe:
        event_count = 0
        for line in ioe:
            line = line.strip('\n')
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) >= 2:
                transcript_ids = parts[-2].split(",")
            else:
                continue

            searchObj = re.search(r'(\S+);MX:(\S+):(\d+)-(\d+):(\d+)-(\d+):(\d+)-(\d+):(\d+)-(\d+):([+-])', line)
            
            if searchObj:
                gene_id = searchObj.group(1)
                chrom = searchObj.group(2)
                coord1, coord2 = int(searchObj.group(3)), int(searchObj.group(4))
                coord3, coord4 = int(searchObj.group(5)), int(searchObj.group(6))
                coord5, coord6 = int(searchObj.group(7)), int(searchObj.group(8))
                coord7, coord8 = int(searchObj.group(9)), int(searchObj.group(10))
                strand = searchObj.group(11)

                full_sequence = ";".join([tras.get(tid, "N/A") for tid in transcript_ids])

                if strand == "+" and gene_id in zhengmin:
                    start = max(zhengmin[gene_id], coord1 - 100)
                    end = min(zhengmax[gene_id], coord8 + 99)
                    up = seqs.get(chrom, "")[start:coord1]
                    mx1 = seqs.get(chrom, "")[coord2 - 1:coord3]
                    mx2 = seqs.get(chrom, "")[coord6 - 1:coord7]
                    mx_combined = mx1 + "|" + mx2
                    down = seqs.get(chrom, "")[coord8 - 1:end]
                    print(f"MX\t{up}\t{mx_combined}\t{down}\t{full_sequence}", file=output_handle)
                    event_count += 1
                elif strand == "-" and gene_id in fanmin:
                    start = max(fanmin[gene_id], coord1 - 100)
                    end = min(fanmax[gene_id], coord8 + 99)
                    down = reverse_complement(seqs.get(chrom, "")[start:coord1])
                    mx1 = reverse_complement(seqs.get(chrom, "")[coord2 - 1:coord3])
                    mx2 = reverse_complement(seqs.get(chrom, "")[coord6 - 1:coord7])
                    mx_combined = mx1 + "|" + mx2
                    up = reverse_complement(seqs.get(chrom, "")[coord8 - 1:end])
                    print(f"MX\t{up}\t{mx_combined}\t{down}\t{full_sequence}", file=output_handle)
                    event_count += 1
                    
        print(f"Processed {event_count} MX events", file=sys.stderr)

def af():
    """Alternative First Exon"""
    print("Processing AF events...", file=sys.stderr)
    if not os.path.exists("./events_AF_strict.ioe"):
        print("Warning: events_AF_strict.ioe not found, skipping AF events", file=sys.stderr)
        return
        
    with open("./events_AF_strict.ioe", "r") as ioe:
        event_count = 0
        for line in ioe:
            line = line.strip('\n')
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) >= 2:
                transcript_ids = parts[-2].split(",")
            else:
                continue

            if ':+' in line or ':-' in line:
                searchObj = re.search(r'(\S+);AF:(\S+):(\d+):(\d+)-(\d+):(\d+):(\d+)-(\d+):([+-])', line)
            else:
                searchObj = re.search(r'(\S+);AF:(\S+):(\d+)-(\d+):(\d+):(\d+)-(\d+):(\d+):([+-])', line)
            
            if searchObj:
                gene_id = searchObj.group(1)
                chrom = searchObj.group(2)
                
                if len(searchObj.groups()) == 9:
                    coord1 = int(searchObj.group(3))
                    coord2, coord3 = int(searchObj.group(4)), int(searchObj.group(5))
                    coord4 = int(searchObj.group(6))
                    coord5, coord6 = int(searchObj.group(7)), int(searchObj.group(8))
                    strand = searchObj.group(9)
                else:
                    coord1, coord2 = int(searchObj.group(3)), int(searchObj.group(4))
                    coord3 = int(searchObj.group(5))
                    coord4, coord5 = int(searchObj.group(6)), int(searchObj.group(7))
                    coord6 = int(searchObj.group(8))
                    strand = searchObj.group(9)

                full_sequence = ";".join([tras.get(tid, "N/A") for tid in transcript_ids])

                if strand == "+" and gene_id in zhengmin:
                    af1 = seqs.get(chrom, "")[coord1:coord2]
                    af2 = seqs.get(chrom, "")[coord4:coord5]
                    af_combined = af1 + "|" + af2
                    down = seqs.get(chrom, "")[coord3:coord3+50]
                    print(f"AF\t\t{af_combined}\t{down}\t{full_sequence}", file=output_handle)
                    event_count += 1
                elif strand == "-" and gene_id in fanmin:
                    af1 = reverse_complement(seqs.get(chrom, "")[coord2:coord3])
                    af2 = reverse_complement(seqs.get(chrom, "")[coord5:coord6])
                    af_combined = af1 + "|" + af2
                    down = reverse_complement(seqs.get(chrom, "")[coord1-50:coord1])
                    print(f"AF\t\t{af_combined}\t{down}\t{full_sequence}", file=output_handle)
                    event_count += 1
                    
        print(f"Processed {event_count} AF events", file=sys.stderr)

def al():
    """Alternative Last Exon"""
    print("Processing AL events...", file=sys.stderr)
    if not os.path.exists("./events_AL_strict.ioe"):
        print("Warning: events_AL_strict.ioe not found, skipping AL events", file=sys.stderr)
        return
        
    with open("./events_AL_strict.ioe", "r") as ioe:
        event_count = 0
        for line in ioe:
            line = line.strip('\n')
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) >= 2:
                transcript_ids = parts[-2].split(",")
            else:
                continue

            if ':+' in line or ':-' in line:
                searchObj = re.search(r'(\S+);AL:(\S+):(\d+)-(\d+):(\d+):(\d+)-(\d+):(\d+):([+-])', line)
            else:
                searchObj = re.search(r'(\S+);AL:(\S+):(\d+):(\d+)-(\d+):(\d+):(\d+)-(\d+):([+-])', line)
            
            if searchObj:
                gene_id = searchObj.group(1)
                chrom = searchObj.group(2)
                
                if len(searchObj.groups()) == 9:
                    coord1, coord2 = int(searchObj.group(3)), int(searchObj.group(4))
                    coord3 = int(searchObj.group(5))
                    coord4, coord5 = int(searchObj.group(6)), int(searchObj.group(7))
                    coord6 = int(searchObj.group(8))
                    strand = searchObj.group(9)
                else:
                    coord1 = int(searchObj.group(3))
                    coord2, coord3 = int(searchObj.group(4)), int(searchObj.group(5))
                    coord4 = int(searchObj.group(6))
                    coord5, coord6 = int(searchObj.group(7)), int(searchObj.group(8))
                    strand = searchObj.group(9)

                full_sequence = ";".join([tras.get(tid, "N/A") for tid in transcript_ids])

                if strand == "+" and gene_id in zhengmin:
                    al1 = seqs.get(chrom, "")[coord2:coord3]
                    al2 = seqs.get(chrom, "")[coord5:coord6]
                    al_combined = al1 + "|" + al2
                    up = seqs.get(chrom, "")[coord4-50:coord4]
                    print(f"AL\t{up}\t{al_combined}\t\t{full_sequence}", file=output_handle)
                    event_count += 1
                elif strand == "-" and gene_id in fanmin:
                    al1 = reverse_complement(seqs.get(chrom, "")[coord1:coord2])
                    al2 = reverse_complement(seqs.get(chrom, "")[coord4:coord5])
                    al_combined = al1 + "|" + al2
                    up = reverse_complement(seqs.get(chrom, "")[coord3:coord3+50])
                    print(f"AL\t{up}\t{al_combined}\t\t{full_sequence}", file=output_handle)
                    event_count += 1
                    
        print(f"Processed {event_count} AL events", file=sys.stderr)

print("\n" + "="*60, file=sys.stderr)
print("Starting to process all splicing events...", file=sys.stderr)
print("="*60 + "\n", file=sys.stderr)

a3()
a5()
se()
ri()
mx()
af()
al()

output_handle.close()

print("\n" + "="*60, file=sys.stderr)
print(f"All done! Results saved to {output_file}", file=sys.stderr)
print("="*60, file=sys.stderr)