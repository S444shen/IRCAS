import re
import sys
from io import StringIO

# ================== 定义全局参数 ==================
nucl = ("A", "T", "C", "G")
stop = ("TAA", "TAG", "TGA")
stop_set = set(stop)
codon = {"GCG": "Ala", "GCA": "Ala", "GCT": "Ala", "GCC": "Ala", "TGT": "Cys", "TGC": "Cys",
         "GAT": "Asp", "GAC": "Asp", "GAG": "Glu", "GAA": "Glu", "TTT": "Phe", "TTC": "Phe",
         "GGG": "Gly", "GGA": "Gly", "GGT": "Gly", "GGC": "Gly", "CAT": "His", "CAC": "His",
         "ATA": "Ile", "ATT": "Ile", "ATC": "Ile", "AAG": "Lys", "AAA": "Lys", "TTG": "Leu",
         "TTA": "Leu", "CTG": "Leu", "CTA": "Leu", "CTT": "Leu", "CTC": "Leu", "ATG": "Met",
         "AAT": "Asn", "AAC": "Asn", "CCG": "Pro", "CCA": "Pro", "CCT": "Pro", "CCC": "Pro",
         "CAG": "Gln", "CAA": "Gln", "AGG": "Arg", "AGA": "Arg", "CGG": "Arg", "CGA": "Arg",
         "CGT": "Arg", "CGC": "Arg", "AGT": "Ser", "AGC": "Ser", "TCG": "Ser", "TCA": "Ser",
         "TCT": "Ser", "TCC": "Ser", "ACG": "Thr", "ACA": "Thr", "ACT": "Thr", "ACC": "Thr",
         "GTG": "Val", "GTA": "Val", "GTT": "Val", "GTC": "Val", "TGG": "Trp", "TAT": "Tyr",
         "TAC": "Tyr", "TGA": "End", "TAG": "End", "TAA": "End"}

# 根据附表5定义的motif（按特征顺序）
# 特征2-16: 下游motif
Dmotifs = [
    "TTCTT", "TAACT", "TCTTT", "TCTGG", "GTAAG", "GTTTT", "GTAAT", 
    "TTCTCT", "TATGT", "TTTCTC", "CTTTT", "TTTAG", "TTTTTC", "TCTTG", "TCTTC"
]

# 特征17-25: 上游motif
Umotifs = [
    "TTCTT", "TCTTT", "CTCTG", "GTAAG", "GTTTT", "AAATT", "TCTCT", "CTTTT", "TTCTC"
]

# ================== 工具函数 ==================
def gc_content(seq):
    """计算序列GC含量"""
    if not seq: 
        return 0.0
    gc = seq.count('G') + seq.count('C')
    return round(gc / len(seq), 4)

def kmer_frequency(seq, kmer):
    """计算k-mer出现频率"""
    k = len(kmer)
    if len(seq) < k:
        return 0.0
    count = seq.count(kmer)
    return round(count / (len(seq) - k + 1), 4)

def distribution_percentile(seq, base, percentile):
    """计算碱基分布分位点（位置/长度）"""
    positions = [i+1 for i, char in enumerate(seq) if char == base]
    if not positions:
        return 0.0
    
    if percentile == 0.01: 
        idx = 0
    elif percentile == 0.25: 
        idx = int(len(positions) * 0.25) - 1
    elif percentile == 0.50: 
        idx = int(len(positions) * 0.50) - 1
    elif percentile == 0.75: 
        idx = int(len(positions) * 0.75) - 1
    elif percentile == 1.00: 
        idx = -1
    else:
        idx = 0
    
    if idx < 0: 
        idx = len(positions) - 1
    if idx >= len(positions): 
        idx = len(positions) - 1
        
    pos = positions[idx]
    return round(pos / len(seq), 4)

def donor_contains(seq, kmer, win_size=5):
    """检查donor区域是否包含k-mer（后5个碱基）"""
    donor_seq = seq[-win_size:] if len(seq) >= win_size else seq
    return 1 if kmer in donor_seq else 0

def acceptor_contains(seq, kmer, win_size=5):
    """检查acceptor区域是否包含k-mer（前5个碱基）"""
    acceptor_seq = seq[:win_size] if len(seq) >= win_size else seq
    return 1 if kmer in acceptor_seq else 0

# ================== 特征计算函数 ==================
def calculate_allseq_features(seq):
    """计算全序列特征（特征26-56）"""
    features = []
    
    # 26. GC含量
    features.append(gc_content(seq))
    
    # 27-29. 终止子数量（TAA, TAG, TGA）
    for stop_codon in stop:
        features.append(seq.count(stop_codon))
    
    # 30-43. k-mer频率（按附表5顺序）
    kmers = ["ATA", "AGA", "T", "TA", "TAA", "TAT", "TT", "TTA", "TTT", 
             "CAG", "CG", "G", "GA", "GTA", "GC", "GGA"]
    for kmer in kmers:
        features.append(kmer_frequency(seq, kmer))
    
    # 44-56. 分布特征
    distributions = [
        ("A", 0.01), ("T", 0.25), ("T", 0.50), ("T", 0.75),
        ("C", 0.25), ("C", 0.50),
        ("G", 0.01), ("G", 0.25), ("G", 0.50), ("G", 0.75), ("G", 1.00)
    ]
    for base, percentile in distributions:
        features.append(distribution_percentile(seq, base, percentile))
    
    return features

def calculate_asseq_features(seq):
    """计算AS区域特征（特征57-164）"""
    features = []
    
    # 57. 长度能否被3整除
    features.append(1 if len(seq) % 3 == 0 else 0)
    
    # 58. GC含量
    features.append(gc_content(seq))
    
    # 59-61. 终止子数量（TAA, TAG, TGA）
    for stop_codon in stop:
        features.append(seq.count(stop_codon))
    
    # 62-136. k-mer频率（按附表5顺序）
    kmers = [
        "A", "AA", "AAA", "AAT", "AAC", "AAG", "AT", "ATA", "ATT", "ATC", "ATG",
        "AC", "ACA", "ACT", "ACG", "AG", "AGA", "AGT", "AGC", "AGG",
        "T", "TA", "TAA", "TAT", "TAC", "TAG", "TT", "TTA", "TTT", "TTC", "TTG",
        "TC", "TCA", "TCT", "TCC", "TCG", "TG", "TGA", "TGT", "TGC", "TGG",
        "C", "CA", "CAA", "CAT", "CAG", "CT", "CTA", "CTT", "CTC", "CTG",
        "CC", "CCA", "CCT", "CG", "CGA",
        "G", "GA", "GAA", "GAT", "GAC", "GAG", "GT", "GTA", "GTT", "GTC", "GTG",
        "GC", "GCA", "GCT", "GG", "GGA", "GGT", "GGC", "GGG"
    ]
    for kmer in kmers:
        features.append(kmer_frequency(seq, kmer))
    
    # 137-156. 分布特征（分位点计算）
    distributions = [
        ("A", 0.01), ("A", 0.25), ("A", 0.50), ("A", 0.75), ("A", 1.00),
        ("T", 0.01), ("T", 0.25), ("T", 0.50), ("T", 0.75), ("T", 1.00),
        ("C", 0.01), ("C", 0.25), ("C", 0.50), ("C", 0.75), ("C", 1.00),
        ("G", 0.01), ("G", 0.25), ("G", 0.50), ("G", 0.75), ("G", 1.00)
    ]
    for base, percentile in distributions:
        features.append(distribution_percentile(seq, base, percentile))
    
    # 157-164. 剪接位点特征
    for kmer in ["GT", "GC", "AT", "AG"]:
        features.append(donor_contains(seq, kmer))  # donor
    for kmer in ["GT", "GC", "AT", "AG"]:
        features.append(acceptor_contains(seq, kmer))  # acceptor
    
    return features

def calculate_up_features(seq):
    """计算上游区域特征（特征165-169）"""
    features = []
    
    # 165-166. k-mer频率
    kmers = ["TC", "C"]
    for kmer in kmers:
        features.append(kmer_frequency(seq, kmer))
    
    # 167. 分布特征
    features.append(distribution_percentile(seq, "G", 1.00))
    
    # 168-169. 剪接位点
    features.append(donor_contains(seq, "GT"))
    features.append(donor_contains(seq, "AG"))
    
    return features

def calculate_down_features(seq):
    """计算下游区域特征（特征170-174）"""
    features = []
    
    # 170-172. k-mer频率
    kmers = ["TC", "TCT", "CT"]
    for kmer in kmers:
        features.append(kmer_frequency(seq, kmer))
    
    # 173. 分布特征
    features.append(distribution_percentile(seq, "G", 1.00))
    
    # 174. 剪接位点
    features.append(donor_contains(seq, "AG"))
    
    return features

def calculate_up30as30_features(seq):
    """计算up30as30组合区域特征（特征175-204）"""
    features = []
    
    # 175. 长度能否被3整除
    features.append(1 if len(seq) % 3 == 0 else 0)
    
    # 176. GC含量
    features.append(gc_content(seq))
    
    # 177. TAA终止子数量
    features.append(seq.count("TAA"))
    
    # 178-190. k-mer频率
    kmers = ["AG", "AGG", "TA", "TAA", "TTT", "CAG", "G", "GA", "GAT", 
             "GTA", "GGA", "GGT", "GGC"]
    for kmer in kmers:
        features.append(kmer_frequency(seq, kmer))
    
    # 191-200. 分布特征
    distributions = [
        ("A", 1.00),
        ("T", 0.50), ("T", 0.75), ("T", 1.00),
        ("C", 0.50), ("C", 0.75),
        ("G", 0.25), ("G", 0.50), ("G", 0.75), ("G", 1.00)
    ]
    for base, percentile in distributions:
        features.append(distribution_percentile(seq, base, percentile))
    
    # 201-204. 剪接位点
    features.append(donor_contains(seq, "GT"))
    features.append(donor_contains(seq, "AT"))
    features.append(donor_contains(seq, "AG"))
    features.append(acceptor_contains(seq, "AG"))
    
    return features

def calculate_up30down30_features(seq):
    """计算up30down30组合区域特征（特征205-206）"""
    features = []
    
    # 205. GC含量
    features.append(gc_content(seq))
    
    # 206. 长度能否被3整除
    features.append(1 if len(seq) % 3 == 0 else 0)
    
    return features

def calculate_as30down30_features(seq):
    """计算as30down30组合区域特征（特征207-229）"""
    features = []
    
    # 207. 长度能否被3整除
    features.append(1 if len(seq) % 3 == 0 else 0)
    
    # 208. GC含量
    features.append(gc_content(seq))
    
    # 209-221. k-mer频率
    kmers = ["A", "T", "TT", "TTC", "TC", "TCT", "TCC", "CT", "CTT", 
             "CTC", "CG", "CGA", "GGG"]
    for kmer in kmers:
        features.append(kmer_frequency(seq, kmer))
    
    # 222-228. 分布特征
    distributions = [
        ("A", 0.01), ("A", 0.25), ("A", 0.50),
        ("T", 0.01), ("T", 0.75),
        ("C", 0.01), ("C", 0.25)
    ]
    for base, percentile in distributions:
        features.append(distribution_percentile(seq, base, percentile))
    
    # 229. acceptor位点GT
    features.append(acceptor_contains(seq, "GT"))
    
    return features

def calculate_up50as50_features(seq):
    """计算up50as50组合区域特征（特征230-261）"""
    features = []
    
    # 230. 长度能否被3整除
    features.append(1 if len(seq) % 3 == 0 else 0)
    
    # 231. GC含量
    features.append(gc_content(seq))
    
    # 232. TGA终止子数量
    features.append(seq.count("TGA"))
    
    # 233-250. k-mer频率
    kmers = ["A", "AA", "AAA", "AT", "AGA", "AGG", "TA", "TAA", "CAG", 
             "CC", "CCT", "CG", "G", "GT", "GTA", "GTG", "GG", "GGT"]
    for kmer in kmers:
        features.append(kmer_frequency(seq, kmer))
    
    # 251-259. 分布特征
    distributions = [
        ("A", 0.50), ("A", 0.75), ("A", 1.00),
        ("T", 0.50), ("T", 0.75), ("T", 1.00),
        ("G", 0.50), ("G", 0.75), ("G", 1.00)
    ]
    for base, percentile in distributions:
        features.append(distribution_percentile(seq, base, percentile))
    
    # 260-261. 剪接位点
    features.append(donor_contains(seq, "GT"))
    features.append(donor_contains(seq, "AG"))
    
    return features

def calculate_up50down50_features(seq):
    """计算up50down50组合区域特征（特征262-266）"""
    features = []
    
    # 262. GC含量
    features.append(gc_content(seq))
    
    # 263-265. k-mer频率
    kmers = ["TC", "TCT", "CT"]
    for kmer in kmers:
        features.append(kmer_frequency(seq, kmer))
    
    # 266. donor位点AG
    features.append(donor_contains(seq, "AG"))
    
    return features

def calculate_as50down50_features(seq):
    """计算as50down50组合区域特征（特征267-297）"""
    features = []
    
    # 267. 长度能否被3整除
    features.append(1 if len(seq) % 3 == 0 else 0)
    
    # 268. GC含量
    features.append(gc_content(seq))
    
    # 269-270. 终止子数量（TAA, TGA）
    features.append(seq.count("TAA"))
    features.append(seq.count("TGA"))
    
    # 271-286. k-mer频率
    kmers = ["A", "AAA", "AAG", "T", "TT", "TTT", "TTC", "TC", "TCG", 
             "TGT", "C", "CAG", "CTC", "CG", "CGA", "G"]
    for kmer in kmers:
        features.append(kmer_frequency(seq, kmer))
    
    # 287-295. 分布特征
    distributions = [
        ("A", 0.01),
        ("T", 0.01), ("T", 0.50),
        ("C", 0.01),
        ("G", 0.01), ("G", 0.25), ("G", 0.50), ("G", 0.75), ("G", 1.00)
    ]
    for base, percentile in distributions:
        features.append(distribution_percentile(seq, base, percentile))
    
    # 296-297. 剪接位点
    features.append(acceptor_contains(seq, "GT"))
    features.append(acceptor_contains(seq, "AG"))
    
    return features

# ================== 主流程 ==================
def main():
    """主函数：处理序列文件并提取特征"""
    input_file = "ricesequence1.csv"
    output_file = "ricesequencefeature.txt"
    
    try:
        with open(input_file, "r") as fin, open(output_file, "w") as fout:
            line_count = 0
            error_count = 0
            
            for line in fin:
                line_count += 1
                
                # 解析数据
                fields = line.strip().replace("a", "A").replace("g", "G")
                fields = fields.replace("c", "C").replace("t", "T").split(",")
                
                if len(fields) < 4:
                    print(f"警告：第{line_count}行格式错误，跳过: {line.strip()[:50]}...", file=sys.stderr)
                    error_count += 1
                    continue
                
                seq_name, up_seq, as_seq, down_seq = fields[0], fields[1], fields[2], fields[3]
                all_seq = up_seq + as_seq + down_seq
                features_list = []
                
                # === 特征1: AS区域长度 ===
                features_list.append(len(as_seq))
                
                # === 特征2-16: 下游motif (Dmotifs) ===
                for motif in Dmotifs:
                    features_list.append(1 if motif in down_seq else 0)
                
                # === 特征17-25: 上游motif (Umotifs) ===
                for motif in Umotifs:
                    features_list.append(1 if motif in up_seq else 0)
                
                # === 特征26-56: 全序列特征 ===
                features_list.extend(calculate_allseq_features(all_seq))
                
                # === 特征57-164: AS区域特征 ===
                features_list.extend(calculate_asseq_features(as_seq))
                
                # === 特征165-169: 上游特征 ===
                features_list.extend(calculate_up_features(up_seq))
                
                # === 特征170-174: 下游特征 ===
                features_list.extend(calculate_down_features(down_seq))
                
                # === 特征175-204: up30as30组合区域特征 ===
                up30 = up_seq[-30:] if len(up_seq) >= 30 else up_seq
                as30_start = as_seq[:30] if len(as_seq) >= 30 else as_seq
                up30as30 = up30 + as30_start
                features_list.extend(calculate_up30as30_features(up30as30))
                
                # === 特征205-206: up30down30区域 ===
                down30 = down_seq[:30] if len(down_seq) >= 30 else down_seq
                up30down30 = up30 + down30
                features_list.extend(calculate_up30down30_features(up30down30))
                
                # === 特征207-229: as30down30区域 ===
                as30_end = as_seq[-30:] if len(as_seq) >= 30 else as_seq
                as30down30 = as30_end + down30
                features_list.extend(calculate_as30down30_features(as30down30))
                
                # === 特征230-261: up50as50区域 ===
                up50 = up_seq[-50:] if len(up_seq) >= 50 else up_seq
                as50_start = as_seq[:50] if len(as_seq) >= 50 else as_seq
                up50as50 = up50 + as50_start
                features_list.extend(calculate_up50as50_features(up50as50))
                
                # === 特征262-266: up50down50区域 ===
                down50 = down_seq[:50] if len(down_seq) >= 50 else down_seq
                up50down50 = up50 + down50
                features_list.extend(calculate_up50down50_features(up50down50))
                
                # === 特征267-297: as50down50区域 ===
                as50_end = as_seq[-50:] if len(as_seq) >= 50 else as_seq
                as50down50 = as50_end + down50
                features_list.extend(calculate_as50down50_features(as50down50))
                
                # 验证特征数量
                if len(features_list) != 297:
                    print(f"错误：第{line_count}行特征数量不正确: {len(features_list)}/297", file=sys.stderr)
                    error_count += 1
                    continue
                
                # 写入结果（共297个特征）
                str_features = ",".join(map(str, features_list))
                fout.write(f"{line.strip()},{str_features}\n")
                
                # 每处理1000行输出进度
                if line_count % 1000 == 0:
                    print(f"Processed {line_count} lines...")
            
            print(f"\nFeature extraction completed!") 
            print(f"- Total processed lines: {line_count}")
            print(f"- Successfully processed: {line_count - error_count}")
            print(f"- Error lines: {error_count}")
            print(f"- Number of extracted features: 297")
            print(f"- Output file: {output_file}")
            
    except FileNotFoundError:
        print(f"错误：找不到输入文件 {input_file}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"错误：处理文件时发生异常: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()