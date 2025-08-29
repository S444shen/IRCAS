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

# 按表格定义的特定motif（有序）
Dmotifs = [
    "AAAGA", "TTCTT", "GTAGG", "TGAGG", "TCTTT", "TAAGT", "GTGAG", "GTAAG",
    "GTCTG", "GTTTT", "TTCTCT", "TAAGG", "TCCTTT", "TGTCT", "TTTCTC", "CTTTT",
    "TGAGT", "TAGGT", "CTTTA", "TTTAG", "TGCTT", "GTGGGT", "TCTCC", "TTTTTC"
]

Umotifs = [
    "TTCTT", "TCTTT", "TAAGT", "CTCTG", "TTTTCC", "GTGAG", "GTAAG", "CTTCT",
    "GTTTT", "CCTCT", "TCTCT", "TGTCT", "TTCCTT", "CTTTT", "TGAGT", "CCCCAG",
    "TTTAG", "TTCTC", "TGCTT", "TCTGC", "CTGAA", "GTAGGT", "CTAAA", "TGTGTC"
]

# ================== 工具函数 ==================
def gc_content(seq):
    """计算序列GC含量"""
    if not seq: 
        return 0.0
    gc = seq.count('G') + seq.count('C')
    return round(gc / len(seq), 2)

def kmer_frequency(seq, kmer):
    """计算k-mer出现频率"""
    k = len(kmer)
    if len(seq) < k:
        return 0.0
    count = seq.count(kmer)
    return round(count / (len(seq) - k + 1), 2)

def distribution_percentile(seq, base, percentile):
    """计算碱基分布分位点（位置/长度）"""
    positions = [i+1 for i, char in enumerate(seq) if char == base]
    if not positions:
        return 0.0
    
    if percentile == 0.01: idx = 0
    elif percentile == 0.25: idx = int(len(positions) * 0.25) - 1
    elif percentile == 0.50: idx = int(len(positions) * 0.50) - 1
    elif percentile == 0.75: idx = int(len(positions) * 0.75) - 1
    elif percentile == 1.00: idx = -1
    
    if idx < 0: idx = 0
    if idx >= len(positions): idx = len(positions) - 1
        
    pos = positions[idx]
    return round(pos / len(seq), 2)

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
    """计算全序列特征（特征45-80）"""
    features = []
    # 45. GC含量
    features.append(gc_content(seq))
    # 46-48. 终止子数量
    for stop_codon in stop:
        features.append(seq.count(stop_codon))
    # 49-77. k-mer频率（按表格顺序）
    kmers = ["A", "AA", "AAA", "AT", "AC", "ACA", "AGG", "TT", "TTT", "C",
             "CA", "CAA", "CC", "CCT", "CCC", "CCG", "CG", "CGC", "CGG",
             "G", "GA", "GAA", "GTA", "GC", "GCG", "GG", "GGT", "GGC", "GGG"]
    for kmer in kmers:
        features.append(kmer_frequency(seq, kmer))
    # 78-79. 分布特征
    features.append(distribution_percentile(seq, "A", 0.50))  # 50%A
    features.append(distribution_percentile(seq, "G", 1.00))  # 100%G
    # 80. donor位点AG
    features.append(donor_contains(seq, "AG"))
    return features

def calculate_asseq_features(seq):
    """计算AS区域特征（特征81-195）"""
    features = []
    # 81. GC含量
    features.append(gc_content(seq))
    # 82-84. 终止子数量
    for stop_codon in stop:
        features.append(seq.count(stop_codon))
    # 85-166. k-mer频率（按表格顺序）
    kmers = ["A", "AA", "AAA", "AAT", "AAC", "AAG", "AT", "ATA", "ATT", "ATC", "ATG",
             "AC", "ACA", "ACT", "ACC", "ACG", "AG", "AGA", "AGT", "AGC", "AGG",
             "T", "TA", "TAA", "TAT", "TAC", "TAG", "TT", "TTA", "TTT", "TTC", "TTG",
             "TC", "TCA", "TCT", "TCC", "TCG", "TG", "TGA", "TGT", "TGC", "TGG",
             "C", "CA", "CAA", "CAT", "CAC", "CAG", "CT", "CTT", "CTC", "CTG",
             "CC", "CCA", "CCT", "CCC", "CCG", "CG", "CGA", "CGC", "CGG",
             "G", "GA", "GAA", "GAT", "GAC", "GAG", "GT", "GTA", "GTT", "GTC", "GTG",
             "GC", "GCA", "GCT", "GCC", "GCG", "GG", "GGA", "GGT", "GGC", "GGG"]
    for kmer in kmers:
        features.append(kmer_frequency(seq, kmer))
    # 167-186. 分布特征（A/T/C/G的5个分位点）
    for base in ["A", "T", "C", "G"]:
        for percentile in [0.01, 0.25, 0.50, 0.75, 1.00]:
            features.append(distribution_percentile(seq, base, percentile))
    # 187-195. 剪接位点特征
    for kmer in ["GT", "GC", "AT", "AG"]:
        features.append(donor_contains(seq, kmer))  # donor
    # acceptor额外计算AC
    for kmer in ["GT", "GC", "AT", "AG", "AC"]:
        features.append(acceptor_contains(seq, kmer))  # acceptor
    return features

def calculate_up_features(seq):
    """计算上游区域特征（特征196-198）"""
    features = []
    # 196-197. 分布特征
    features.append(distribution_percentile(seq, "A", 1.00))  # 100%A
    features.append(distribution_percentile(seq, "G", 1.00))  # 100%G
    # 198. donor位点AG
    features.append(donor_contains(seq, "AG"))
    return features

def calculate_down_features(seq):
    """计算下游区域特征（特征199-208）"""
    features = []
    # 199. GC含量
    features.append(gc_content(seq))
    # 200-206. k-mer频率
    kmers = ["AT", "T", "CCG", "CG", "CGC", "CGG", "G"]
    for kmer in kmers:
        features.append(kmer_frequency(seq, kmer))
    # 207. 分布特征
    features.append(distribution_percentile(seq, "G", 1.00))  # 100%G
    # 208. donor位点AG
    features.append(donor_contains(seq, "AG"))
    return features

def calculate_region_features(region_name, region_seq, features_list):
    """通用区域特征计算路由"""
    if region_name == "allseq":
        return calculate_allseq_features(region_seq)
    elif region_name == "asseq":
        return calculate_asseq_features(region_seq)
    elif region_name == "up":
        return calculate_up_features(region_seq)
    elif region_name == "down":
        return calculate_down_features(region_seq)
    elif region_name == "up30as30":
        # 209-232: 组合区域特殊逻辑
        features = []
        # 209. 长度能否被3整除
        features.append(1 if len(region_seq) % 3 == 0 else 0)
        # 210. GC含量
        features.append(gc_content(region_seq))
        # 211-222. k-mer频率
        kmers = ["AT", "AGG", "TAA", "CA", "CG", "G", "GT", "GTA", "GTG", "GG", "GGT", "GGG"]
        for kmer in kmers:
            features.append(kmer_frequency(region_seq, kmer))
        # 223-230. 分布特征
        features.append(distribution_percentile(region_seq, "A", 0.50))
        features.append(distribution_percentile(region_seq, "A", 0.75))
        features.append(distribution_percentile(region_seq, "A", 1.00))
        features.append(distribution_percentile(region_seq, "T", 0.75))
        features.append(distribution_percentile(region_seq, "T", 1.00))
        features.append(distribution_percentile(region_seq, "G", 0.50))
        features.append(distribution_percentile(region_seq, "G", 0.75))
        features.append(distribution_percentile(region_seq, "G", 1.00))
        # 231-232. 剪接位点
        features.append(donor_contains(region_seq, "GT"))
        features.append(donor_contains(region_seq, "AG"))
        return features
    # 其他区域处理（根据需要添加）
    else:
        return []

def calculate_up30down30_features(region_seq):
    """计算up30down30组合区域特征(233-234)"""
    features = []
    # 233. GC含量
    features.append(gc_content(region_seq))
    # 234. 长度能否被3整除
    features.append(1 if len(region_seq) % 3 == 0 else 0)
    return features

def calculate_as30down30_features(region_seq):
    """计算as30down30组合区域特征(235-286)"""
    features = []
    # 235. 长度能否被3整除
    features.append(1 if len(region_seq) % 3 == 0 else 0)
    # 236. GC含量
    features.append(gc_content(region_seq))
    # 237-271. k-mer频率(详细列表见表格)
    kmers = ["A", "AA", "AAA", "AAT", "AAG", "AT", "AG", "T", "TT", "TTT", 
             "TTC", "TC", "TCT", "TCC", "TGT", "C", "CAA", "CT", "CTT", 
             "CTC", "CC", "CCT", "CCC", "CG", "CGC", "CGG", "G", "GA", 
             "GAA", "GAG", "GC", "GCG", "GG", "GGA", "GGC", "GGG"]
    for kmer in kmers:
        features.append(kmer_frequency(region_seq, kmer))
    # 272-285. 分布特征(分位点计算)
    distributions = [
        ("A", 0.01), ("A", 0.25), ("A", 0.50), ("A", 0.75),
        ("T", 0.01), ("T", 0.50), ("T", 0.75),
        ("C", 0.01), ("C", 0.25), ("C", 0.50),
        ("G", 0.01), ("G", 0.25), ("G", 0.50), ("G", 0.75)
    ]
    for base, percentile in distributions:
        features.append(distribution_percentile(region_seq, base, percentile))
    # 286. acceptor位点GT
    features.append(acceptor_contains(region_seq, "GT"))
    return features

def calculate_up50as50_features(region_seq):
    """计算up50as50组合区域特征(287-320)"""
    features = []
    # 287. 长度能否被3整除
    features.append(1 if len(region_seq) % 3 == 0 else 0)
    # 288. GC含量
    features.append(gc_content(region_seq))
    # 289. TGA终止子数量
    features.append(region_seq.count("TGA"))
    # 290-306. k-mer频率
    kmers = ["A", "AA", "AAA", "AT", "AGG", "CA", "CC", "CCT", "CCC", "CG", 
             "G", "GT", "GTA", "GTG", "GG", "GGT", "GGG"]
    for kmer in kmers:
        features.append(kmer_frequency(region_seq, kmer))
    # 307-318. 分布特征
    distributions = [
        ("A", 0.25), ("A", 0.50), ("A", 0.75), ("A", 1.00),
        ("T", 0.50), ("T", 0.75), ("T", 1.00),
        ("C", 1.00),
        ("G", 0.25), ("G", 0.50), ("G", 0.75), ("G", 1.00)
    ]
    for base, percentile in distributions:
        features.append(distribution_percentile(region_seq, base, percentile))
    # 319-320. 剪接位点
    features.append(donor_contains(region_seq, "GT"))
    features.append(donor_contains(region_seq, "AG"))
    return features

def calculate_up50down50_features(region_seq):
    """计算up50down50组合区域特征(321-323)"""
    features = []
    # 321. GC含量
    features.append(gc_content(region_seq))
    # 322. 100%G分位点
    features.append(distribution_percentile(region_seq, "G", 1.00))
    # 323. donor位点AG
    features.append(donor_contains(region_seq, "AG"))
    return features

def calculate_as50down50_features(region_seq):
    """计算as50down50组合区域特征(324-374)"""
    features = []
    # 324. 长度能否被3整除
    features.append(1 if len(region_seq) % 3 == 0 else 0)
    # 325. GC含量
    features.append(gc_content(region_seq))
    # 326-327. 终止子数量
    features.append(region_seq.count("TAA"))
    features.append(region_seq.count("TGA"))
    # 328-361. k-mer频率
    kmers = ["A", "AA", "AAA", "AAG", "AT", "ATT", "AG", "AGA", "T", "TT", 
             "TTT", "TTC", "TC", "TCT", "TCC", "C", "CAA", "CT", "CTT", 
             "CTC", "CC", "CCT", "CCC", "CCG", "CG", "CGC", "CGG", "G", 
             "GA", "GAA", "GC", "GCG", "GG", "GGG"]
    for kmer in kmers:
        features.append(kmer_frequency(region_seq, kmer))
    # 362-371. 分布特征
    distributions = [
        ("A", 0.01), ("A", 0.50),
        ("T", 0.01), ("T", 0.50), ("T", 0.75),
        ("C", 0.01),
        ("G", 0.01), ("G", 0.25), ("G", 0.50), ("G", 1.00)
    ]
    for base, percentile in distributions:
        features.append(distribution_percentile(region_seq, base, percentile))
    # 372-374. 剪接位点
    features.append(acceptor_contains(region_seq, "GT"))
    features.append(donor_contains(region_seq, "AG"))
    features.append(acceptor_contains(region_seq, "AG"))
    return features

def calculate_down20bp_features(region_seq):
    """计算下游20bp特征(375-376)"""
    features = []
    # 375. GC含量
    features.append(gc_content(region_seq))
    # 376. CG频率
    features.append(kmer_frequency(region_seq, "CG"))
    return features
# ================== 主流程 ==================
input_file = "/home/shencc/ASresult/SUPPA/sequence.tsv"
output_file = "/home/shencc/ASresult/SUPPA/sequencefeature0.txt"

with open(input_file, "r") as fin, open(output_file, "w") as fout:
    for line in fin:
        # 解析数据
        fields = line.strip().replace("a", "A").replace("g", "G")
        fields = fields.replace("c", "C").replace("t", "T").split(",")
        if len(fields) < 4:
            print(f"跳过无效行: {line.strip()}", file=sys.stderr)
            continue
            
        seq_name, up_seq, as_seq, down_seq = fields[0], fields[1], fields[2], fields[3]
        all_seq = up_seq + as_seq + down_seq
        features_list = []
        
        # === 1. 特征1: AS区域长度 ===
        features_list.append(len(as_seq))
        
        # === 2-25. 下游motif (Dmotifs) ===
        for motif in Dmotifs:
            features_list.append(1 if motif in down_seq else 0)
        
        # === 26-49. 上游motif (Umotifs) ===
        for motif in Umotifs:
            features_list.append(1 if motif in up_seq else 0)
        
        # === 45-80. 全序列特征 ===
        features_list.extend(calculate_allseq_features(all_seq))
        
        # === 81-195. AS区域特征 ===
        features_list.extend(calculate_asseq_features(as_seq))
        
        # === 196-198. 上游特征 ===
        features_list.extend(calculate_up_features(up_seq))
        
        # === 199-208. 下游特征 ===
        features_list.extend(calculate_down_features(down_seq))
        
        # === 209-232. up30as30组合区域特征 ===
        up30 = up_seq[-30:] if len(up_seq) >= 30 else up_seq
        as30 = as_seq[:30] if len(as_seq) >= 30 else as_seq
        up30as30 = up30 + as30
        features_list.extend(calculate_region_features("up30as30", up30as30, features_list))

        # 特征233-234: up30down30区域
        up30 = up_seq[-30:] if len(up_seq) >= 30 else up_seq
        down30 = down_seq[:30] if len(down_seq) >= 30 else down_seq
        up30down30 = up30 + down30
        features_list.extend(calculate_up30down30_features(up30down30))
        
        # 特征235-286: as30down30区域
        as30 = as_seq[-30:] if len(as_seq) >= 30 else as_seq
        as30down30 = as30 + down30
        features_list.extend(calculate_as30down30_features(as30down30))
        
        # 特征287-320: up50as50区域
        up50 = up_seq[-50:] if len(up_seq) >= 50 else up_seq
        as50 = as_seq[:50] if len(as_seq) >= 50 else as_seq
        up50as50 = up50 + as50
        features_list.extend(calculate_up50as50_features(up50as50))
        
        # 特征321-323: up50down50区域
        down50 = down_seq[:50] if len(down_seq) >= 50 else down_seq
        up50down50 = up50 + down50
        features_list.extend(calculate_up50down50_features(up50down50))
        
        # 特征324-374: as50down50区域
        as50 = as_seq[-50:] if len(as_seq) >= 50 else as_seq
        as50down50 = as50 + down50
        features_list.extend(calculate_as50down50_features(as50down50))
        
        # 特征375-376: down20bp区域
        down20 = down_seq[:20] if len(down_seq) >= 20 else down_seq
        features_list.extend(calculate_down20bp_features(down20))
        # 写入结果
        str_features = ",".join(map(str, features_list))
        fout.write(f"{line.strip()},{str_features}\n")

print(f"特征提取完成，保存至 {output_file}")