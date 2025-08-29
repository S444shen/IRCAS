import os
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MKL_THREADING_LAYER"] = "GNU"

import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from torch_geometric.data import Data


# ======================== 配置参数 ========================
# LABEL_MAP = {"A3": 0, "A5": 1, "SE": 2, "RI": 3}  # 不再需要标签映射
K_MIN = 4
FEATURE_DIM = 12  # 4 (one-hot) + 8 (PE)
EDGE_FEATURE_DIM = 4  # 2 (out-degree + in-degree) + 2 (relative position PE)
BATCH_SIZE = 256

# ======================== 轻量图结构 ========================
class LiteGraph:
    """内存优化的图表示"""
    __slots__ = ['nodes', 'edges']
    
    def __init__(self):
        self.nodes = defaultdict(dict)  # {node: {position1, position2}}
        self.edges = defaultdict(list)  # {src: [dst1, dst2...]}

# ======================== 正弦位置编码 ========================
def sinusoidal_pe(positions, pe_dim=8):
    """生成正弦/余弦位置编码"""
    positions = torch.tensor(positions, dtype=torch.float).unsqueeze(1)  # [num_nodes, 1]
    div_term = torch.exp(torch.arange(0, pe_dim, 2).float() * (-np.log(10000.0) / pe_dim))  # [pe_dim/2]
    pe = torch.zeros(len(positions), pe_dim)
    pe[:, 0::2] = torch.sin(positions * div_term)
    pe[:, 1::2] = torch.cos(positions * div_term)
    return pe

# ======================== 特征处理器 ========================
class OptimizedFeatureProcessor:
    def __init__(self):
        self.graph_stats = None

    def collect_global_stats(self, df):
        """仅收集图级特征统计量"""
        # 跳过前4列：transcript_pair, upstream, as_region, downstream
        # 从第4列开始是实际的特征数据
        feature_cols = df.iloc[:, 4:]
        graph_min = np.full(feature_cols.shape[1], np.inf)
        graph_max = np.full(feature_cols.shape[1], -np.inf)

        for _, row in df.iterrows():
            graph_feats = np.nan_to_num(row[4:].values.astype(np.float32), nan=0.0)
            graph_min = np.minimum(graph_min, graph_feats)
            graph_max = np.maximum(graph_max, graph_feats)

        self.graph_stats = {
            'min': graph_min,
            'max': graph_max,
            'range': np.clip(graph_max - graph_min, 1e-6, None)
        }

    def process_dataset(self, df):
        """分批处理数据"""
        data_list = []
        for i in range(0, len(df), BATCH_SIZE):
            batch = df.iloc[i:i+BATCH_SIZE]
            batch_data = self._process_batch(batch)
            data_list.extend(batch_data)
            del batch, batch_data
        return self._postprocess(data_list)

    def _process_batch(self, batch):
        """处理单个批次"""
        batch_data = []
        for _, row in batch.iterrows():
            graph = self._build_optimized_graph(row)
            data = self._convert_to_pyg(graph, row)
            batch_data.append(data)
            
        return batch_data

    def _build_optimized_graph(self, row):
        """构建内存优化的图结构"""
        seq1 = row[1] + row[2] + row[3]
        seq2 = row[1] + row[3]
        k = max(self._find_min_k(seq1, seq2), K_MIN)
        
        graph = LiteGraph()
        
        # 处理seq1
        prev = None
        for i in range(len(seq1) - k + 1):
            node = seq1[i:i+k]
            graph.nodes[node]['position1'] = i + 1
            if prev is not None:
                graph.edges[prev].append(node)
            prev = node
        
        # 处理seq2
        prev = None
        for i in range(len(seq2) - k + 1):
            node = seq2[i:i+k]
            if 'position2' not in graph.nodes[node]:
                graph.nodes[node]['position2'] = i + 1
            if prev is not None:
                if node not in graph.edges[prev]:
                    graph.edges[prev].append(node)
            prev = node
        
        return graph

    def _convert_to_pyg(self, graph, row):
        """转换为PyG Data对象"""
        nodes = list(graph.nodes.keys())
        node_map = {n: i for i, n in enumerate(nodes)}
        
        x = self._gen_node_features(nodes, graph.nodes)
        edge_index, edge_attr = self._gen_edge_features(graph, node_map, nodes)
        graph_feats = self._gen_graph_features(row)
        
        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor([0], dtype=torch.long),  # 临时标签，将由分类器预测
            transcript_pair=row[0],  # 保存转录本对名称
            graph_features=graph_feats
        )

    def _gen_node_features(self, nodes, node_positions):
        """生成节点特征：one-hot编码 + 正弦PE (float32)"""
        features = []
        positions = []
        for node in nodes:
            # One-hot编码（基于最后一个核苷酸）
            last_char = node[-1]
            one_hot = [1.0 if c == last_char else 0.0 for c in 'ACGT']
            
            # 获取位置（优先 position1，若无则用 position2）
            pos = node_positions[node].get('position1', node_positions[node].get('position2', 0))
            positions.append(pos)
            features.append(one_hot)
        
        # 生成正弦PE
        pe = sinusoidal_pe(positions, pe_dim=8)
        
        # 拼接 one-hot 和 PE
        features = torch.tensor(features, dtype=torch.float32)
        features = torch.cat([features, pe], dim=1)  # [num_nodes, 4 + 8]
        return features

    def _gen_edge_features(self, graph, node_map, nodes):
        """生成边特征：出度、入度 + 相对位置PE (float32)"""
        edge_index = []
        edge_attr = []
        
        # 计算每个节点的入度
        in_degree = defaultdict(int)
        for src, dsts in graph.edges.items():
            for dst in dsts:
                in_degree[dst] += 1
        
        for src, dsts in graph.edges.items():
            src_idx = node_map.get(src, -1)
            if src_idx == -1:
                continue
                
            out_degree = len(dsts)
            for dst in dsts:
                dst_idx = node_map.get(dst, -1)
                if dst_idx == -1:
                    continue
                
                # 出度和入度特征
                in_deg = in_degree[dst]
                degree_feats = [float(out_degree >= 2), float(in_deg >= 2)]
                
                # 相对位置PE
                src_pos = graph.nodes[src].get('position1', graph.nodes[src].get('position2', 0))
                dst_pos = graph.nodes[dst].get('position1', graph.nodes[dst].get('position2', 0))
                rel_pos = abs(dst_pos - src_pos)
                rel_pe = sinusoidal_pe([rel_pos], pe_dim=2)[0].tolist()  # 2维PE
                
                edge_index.append([src_idx, dst_idx])
                edge_attr.append(degree_feats + rel_pe)
                
        if not edge_index:
            num_nodes = len(graph.nodes)
            if num_nodes == 1:
                edge_index = [[0, 0]]
                edge_attr = [[0.0] * EDGE_FEATURE_DIM]
        
        return (
            torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
            torch.tensor(edge_attr, dtype=torch.float32)
        )

    def _gen_graph_features(self, row):
        """生成图级特征 (float32)"""
        # 从第4列开始提取特征（跳过转录本对名称和3个序列列）
        raw = np.nan_to_num(row[4:].values.astype(np.float32), nan=0.0)
        
        # 对于plant模型，确保有296个特征（297-1，因为不需要标签）
        expected_features = 296 if len(raw) >= 296 else len(self.graph_stats['min'])
        
        if len(raw) < expected_features:
            # 用0补齐缺失的特征
            padding = np.zeros(expected_features - len(raw), dtype=np.float32)
            raw = np.concatenate([raw, padding])
            print(f"Padded {expected_features - len(raw)} features to reach {expected_features}")
        elif len(raw) > expected_features:
            # 截断多余的特征
            raw = raw[:expected_features]
        
        # 确保统计量维度匹配
        if len(self.graph_stats['min']) != len(raw):
            # 重新调整统计量维度
            if len(raw) > len(self.graph_stats['min']):
                # 扩展统计量
                extra_count = len(raw) - len(self.graph_stats['min'])
                self.graph_stats['min'] = np.concatenate([self.graph_stats['min'], np.zeros(extra_count)])
                self.graph_stats['max'] = np.concatenate([self.graph_stats['max'], np.ones(extra_count)])
                self.graph_stats['range'] = np.concatenate([self.graph_stats['range'], np.ones(extra_count)])
            else:
                # 截断统计量
                self.graph_stats['min'] = self.graph_stats['min'][:len(raw)]
                self.graph_stats['max'] = self.graph_stats['max'][:len(raw)]
                self.graph_stats['range'] = self.graph_stats['range'][:len(raw)]
        
        normalized = (raw - self.graph_stats['min']) / self.graph_stats['range']
        # 返回2D张量 (1, features) 而不是1D
        return torch.tensor(normalized, dtype=torch.float32).unsqueeze(0)

    def _postprocess(self, data_list):
        """后处理验证"""
        valid_data = []
        for data in data_list:
            if data.edge_index.numel() > 0:
                max_idx = data.edge_index.max()
                if max_idx >= data.num_nodes:
                    continue
            if data.x.size(1) != FEATURE_DIM:
                continue
            if data.edge_attr.size(1) != EDGE_FEATURE_DIM:
                continue
            valid_data.append(data)
        return valid_data

    def _find_min_k(self, seq1, seq2):
        """动态确定最小k值"""
        max_len = max(len(seq1), len(seq2))
        for k in range(K_MIN, max_len + 1):
            if self._has_unique_kmers(seq1, seq2, k):
                return k
        return max_len

    def _has_unique_kmers(self, seq1, seq2, k):
        """检查k-mer唯一性"""
        kmers1 = {seq1[i:i+k] for i in range(len(seq1) - k + 1)}
        kmers2 = {seq2[i:i+k] for i in range(len(seq2) - k + 1)}
        return kmers1.isdisjoint(kmers2)

# ======================== 执行流程 ========================
if __name__ == "__main__":
    df = pd.read_csv("mousesequencefeature.txt", header=None, engine='c')
    processor = OptimizedFeatureProcessor()
    
    print("Collecting Global Statistics...")
    processor.collect_global_stats(df)
    
    print("Processing Data...")
    dataset = processor.process_dataset(df)
    
    torch.save(dataset, 'mousedataset.pt')
    print(f"Dataset processing completed. Valid samples: {len(dataset)}")