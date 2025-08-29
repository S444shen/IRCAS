#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复的预测脚本 - 自动适配模型结构
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch_geometric.nn import GATv2Conv, TransformerConv, AttentionalAggregation, global_mean_pool
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import json
from typing import List, Dict, Optional, Union

# ===================== 模型定义（与训练时保持一致） =====================
class EnhancedLightGNN(nn.Module):
    def __init__(self, num_node_features, num_edge_features, num_graph_features, num_classes):
        super().__init__()
        self.hidden_dim = 64
        self.num_heads = 4
        
        # 节点编码器
        self.node_encoder = nn.Sequential(
            nn.Linear(num_node_features, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        # 边编码器
        self.edge_encoder = nn.Sequential(
            nn.Linear(num_edge_features, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, self.hidden_dim // 2)
        ) if num_edge_features > 0 else None
        
        # 图级特征编码器 - 只在需要时创建
        if num_graph_features > 0:
            self.graph_encoder = nn.Sequential(
                nn.Linear(num_graph_features, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.LayerNorm(64),
                nn.GELU()
            )
        else:
            self.graph_encoder = None
        
        # GNN层
        self.gat1 = GATv2Conv(
            self.hidden_dim,
            self.hidden_dim,
            heads=self.num_heads,
            edge_dim=self.hidden_dim // 2 if num_edge_features > 0 else None,
            concat=True,
            dropout=0.2,
            add_self_loops=True
        )
        
        self.gat_projection = nn.Sequential(
            nn.Linear(self.hidden_dim * self.num_heads, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        self.transformer1 = TransformerConv(
            self.hidden_dim,
            self.hidden_dim,
            heads=self.num_heads,
            edge_dim=self.hidden_dim // 2 if num_edge_features > 0 else None,
            concat=False,
            dropout=0.2,
            beta=True
        )
        
        self.transformer2 = TransformerConv(
            self.hidden_dim,
            self.hidden_dim,
            heads=self.num_heads,
            edge_dim=self.hidden_dim // 2 if num_edge_features > 0 else None,
            concat=False,
            dropout=0.2,
            beta=True
        )
        
        self.gat2 = GATv2Conv(
            self.hidden_dim,
            self.hidden_dim,
            heads=1,
            edge_dim=self.hidden_dim // 2 if num_edge_features > 0 else None,
            concat=False,
            dropout=0.2,
            add_self_loops=True
        )
        
        # 归一化层
        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.norm2 = nn.LayerNorm(self.hidden_dim)
        self.norm3 = nn.LayerNorm(self.hidden_dim)
        self.norm4 = nn.LayerNorm(self.hidden_dim)
        self.dropout = nn.Dropout(0.2)
        
        # 图池化
        self.graph_attention = AttentionalAggregation(
            nn.Sequential(
                nn.Linear(self.hidden_dim, 32),
                nn.Tanh(),
                nn.Linear(32, 1)
            )
        )
        
        # 自注意力
        self.self_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=0.2,
            batch_first=True
        )
        
        # 特征融合器 - 根据是否有图特征调整输入维度
        if num_graph_features > 0:
            fusion_input_dim = self.hidden_dim * 2 + 64  # 192
        else:
            fusion_input_dim = self.hidden_dim * 2  # 128
            
        self.feature_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 192),
            nn.LayerNorm(192),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(192, 128),
            nn.LayerNorm(128),
            nn.GELU()
        )
        
        # 分类器
        self.classifier = nn.Linear(128, num_classes)
        
        # 位置编码投影
        self.pe_projection = nn.Linear(8, self.hidden_dim, bias=False)
        
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # 获取图级特征
        graph_features = data.graph_features if hasattr(data, 'graph_features') else None
        
        # 保存原始位置编码
        original_pe = x[:, -8:]
        
        # 节点编码
        x = self.node_encoder(x)
        
        # 边编码
        edge_feat = self.edge_encoder(edge_attr) if self.edge_encoder and edge_attr is not None else None
        
        # 位置编码残差
        pe_enhanced = self.pe_projection(original_pe)
        x = x + 0.1 * pe_enhanced
        
        # GNN层处理
        if edge_index.numel() > 0:
            # GAT层1
            x_residual = x
            x = self.gat1(x, edge_index, edge_attr=edge_feat)
            x = F.gelu(x)
            x = self.gat_projection(x)
            x = self.norm1(x + x_residual)
            x = self.dropout(x)
            
            # Transformer层
            x_residual = x
            x = self.transformer1(x, edge_index, edge_attr=edge_feat)
            x = F.gelu(x)
            x = self.norm2(x + x_residual)
            x = self.dropout(x)
            
            x_residual = x
            x = self.transformer2(x, edge_index, edge_attr=edge_feat)
            x = F.gelu(x)
            x = self.norm3(x + x_residual)
            x = self.dropout(x)
            
            # GAT层2
            x_residual = x
            x = self.gat2(x, edge_index, edge_attr=edge_feat)
            x = F.gelu(x)
            x = self.norm4(x + x_residual)
            x = self.dropout(x)
        else:
            x = F.gelu(x)
        
        # 图池化
        attention_pool = self.graph_attention(x, batch)
        mean_pool = global_mean_pool(x, batch)
        
        # 自注意力增强
        attention_pool_expanded = attention_pool.unsqueeze(0)
        attention_pool_refined, _ = self.self_attention(
            attention_pool_expanded,
            attention_pool_expanded,
            attention_pool_expanded
        )
        attention_pool_refined = attention_pool_refined.squeeze(0)
        
        # 组合池化特征
        pooled_features = torch.cat([attention_pool_refined, mean_pool], dim=1)
        
        # 处理图级特征并融合
        if graph_features is not None and self.graph_encoder is not None:
            if graph_features.dim() == 1:
                graph_features = graph_features.unsqueeze(0)
            graph_features = graph_features.float()
            graph_encoded = self.graph_encoder(graph_features)
            combined_features = torch.cat([pooled_features, graph_encoded], dim=1)
        else:
            # 如果模型有graph_encoder但数据没有graph_features，填充零
            if self.graph_encoder is not None:
                zero_graph_features = torch.zeros(pooled_features.size(0), 64, device=pooled_features.device)
                combined_features = torch.cat([pooled_features, zero_graph_features], dim=1)
            else:
                combined_features = pooled_features
        
        # 特征融合和分类
        features = self.feature_fusion(combined_features)
        logits = self.classifier(features)
        
        return logits, features

# ===================== 改进的预测器类 =====================
class ModelPredictor:
    """模型预测器 - 自动检测模型配置"""
    
    def __init__(self, model_path: str, device: str = None):
        """
        初始化预测器
        
        Args:
            model_path: 模型checkpoint路径
            device: 运行设备 ('cuda' 或 'cpu')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.model = None
        self.label_map = {0: "A3", 1: "A5", 2: "SE", 3: "RI"}
        self.has_graph_features = False
        
        # 加载模型
        self._load_model()
        
    def _detect_model_config(self, state_dict):
        """自动检测模型配置"""
        # 检测是否有图级特征编码器
        has_graph_encoder = any('graph_encoder' in k for k in state_dict.keys())
        
        # 检测图级特征的输入维度
        num_graph_features = 0
        if has_graph_encoder:
            # 从graph_encoder第一层的权重获取输入维度
            graph_encoder_key = 'graph_encoder.0.weight'
            if graph_encoder_key in state_dict:
                num_graph_features = state_dict[graph_encoder_key].shape[1]
                print(f"  - Graph features dimension: {num_graph_features}")
        
        return {
            'has_graph_features': has_graph_encoder,
            'num_graph_features': num_graph_features
        }
        
    def _load_model(self):
        """加载训练好的模型"""
        print(f"Loading model from {self.model_path}")
        
        # 加载checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # 获取state_dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        # 自动检测模型配置
        config = self._detect_model_config(state_dict)
        self.has_graph_features = config['has_graph_features']
        
        print(f"Detected model configuration:")
        print(f"  - Has graph features: {self.has_graph_features}")
        
        # 初始化模型
        self.model = EnhancedLightGNN(
            num_node_features=12,  # 4 (one-hot) + 8 (PE)
            num_edge_features=4,    # 2 (degree) + 2 (relative PE)
            num_graph_features=config['num_graph_features'],
            num_classes=4
        ).to(self.device)
        
        # 加载模型权重
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        print(f"Model loaded successfully. Device: {self.device}")
        
    def preprocess_data(self, dataset: Union[str, List[Data]]) -> DataLoader:
        """预处理数据"""
        if isinstance(dataset, str):
            print(f"Loading dataset from {dataset}")
            data_list = torch.load(dataset)
        else:
            data_list = dataset
        
        # 数据验证和修复
        fixed_data = self._validate_and_fix_data(data_list)
        
        # 创建DataLoader
        loader = DataLoader(
            fixed_data,
            batch_size=64,
            shuffle=False,
            num_workers=0
        )
        
        return loader
    
    def _validate_and_fix_data(self, data_list: List[Data]) -> List[Data]:
        """验证和修复数据"""
        fixed_data = []
        
        # 获取期望的图特征维度
        expected_graph_dim = 0
        if self.has_graph_features and self.model.graph_encoder is not None:
            # 从模型的第一层获取期望的输入维度
            expected_graph_dim = self.model.graph_encoder[0].in_features
        
        for i, g in enumerate(data_list):
            # 确保特征是float类型
            if hasattr(g, 'x') and g.x is not None:
                g.x = g.x.float()
            
            if hasattr(g, 'edge_attr') and g.edge_attr is not None:
                g.edge_attr = g.edge_attr.float()
            
            # 处理图级特征
            if self.has_graph_features and expected_graph_dim > 0:
                if hasattr(g, 'graph_features') and g.graph_features is not None:
                    g.graph_features = g.graph_features.float()
                    # 检查维度是否匹配
                    if g.graph_features.dim() == 1:
                        current_dim = g.graph_features.size(0)
                    else:
                        current_dim = g.graph_features.size(-1)
                    
                    if current_dim != expected_graph_dim:
                        print(f"Warning: Graph {i} has {current_dim} graph features, expected {expected_graph_dim}")
                        # 调整维度
                        if current_dim < expected_graph_dim:
                            # 填充零
                            padding = torch.zeros(expected_graph_dim - current_dim)
                            g.graph_features = torch.cat([g.graph_features, padding])
                        else:
                            # 截断
                            g.graph_features = g.graph_features[:expected_graph_dim]
                else:
                    # 如果模型需要但数据没有，创建零特征
                    if i == 0:  # 只打印一次警告
                        print(f"Warning: Graphs missing graph_features, using zeros (dim={expected_graph_dim})")
                    g.graph_features = torch.zeros(expected_graph_dim, dtype=torch.float32)
            
            fixed_data.append(g)
        
        print(f"Processed {len(fixed_data)} graphs")
        return fixed_data
    
    def predict(self, dataset: Union[str, List[Data]], 
                return_probs: bool = False,
                batch_size: int = 64) -> Dict:
        """对数据集进行预测"""
        # 预处理数据
        loader = self.preprocess_data(dataset)
        
        predictions = []
        probabilities = []
        features_list = []
        
        print("Starting prediction...")
        self.model.eval()
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Predicting"):
                batch = batch.to(self.device)
                
                try:
                    # 前向传播
                    logits, features = self.model(batch)
                    
                    # 计算概率
                    probs = F.softmax(logits, dim=1)
                    
                    # 获取预测类别
                    preds = probs.argmax(dim=1)
                    
                    predictions.extend(preds.cpu().numpy())
                    probabilities.extend(probs.cpu().numpy())
                    features_list.extend(features.cpu().numpy())
                except Exception as e:
                    print(f"Error in batch: {e}")
                    # 对错误的批次使用默认预测
                    batch_size = batch.batch.max().item() + 1
                    predictions.extend([0] * batch_size)
                    if return_probs:
                        probabilities.extend([[0.25, 0.25, 0.25, 0.25]] * batch_size)
                    features_list.extend(np.zeros((batch_size, 128)))
        
        # 整理结果
        results = {
            'predictions': np.array(predictions),
            'labels': [self.label_map[p] for p in predictions],
            'features': np.array(features_list)
        }
        
        if return_probs:
            results['probabilities'] = np.array(probabilities)
            results['confidence'] = np.max(probabilities, axis=1)
        
        print(f"Prediction completed. Processed {len(predictions)} samples.")
        
        return results
    
    def save_predictions(self, results: Dict, output_path: str):
        """保存预测结果"""
        # 保存为CSV
        df = pd.DataFrame({
            'sample_id': range(len(results['predictions'])),
            'predicted_class': results['predictions'],
            'predicted_label': results['labels']
        })
        
        if 'confidence' in results:
            df['confidence'] = results['confidence']
        
        if 'probabilities' in results:
            for i in range(results['probabilities'].shape[1]):
                df[f'prob_class_{i}'] = results['probabilities'][:, i]
        
        csv_path = output_path.replace('.json', '.csv')
        df.to_csv(csv_path, index=False)
        print(f"Predictions saved to {csv_path}")
        
        # 保存为JSON - 修复int64问题
        # 获取类别分布
        unique, counts = np.unique(results['predictions'], return_counts=True)
        class_distribution = {}
        for cls, count in zip(unique, counts):
            # 转换为Python原生类型
            class_distribution[str(int(cls))] = int(count)
        
        json_results = {
            'predictions': [int(x) for x in results['predictions']],  # 转换为int
            'labels': results['labels'],
            'statistics': {
                'total_samples': len(results['predictions']),
                'class_distribution': class_distribution
            }
        }
        
        if 'confidence' in results:
            json_results['mean_confidence'] = float(np.mean(results['confidence']))
            json_results['min_confidence'] = float(np.min(results['confidence']))
            json_results['max_confidence'] = float(np.max(results['confidence']))
        
        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=4)
        print(f"Results summary saved to {output_path}")

# ===================== 快速使用函数 =====================
def quick_predict(model_path: str, data_path: str, output_path: str = None):
    """快速预测函数"""
    predictor = ModelPredictor(model_path)
    results = predictor.predict(data_path, return_probs=True)
    
    # 打印统计
    print("\n=== Prediction Statistics ===")
    unique, counts = np.unique(results['predictions'], return_counts=True)
    for cls, count in zip(unique, counts):
        label = predictor.label_map[cls]
        percentage = count / len(results['predictions']) * 100
        print(f"Class {cls} ({label}): {count} samples ({percentage:.2f}%)")
    
    if 'confidence' in results:
        print(f"\nMean confidence: {np.mean(results['confidence']):.4f}")
    
    # 保存结果
    if output_path:
        predictor.save_predictions(results, output_path)
    
    return results

# ===================== 主函数 =====================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict using trained GNN model')
    parser.add_argument('--model_path', type=str, default='best_model1.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to dataset for prediction')
    parser.add_argument('--output_path', type=str, default='predictions.json',
                       help='Path to save predictions')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # 快速预测
    results = quick_predict(
        model_path=args.model_path,
        data_path=args.data_path,
        output_path=args.output_path
    )
    
    print("\nPrediction completed successfully!")