#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, TransformerConv, AttentionalAggregation, global_mean_pool
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix, classification_report, precision_score
from sklearn.model_selection import train_test_split

# Configure environment and threads
torch.multiprocessing.set_sharing_strategy('file_system')
os.environ["OMP_NUM_THREADS"] = "128"
os.environ["MKL_NUM_THREADS"] = "128"
torch.set_num_threads(128)

# Set random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
set_seed(42)

# Custom transform to normalize only one-hot features, preserving PE
class SelectiveNormalizeFeatures:
    def __init__(self, pe_dim=8):
        self.pe_dim = pe_dim

    def __call__(self, data):
        if data.x is not None:
            # Normalize one-hot part, preserve PE
            one_hot = data.x[:, :-self.pe_dim]
            pe = data.x[:, -self.pe_dim:]
            one_hot = torch.nn.functional.normalize(one_hot, p=1, dim=1)
            data.x = torch.cat([one_hot, pe], dim=1)
        return data

# Improved Center Loss
class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, lambda_c=0.01):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.lambda_c = lambda_c
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.centers)

    def forward(self, features, labels):
        batch_size = features.size(0)
        centers_batch = self.centers.index_select(0, labels.long())
        # 使用L2归一化的特征和中心
        features_norm = F.normalize(features, p=2, dim=1)
        centers_norm = F.normalize(centers_batch, p=2, dim=1)
        distances = torch.sum((features_norm - centers_norm)**2, dim=1)
        return self.lambda_c * torch.mean(distances)

# Focal Loss with Class-Balanced weighting
class CB_FocalLoss(nn.Module):
    def __init__(self, samples_per_cls, num_classes, beta=0.999, gamma=2.0):
        super(CB_FocalLoss, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.num_classes = num_classes
        self.samples_per_cls = torch.tensor(samples_per_cls, dtype=torch.float32)
        effective_num = 1.0 - torch.pow(self.beta, self.samples_per_cls)
        self.class_weights = (1.0 - self.beta) / (effective_num + 1e-8)
        self.class_weights = self.class_weights / self.class_weights.sum() * self.num_classes

    def forward(self, logits, labels):
        logits = logits.float()
        probs = F.softmax(logits, dim=1)
        labels_one_hot = F.one_hot(labels, num_classes=self.num_classes).float().to(logits.device)
        pt = (probs * labels_one_hot).sum(dim=1)
        weights = self.class_weights.to(logits.device)[labels]
        focal_weight = (1.0 - pt) ** self.gamma
        loss = -weights * focal_weight * torch.log(pt + 1e-8)
        return loss.mean()

# Improved Hybrid Loss
class ImprovedHybridLoss(nn.Module):
    def __init__(self, samples_per_cls, num_classes, feat_dim=128, 
                 alpha=0.15, beta=0.75, gamma=0.1, focal_gamma=2.0):
        super(ImprovedHybridLoss, self).__init__()
        self.alpha = alpha  # Focal Loss
        self.beta = beta    # Cross-Entropy (主导)
        self.gamma = gamma  # Center Loss
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        
        # 损失函数组件
        self.focal_loss = CB_FocalLoss(
            samples_per_cls=samples_per_cls,
            num_classes=num_classes,
            gamma=focal_gamma
        )
        
        # 使用带权重的交叉熵 - 确保权重是float类型
        weights = torch.tensor([1.0 / (s + 1.0) for s in samples_per_cls], dtype=torch.float32)
        weights = weights / weights.sum() * num_classes
        self.ce_loss = nn.CrossEntropyLoss(weight=weights)
        
        self.center_loss = CenterLoss(
            num_classes=num_classes, 
            feat_dim=feat_dim, 
            lambda_c=0.01
        )

    def forward(self, logits, features, labels):
        # 确保输入是正确的类型
        logits = logits.float()
        features = features.float()
        
        # 计算各项损失
        focal_loss = self.focal_loss(logits, labels)
        ce_loss = self.ce_loss(logits, labels)
        center_loss = self.center_loss(features, labels)
        
        # 直接加权求和，不做动态归一化
        total_loss = self.alpha * focal_loss + self.beta * ce_loss + self.gamma * center_loss
        
        # 返回总损失和各分项（用于监控）
        return total_loss, {
            'focal': focal_loss.item(),
            'ce': ce_loss.item(), 
            'center': center_loss.item()
        }

# Enhanced LightGNN with Graph Features
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
        
        # 图级特征编码器
        self.graph_encoder = nn.Sequential(
            nn.Linear(num_graph_features, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU()
        )
        
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
        
        # 特征融合器 - 结合节点特征和图级特征
        self.feature_fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 2 + 64, 256),  # 池化特征 + 图特征
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
        nn.init.xavier_normal_(self.pe_projection.weight, gain=0.1)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
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
        if graph_features is not None:
            # 确保图级特征是2D的并转换为float
            if graph_features.dim() == 1:
                graph_features = graph_features.unsqueeze(0)
            graph_features = graph_features.float()  # 转换为float32
            graph_encoded = self.graph_encoder(graph_features)
            # 结合池化特征和图级特征
            combined_features = torch.cat([pooled_features, graph_encoded], dim=1)
        else:
            # 如果没有图级特征，用零填充
            zero_graph_features = torch.zeros(pooled_features.size(0), 64, device=pooled_features.device)
            combined_features = torch.cat([pooled_features, zero_graph_features], dim=1)
        
        # 特征融合和分类
        features = self.feature_fusion(combined_features)
        logits = self.classifier(features)
        
        return logits, features

# 其他辅助函数保持不变
def validate_and_fix_dataset(dataset):
    if not isinstance(dataset, list):
        print(f"Error: Input dataset is not a list, type: {type(dataset)}")
        return []
    
    fixed_dataset = []
    expected_node_dim = 12  # 4 (one-hot) + 8 (PE)
    expected_edge_dim = 4   # 2 (degree) + 2 (relative PE)
    
    for i, g in enumerate(dataset):
        if not hasattr(g, 'y') or g.y is None:
            print(f"Skipping graph {i}: Missing 'y' attribute or 'y' is None")
            continue
        
        # 保留graph_features但确保是float类型
        if hasattr(g, 'graph_features') and g.graph_features is not None:
            g.graph_features = g.graph_features.float()
            # 处理NaN和Inf
            if torch.isnan(g.graph_features).any() or torch.isinf(g.graph_features).any():
                g.graph_features = torch.where(
                    torch.isnan(g.graph_features) | torch.isinf(g.graph_features),
                    torch.tensor(0.0, dtype=torch.float32),
                    g.graph_features
                )
        
        if hasattr(g, 'x') and g.x is not None:
            # 确保节点特征是float类型
            g.x = g.x.float()
            if g.x.size(1) != expected_node_dim:
                print(f"Error: Graph {i} node features dimension {g.x.size(1)} != expected {expected_node_dim}")
                continue
            if torch.isnan(g.x).any() or torch.isinf(g.x).any():
                g.x = torch.where(
                    torch.isnan(g.x) | torch.isinf(g.x),
                    torch.tensor(0.0, dtype=torch.float32),
                    g.x
                )
        else:
            print(f"Skipping graph {i} (y={g.y.item()})：Missing node features (x)")
            continue
        
        if g.edge_attr is not None:
            # 确保边特征是float类型
            g.edge_attr = g.edge_attr.float()
            if g.edge_attr.size(1) != expected_edge_dim:
                print(f"Error: Graph {i} edge features dimension {g.edge_attr.size(1)} != expected {expected_edge_dim}")
                continue
            if torch.isnan(g.edge_attr).any() or torch.isinf(g.edge_attr).any():
                g.edge_attr = torch.where(
                    torch.isnan(g.edge_attr) | torch.isinf(g.edge_attr),
                    torch.tensor(0.0, dtype=torch.float32),
                    g.edge_attr
                )
        
        if g.edge_index.nelement() > 0 and hasattr(g, 'edge_attr') and g.edge_attr is not None:
            if g.edge_attr.size(0) != g.edge_index.size(1):
                print(f"Fixing graph {i} (y={g.y.item()}): Edge attribute size {g.edge_attr.size(0)} does not match edge index size {g.edge_index.size(1)}")
                num_edges = g.edge_index.size(1)
                if g.edge_attr.size(0) > num_edges:
                    g.edge_attr = g.edge_attr[:num_edges]
                else:
                    padding = torch.zeros((num_edges - g.edge_attr.size(0), g.edge_attr.size(1)), dtype=torch.float32)
                    g.edge_attr = torch.cat([g.edge_attr, padding], dim=0)
        
        fixed_dataset.append(g)
    
    print(f"Original dataset size: {len(dataset)}, Fixed dataset size: {len(fixed_dataset)}")
    return fixed_dataset

# 计算度量的函数保持不变
def calculate_metrics(true_labels, predictions, num_classes=4):
    macro_f1 = f1_score(true_labels, predictions, average='macro', zero_division=0)
    weighted_f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)
    per_class_f1 = f1_score(true_labels, predictions, average=None, labels=range(num_classes), zero_division=0)
    macro_precision = precision_score(true_labels, predictions, average='macro', zero_division=0)
    weighted_precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
    cm = confusion_matrix(true_labels, predictions, labels=range(num_classes))
    report = classification_report(true_labels, predictions, zero_division=0)
    return {
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'per_class_f1': per_class_f1,
        'macro_precision': macro_precision,
        'weighted_precision': weighted_precision,
        'confusion_matrix': cm,
        'classification_report': report
    }

# 分析类特征的函数
def analyze_class_features(dataset, class_idx):
    class_data = [g for g in dataset if hasattr(g, 'y') and g.y is not None and g.y.item() == class_idx]
    if not class_data:
        return {"num_samples": 0, "x_mean": None, "edge_attr_mean": None}
    x_means = torch.stack([g.x.mean(dim=0) for g in class_data]).mean(dim=0)
    edge_attr_means = torch.stack([g.edge_attr.mean(dim=0) for g in class_data if g.edge_attr is not None]).mean(dim=0) if any(g.edge_attr is not None for g in class_data) else None
    return {
        "num_samples": len(class_data),
        "x_mean": x_means.tolist()[:5],
        "edge_attr_mean": edge_attr_means.tolist() if edge_attr_means is not None else None
    }

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Use SelectiveNormalizeFeatures to preserve PE
    transform = Compose([SelectiveNormalizeFeatures(pe_dim=8)])

    print("Loading dataset...")
    try:
        dataset = torch.load('optimized_dataset.pt')
        if not isinstance(dataset, list) or not dataset:
            raise ValueError("Dataset optimized_dataset.pt is empty or invalid")
        print(f"Dataset size: {len(dataset)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit(1)

    print("Validating and fixing dataset...")
    dataset = validate_and_fix_dataset(dataset)

    if dataset is None or not isinstance(dataset, list):
        print("Error: validate_and_fix_dataset returned None or invalid dataset")
        exit(1)

    # 检查图级特征维度
    sample_graph = dataset[0]
    if hasattr(sample_graph, 'graph_features'):
        if sample_graph.graph_features.dim() == 1:
            num_graph_features = sample_graph.graph_features.size(0)
        else:
            num_graph_features = sample_graph.graph_features.size(1)
        print(f"Graph features dimension: {num_graph_features}")
    else:
        num_graph_features = 0
        print("No graph features found in dataset")

    print("Splitting dataset...")
    train_val_data, test_data = train_test_split(
        dataset,
        test_size=0.2,
        random_state=42,
        stratify=[g.y.item() for g in dataset if hasattr(g, 'y') and g.y is not None]
    )
    train_data, val_data = train_test_split(
        train_val_data,
        test_size=0.25,  # 0.25 of 0.8 = 0.2 of total
        random_state=42,
        stratify=[g.y.item() for g in train_val_data if hasattr(g, 'y') and g.y is not None]
    )

    print("Analyzing class distribution...")
    train_labels_all = [g.y.item() for g in train_data if hasattr(g, 'y') and g.y is not None]
    classes, counts = np.unique(train_labels_all, return_counts=True)
    print(f"Class distribution after fixing: {dict(zip(classes, counts))}")

    # Calculate samples per class for HybridLoss
    samples_per_cls = [counts[classes.tolist().index(i)] if i in classes else 0 for i in range(max(classes) + 1)]
    print(f"Samples per class: {samples_per_cls}")

    for cls in range(len(classes)):
        stats = analyze_class_features(train_data, class_idx=cls)
        print(f"Class {cls} training stats: {stats}")

    sample_feature_dim = train_data[0].x.size(1)
    sample_edge_dim = train_data[0].edge_attr.size(1) if train_data[0].edge_attr is not None else 0
    print(f"Node feature dimension: {sample_feature_dim} (expected 12)")
    print(f"Edge feature dimension: {sample_edge_dim} (expected 4)")

    print(f"Training set size: {len(train_data)}")
    print(f"Validation set size: {len(val_data)}")
    print(f"Test set size: {len(test_data)}")

    train_labels = [g.y.item() for g in train_data if hasattr(g, 'y') and g.y is not None]
    val_labels = [g.y.item() for g in val_data if hasattr(g, 'y') and g.y is not None]
    train_classes, train_counts = np.unique(train_labels, return_counts=True)
    val_classes, val_counts = np.unique(val_labels, return_counts=True)
    print(f"Training set class distribution: {dict(zip(train_classes, train_counts))}")
    print(f"Validation set class distribution: {dict(zip(val_classes, val_counts))}")

    print("Applying transformations...")
    try:
        train_data = [transform(g) for g in train_data if hasattr(g, 'y') and g.y is not None]
        val_data = [transform(g) for g in val_data if hasattr(g, 'y') and g.y is not None]
        test_data = [transform(g) for g in test_data if hasattr(g, 'y') and g.y is not None]
    except Exception as e:
        print(f"Error applying transformations: {e}")
        exit(1)

    train_loader = DataLoader(
        train_data,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=False,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_data, 
        batch_size=64, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=False,
        persistent_workers=True
    )
    test_loader = DataLoader(
        test_data, 
        batch_size=64, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=False,
        persistent_workers=True
    )

    num_classes = len(np.unique(train_labels))
    
    # 使用增强模型
    model = EnhancedLightGNN(
        num_node_features=sample_feature_dim,
        num_edge_features=sample_edge_dim,
        num_graph_features=num_graph_features,
        num_classes=num_classes
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 使用改进的损失函数
    criterion = ImprovedHybridLoss(
        samples_per_cls=samples_per_cls,
        num_classes=num_classes,
        feat_dim=128,
        alpha=0.15,   # Focal Loss
        beta=0.75,    # Cross-Entropy (主导)
        gamma=0.1,   # Center Loss
        focal_gamma=2.0
    ).to(device)

    # 使用更高的学习率和不同的优化器设置
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.center_loss.parameters()),
        lr=1e-3,  # 提高学习率
        weight_decay=1e-5,  # 减少权重衰减
        eps=1e-8
    )
    
    # 使用OneCycleLR调度器以获得更好的收敛
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,
        total_steps=100 * len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
    )

    best_macro_precision = 0
    best_epoch = 0
    early_stop_counter = 0
    early_stop_patience = 30

    print("\nStarting training with improved hybrid loss...")
    for epoch in range(100):
        start_time = time.time()
        model.train()
        total_loss = 0
        loss_components = {'focal': 0, 'ce': 0, 'center': 0}
        train_preds = []
        train_labels_epoch = []
        batch_count = 0

        for batch_idx, data in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}')):
            data = data.to(device)
            optimizer.zero_grad()
            try:
                logits, features = model(data)
                loss, components = criterion(logits, features, data.y)
                
                # 梯度裁剪前检查
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: NaN/Inf loss detected in batch {batch_idx}")
                    continue
                
                loss.backward()
                
                # 更温和的梯度裁剪
                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(criterion.center_loss.parameters()), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                for k, v in components.items():
                    loss_components[k] += v
                batch_count += 1
                
                preds = torch.argmax(logits, dim=1)
                train_preds.extend(preds.cpu().numpy())
                train_labels_epoch.extend(data.y.cpu().numpy())
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue

        model.eval()
        val_preds = []
        val_true = []
        val_loss = 0
        val_batch_count = 0

        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                try:
                    logits, features = model(data)
                    loss, _ = criterion(logits, features, data.y)
                    val_loss += loss.item()
                    val_batch_count += 1
                    preds = torch.argmax(logits, dim=1)
                    val_preds.extend(preds.cpu().numpy())
                    val_true.extend(data.y.cpu().numpy())
                except Exception as e:
                    print(f"Error during validation: {e}")
                    continue

        if len(train_preds) > 0 and len(val_preds) > 0:
            train_metrics = calculate_metrics(train_labels_epoch, train_preds, num_classes)
            val_metrics = calculate_metrics(val_true, val_preds, num_classes)
            
            epoch_time = time.time() - start_time
            avg_train_loss = total_loss / batch_count if batch_count > 0 else float('inf')
            avg_val_loss = val_loss / val_batch_count if val_batch_count > 0 else float('inf')
            
            # 计算各损失分量的平均值
            avg_loss_components = {k: v/batch_count for k, v in loss_components.items()}

            print(f"\nEpoch {epoch+1} | Time: {epoch_time:.1f}s | LR: {optimizer.param_groups[0]['lr']:.2e}")
            print(f"Training loss: {avg_train_loss:.4f} | Validation loss: {avg_val_loss:.4f}")
            print(f"Loss components - Focal: {avg_loss_components['focal']:.4f}, CE: {avg_loss_components['ce']:.4f}, Center: {avg_loss_components['center']:.4f}")
            print(f"Training macro F1: {train_metrics['macro_f1']:.4f} | Training weighted F1: {train_metrics['weighted_f1']:.4f}")
            print(f"Training macro precision: {train_metrics['macro_precision']:.4f} | Training weighted precision: {train_metrics['weighted_precision']:.4f}")
            print(f"Training per-class F1: {train_metrics['per_class_f1']}")
            print(f"Validation macro F1: {val_metrics['macro_f1']:.4f} | Validation weighted F1: {val_metrics['weighted_f1']:.4f}")
            print(f"Validation macro precision: {val_metrics['macro_precision']:.4f} | Validation weighted precision: {val_metrics['weighted_precision']:.4f}")
            print(f"Validation per-class F1: {val_metrics['per_class_f1']}")
            print("Validation confusion matrix:")
            print(val_metrics['confusion_matrix'])

            if val_metrics['macro_precision'] > best_macro_precision:
                best_macro_precision = val_metrics['macro_precision']
                best_epoch = epoch + 1
                early_stop_counter = 0
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_macro_precision': best_macro_precision
                }, 'best_modelp.pth')
                print(f"Saved new best model with validation macro precision: {best_macro_precision:.4f}")
            else:
                early_stop_counter += 1
                if early_stop_counter >= early_stop_patience:
                    print(f"Early stopping triggered! Best epoch: {best_epoch}")
                    break
        else:
            print(f"Epoch {epoch+1}: No valid predictions, skipping metrics calculation...")

    if os.path.exists('best_modelp.pth'):
        checkpoint = torch.load('best_modelp.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"\nLoading best model from epoch {best_epoch} for testing")
        print(f"Best validation macro precision: {checkpoint['best_macro_precision']:.4f}")

    model.eval()
    test_preds = []
    test_true = []
    test_loss = 0
    test_batch_count = 0

    print("\nEvaluating on test set...")
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Testing"):
            data = data.to(device)
            try:
                logits, features = model(data)
                loss, _ = criterion(logits, features, data.y)
                test_loss += loss.item()
                test_batch_count += 1
                preds = torch.argmax(logits, dim=1)
                test_preds.extend(preds.cpu().numpy())
                test_true.extend(data.y.cpu().numpy())
            except Exception as e:
                print(f"Error during testing: {e}")
                continue

    if len(test_preds) > 0:
        test_metrics = calculate_metrics(test_true, test_preds, num_classes)
        avg_test_loss = test_loss / test_batch_count if test_batch_count > 0 else float('inf')
        
        print("\n" + "="*50)
        print("=== Final Test Results ===")
        print("="*50)
        print(f"Test loss: {avg_test_loss:.4f}")
        print(f"Test macro F1: {test_metrics['macro_f1']:.4f}")
        print(f"Test weighted F1: {test_metrics['weighted_f1']:.4f}")
        print(f"Test macro precision: {test_metrics['macro_precision']:.4f}")
        print(f"Test weighted precision: {test_metrics['weighted_precision']:.4f}")
        print(f"Test per-class F1: {test_metrics['per_class_f1']}")
        print("\nTest confusion matrix:")
        print(test_metrics['confusion_matrix'])
        print("\nTest classification report:")
        print(test_metrics['classification_report'])
        
        # 保存测试结果
        import json
        results = {
            'test_loss': avg_test_loss,
            'test_macro_f1': test_metrics['macro_f1'],
            'test_weighted_f1': test_metrics['weighted_f1'],
            'test_macro_precision': test_metrics['macro_precision'],
            'test_weighted_precision': test_metrics['weighted_precision'],
            'test_per_class_f1': test_metrics['per_class_f1'].tolist(),
            'confusion_matrix': test_metrics['confusion_matrix'].tolist()
        }
        with open('test_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        print("\nTest results saved to test_results.json")
    else:
        print("No valid test predictions available.")