import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import random

# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

# 自定义数据集
class SpliceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 6维向量编码
def encode_base(base, is_start=False, is_end=False):
    """编码单个碱基或特殊位点为 6 维向量"""
    if is_start:
        return np.array([0, 0, 0, 0, 1, 0], dtype=np.float32)  # 剪切开始
    if is_end:
        return np.array([0, 0, 0, 0, 0, 1], dtype=np.float32)  # 剪切结束
    base_to_idx = {
        'A': [1, 0, 0, 0, 0, 0],
        'C': [0, 1, 0, 0, 0, 0],
        'G': [0, 0, 1, 0, 0, 0],
        'T': [0, 0, 0, 1, 0, 0]
    }
    return np.array(base_to_idx.get(base, [0, 0, 0, 0, 0, 0]), dtype=np.float32)

def encode_sequence(newup, newaa, newdown, max_newaa_len=400):
    """合并并编码 newup (50), S, newaa (补齐到 400), E, newdown (50)"""
    if not isinstance(newup, str) or not isinstance(newaa, str) or not isinstance(newdown, str):
        raise ValueError(f"Invalid sequence input: newup={newup}, newaa={newaa}, newdown={newdown}")
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
    return np.array(encoded)  # 形状：(502, 6)

# 读取数据并检查异常
def load_data(file_path):
    """读取 bubble_relativesuppa_comparison.tsv 并过滤 ground truth 为 [0, 0] 的数据"""
    print(f"Loading data from {file_path}...")
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist!")
        return None
    
    try:
        df = pd.read_csv(file_path, sep='\t', header=0, encoding='utf-8')
        print(f"Loaded {len(df)} rows, columns: {df.shape[1]}")
        
        # 检查是否为空DataFrame
        if df.empty:
            print("Error: Loaded DataFrame is empty!")
            return None
        
        # 检查必需列是否存在
        required_columns = ['newup', 'newaa', 'newdown', 'start_diff', 'end_diff']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            print(f"Available columns: {list(df.columns)}")
            return None
        
        # 检查数据中的 nan 或 inf
        if df[['newup', 'newaa', 'newdown', 'start_diff', 'end_diff']].isnull().any().any():
            print("Warning: Found NaN values in data. Removing rows with NaN...")
            df = df.dropna(subset=['newup', 'newaa', 'newdown', 'start_diff', 'end_diff'])
            print(f"After removing NaN, remaining rows: {len(df)}")
        
        if np.isinf(df[['start_diff', 'end_diff']].values).any():
            print("Warning: Found Inf values in start_diff or end_diff. Removing rows with Inf...")
            df = df[~np.isinf(df[['start_diff', 'end_diff']].values).any(axis=1)]
            print(f"After removing Inf, remaining rows: {len(df)}")
        
        # 进一步检查序列是否包含非法字符
        valid_bases = set('ACGT')
        invalid_indices = []
        for idx, row in df.iterrows():
            seq = str(row['newup']) + str(row['newaa']) + str(row['newdown'])
            if not all(c in valid_bases for c in seq):
                print(f"Warning: Invalid characters in sequence at index {idx}: {seq[:50]}...")
                invalid_indices.append(idx)
        
        if invalid_indices:
            df = df.drop(index=invalid_indices)
            print(f"Removed {len(invalid_indices)} rows with invalid characters")
        
        # 最终检查
        if df.empty:
            print("Error: No valid data remaining after filtering!")
            return None
        
        print("Statistics of start_diff and end_diff after filtering:")
        print(df[['start_diff', 'end_diff']].describe())
        
        return df[['newup', 'newaa', 'newdown', 'start_diff', 'end_diff']]
        
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

# 数据预处理
def prepare_data(df, max_newaa_len=400):
    """编码序列并准备输入和输出"""
    print("Preparing data...")
    
    # 先编码原始数据
    X = []
    y = []
    valid_rows = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Encoding sequences"):
        try:
            encoded_seq = encode_sequence(row['newup'], row['newaa'], row['newdown'], max_newaa_len)
            X.append(encoded_seq)
            y.append([row['start_diff'], row['end_diff']])
            valid_rows.append(row)
        except ValueError as e:
            print(f"Error at index {idx}: {e}")
            continue
    
    if not X:
        raise ValueError("No sequences could be encoded successfully!")
    
    X = np.array(X)  # 形状：(样本数, 502, 6)
    y = np.array(y)  # 形状：(样本数, 2)
    valid_df = pd.DataFrame(valid_rows)
    
    # 过采样大偏差点
    large_deviation_mask = (np.abs(y[:, 0]) > 5) | (np.abs(y[:, 1]) > 5)
    large_deviation_indices = np.where(large_deviation_mask)[0]
    
    if len(large_deviation_indices) > 0:
        # 复制大偏差样本两次
        X_large = X[large_deviation_indices]
        y_large = y[large_deviation_indices]
        
        # 合并原始数据和过采样数据
        X = np.concatenate([X, X_large, X_large], axis=0)
        y = np.concatenate([y, y_large, y_large], axis=0)
        
        print(f"Oversampled {len(large_deviation_indices)} large deviation samples (duplicated 2 times)")
    
    # 检查 X 和 y 中的 nan 或 inf
    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("Encoded sequences (X) contain NaN or Inf values")
    if np.isnan(y).any() or np.isinf(y).any():
        raise ValueError("Labels (y) contain NaN or Inf values")
    
    print(f"Data prepared: X shape {X.shape}, y shape {y.shape}")
    return X, y

# Attention-Based CNN 模型
class SpliceCNNAttention(nn.Module):
    def __init__(self, input_channels=6, seq_len=502):
        super(SpliceCNNAttention, self).__init__()
        # CNN 层
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
        self.dropout = nn.Dropout(0.1)  # 降低 Dropout 率
        
        # 计算卷积后的序列长度
        conv_len = seq_len // 2  # 第一次 MaxPool1d
        conv_len = conv_len // 2  # 第二次 MaxPool1d
        self.conv_out_dim = 512 * conv_len  # 展平后的维度
        
        # Attention 层
        self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=8, dropout=0.1)  # 减少头数
        self.ln = nn.LayerNorm(512)  # 添加 LayerNorm
        
        # 全连接层
        self.fc1 = nn.Linear(self.conv_out_dim, 128)
        self.fc2 = nn.Linear(128, 2)  # 输出 start_diff 和 end_diff
        
        # Xavier 初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """对卷积层和全连接层应用 Xavier 初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # x: (batch_size, seq_len, input_channels) -> (batch_size, input_channels, seq_len)
        x = x.transpose(1, 2)  # (batch_size, 6, 502)
        
        # CNN 层
        x = self.relu(self.bn1(self.conv1(x)))  # (batch_size, 64, 502)
        x = self.pool(x)  # (batch_size, 64, 251)
        x = self.relu(self.bn2(self.conv2(x)))  # (batch_size, 128, 251)
        x = self.pool(x)  # (batch_size, 128, 125)
        x = self.relu(self.bn3(self.conv3(x)))  # (batch_size, 256, 125)
        x = self.relu(self.bn4(self.conv4(x)))  # (batch_size, 512, 125)
        
        # 转换为 Attention 格式：(seq_len, batch_size, embed_dim)
        x = x.permute(2, 0, 1)  # (125, batch_size, 512)
        x = self.ln(x)  # LayerNorm
        
        # Attention 层
        attn_output, _ = self.attention(x, x, x)  # (125, batch_size, 512)
        
        # 展平并通过全连接层
        x = attn_output.permute(1, 0, 2).contiguous().view(-1, self.conv_out_dim)  # (batch_size, 512 * 125)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # (batch_size, 2)
        return x

# 计算准确率
def calculate_accuracy(pred_y, true_y, tolerance=0.5):
    """计算预测值与真实值在容差范围内的准确率"""
    errors = np.abs(pred_y - true_y)
    correct = np.all(errors <= tolerance, axis=1)
    accuracy = np.mean(correct)
    return accuracy

# 训练模型
def train_model(model, train_loader, val_loader, epochs=30, device='cuda' if torch.cuda.is_available() else 'cpu'):
    print(f"Training model on {device}...")
    model = model.to(device)
    criterion = nn.HuberLoss(delta=1.0)  # 使用更稳定的 Huber Loss
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-5)  # 使用 AdamW 和更低学习率
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    best_val_loss = float('inf')
    
    epoch_metrics = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_preds, train_true = [], []
        for batch_idx, (X_batch, y_batch) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            
            # 检查 outputs 和 y_batch 是否包含 nan
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                print(f"Warning: NaN or Inf detected in model outputs at batch {batch_idx}")
                continue
            if torch.isnan(y_batch).any() or torch.isinf(y_batch).any():
                print(f"Warning: NaN or Inf detected in labels at batch {batch_idx}")
                continue
            
            loss = criterion(outputs, y_batch)
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN or Inf detected in loss at batch {batch_idx}")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 更严格的梯度裁剪
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
            train_preds.append(outputs.detach().cpu().numpy())
            train_true.append(y_batch.cpu().numpy())
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        train_preds = np.concatenate(train_preds, axis=0)
        train_true = np.concatenate(train_true, axis=0)
        train_accuracy = calculate_accuracy(train_preds, true_y=train_true, tolerance=0.5)
        train_accuracies.append(train_accuracy)
        
        model.eval()
        val_loss = 0
        val_preds, val_true = [], []
        with torch.no_grad():
            for X_batch, y_batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                val_preds.append(outputs.detach().cpu().numpy())
                val_true.append(y_batch.cpu().numpy())
        
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        val_preds = np.concatenate(val_preds, axis=0)
        val_true = np.concatenate(val_true, axis=0)
        val_accuracy = calculate_accuracy(val_preds, true_y=val_true, tolerance=0.5)
        val_accuracies.append(val_accuracy)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"Train Accuracy (tolerance=0.5): {train_accuracy:.4f}, Val Accuracy (tolerance=0.5): {val_accuracy:.4f}")
        
        epoch_metrics.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'learning_rate': optimizer.param_groups[0]['lr']
        })
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss and not np.isnan(val_loss):
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_splice_diff_cnn_attention.pt')
            print(f"Best model saved at epoch {epoch+1} with Val Loss: {val_loss:.4f}")
    
    pd.DataFrame(epoch_metrics).to_csv('epoch_metrics.csv', index=False)
    print("Epoch metrics saved as epoch_metrics.csv")
    
    return train_losses, val_losses, train_accuracies, val_accuracies

# 测试集预测并保存数据
def evaluate_model(model, X_test, y_test, scaler, device='cuda' if torch.cuda.is_available() else 'cpu'):
    print("Evaluating model on test set...")
    model.eval()
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        pred_y = model(X_test).cpu().numpy()
    pred_y = scaler.inverse_transform(pred_y)
    true_y = scaler.inverse_transform(y_test)
    
    test_accuracy_05 = calculate_accuracy(pred_y, true_y, tolerance=0.5)
    print(f"Test Accuracy (tolerance=0.5): {test_accuracy_05:.4f}")
    
    print(f"Sample Prediction (first test sample):")
    print(f"True Splice Site Offset (start): {true_y[0, 0]:.2f}, Predicted Splice Site Offset (start): {pred_y[0, 0]:.2f}")
    print(f"True Splice Site Offset (end): {true_y[0, 1]:.2f}, Predicted Splice Site Offset (end): {pred_y[0, 1]:.2f}")
    
    test_predictions = pd.DataFrame({
        'true_start_offset': true_y[:, 0],
        'pred_start_offset': pred_y[:, 0],
        'true_end_offset': true_y[:, 1],
        'pred_end_offset': pred_y[:, 1]
    })
    test_predictions.to_csv('test_predictions.csv', index=False)
    print("Test predictions saved as test_predictions.csv")

# 预测新序列
def predict_new_sequences(newup, newaa, newdown, model, scaler, max_newaa_len=400, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """预测新输入序列的 start_diff 和 end_diff"""
    print("Predicting for new sequences...")
    model.eval()
    encoded_seq = encode_sequence(newup, newaa, newdown, max_newaa_len)
    X_new = torch.tensor([encoded_seq], dtype=torch.float32).to(device)
    with torch.no_grad():
        pred_y = model(X_new).cpu().numpy()
    pred_y = scaler.inverse_transform(pred_y)[0]
    
    print(f"Input Sequences: newup={newup[:10]}..., newaa={newaa[:10]}..., newdown={newdown[:10]}...")
    print(f"Predicted Splice Site Offset (start): {pred_y[0]:.2f}")
    print(f"Predicted Splice Site Offset (end): {pred_y[1]:.2f}")

# 主函数
def main(file_path="humancomparison.tsv"):
    # 加载数据
    df = load_data(file_path)
    
    # 检查数据是否成功加载
    if df is None:
        print("Error: Failed to load data. Please check the file path and format.")
        return
    
    if df.empty:
        print("Error: Input file is empty or all rows have zero offsets.")
        return
    
    # 预处理
    max_newaa_len = 400
    total_seq_len = 50 + 1 + max_newaa_len + 1 + 50  # 502
    
    try:
        X, y = prepare_data(df, max_newaa_len)
    except Exception as e:
        print(f"Error in data preparation: {e}")
        return
    
    # 分割训练和测试集
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
    
    # 标准化输出
    print("Standardizing output...")
    scaler = StandardScaler()
    y_train = scaler.fit_transform(y_train)
    y_test = scaler.transform(y_test)
    
    # 检查标准化后的 y_train 和 y_test
    if np.isnan(y_train).any() or np.isinf(y_train).any():
        raise ValueError("Standardized y_train contains NaN or Inf values")
    if np.isnan(y_test).any() or np.isinf(y_test).any():
        raise ValueError("Standardized y_test contains NaN or Inf values")
    
    # 创建数据集和加载器
    print("Creating data loaders...")
    train_dataset = SpliceDataset(X_train, y_train)
    val_dataset = SpliceDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    
    # 初始化模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SpliceCNNAttention(input_channels=6, seq_len=total_seq_len)
    
    # 训练模型
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(model, train_loader, val_loader, epochs=30, device=device)
    
    # 评估模型
    evaluate_model(model, X_test, y_test, scaler, device)
    
    # 保存最终模型和标准化器
    torch.save(model.state_dict(), 'splice_diff_cnn_attention.pt')
    np.save('scaler.npy', scaler)
    print("Final model saved as splice_diff_cnn_attention.pt")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()