#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简单的预测示例
"""
import torch
import numpy as np
from predictionmodel import ModelPredictor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import json
model_path='best_model2.pth'
device='cpu'
dataset='mousedataset.pt'

# ===================== 方法1：最简单的使用 =====================
def simple_prediction():
    """最简单的预测示例"""
    
    # 1. 初始化预测器
    predictor = ModelPredictor(
        model_path= model_path,  # 您的模型路径
        device='cpu'  # 或 'cuda'
    )
    
    # 2. 进行预测
    results = predictor.predict(
        dataset=dataset,  # 新数据集路径
        return_probs=True  # 返回概率值
    )
    
    # 3. 查看结果
    print("预测标签:", results['labels'][:10])  # 前10个预测
    print("置信度:", results['confidence'][:10])
    
    # 4. 保存结果
    predictor.save_predictions(results, 'predictions.json')

# ===================== 方法2：批量预测多个文件 =====================
def batch_prediction():
    """批量预测多个数据文件"""
    
    predictor = ModelPredictor('best_model2.pth')
    
    # 多个数据文件
    data_files = [
        'dataset1.pt',
        'dataset2.pt',
        'dataset3.pt'
    ]
    
    all_results = []
    
    for data_file in data_files:
        print(f"\nProcessing {data_file}...")
        results = predictor.predict(data_file, return_probs=True)
        
        # 添加文件信息
        results['source_file'] = data_file
        all_results.append(results)
        
        # 保存单个文件的结果
        output_name = data_file.replace('.pt', '_predictions.csv')
        predictor.save_predictions(results, output_name)
    
    print(f"\nProcessed {len(data_files)} files")
    return all_results

# ===================== 方法3：使用TTA提高准确率 =====================
def prediction_with_tta():
    """使用测试时增强提高预测准确率"""
    
    predictor = ModelPredictor('best_model2.pth')
    
    # 使用TTA（会稍微慢一些，但准确率更高）
    results = predictor.predict_with_tta(
        dataset=dataset,
        n_augmentations=5  # 做5次增强并平均
    )
    
    print("TTA预测完成")
    print(f"平均置信度: {np.mean(results['confidence']):.4f}")
    
    # 找出低置信度样本
    low_conf_indices = np.where(results['confidence'] < 0.5)[0]
    print(f"低置信度样本数: {len(low_conf_indices)}")
    
    return results

# ===================== 方法4：分析预测结果 =====================
def analyze_predictions():
    """分析预测结果"""
    
    predictor = ModelPredictor('best_model2.pth')
    results = predictor.predict(dataset, return_probs=True)
    
    # 1. 类别分布
    print("\n=== 类别分布 ===")
    unique, counts = np.unique(results['predictions'], return_counts=True)
    for cls, count in zip(unique, counts):
        percentage = count / len(results['predictions']) * 100
        print(f"类别 {cls}: {count} ({percentage:.2f}%)")
    
    # 2. 置信度分析
    print("\n=== 置信度分析 ===")
    confidence = results['confidence']
    print(f"平均置信度: {np.mean(confidence):.4f}")
    print(f"中位数置信度: {np.median(confidence):.4f}")
    print(f"最低置信度: {np.min(confidence):.4f}")
    print(f"最高置信度: {np.max(confidence):.4f}")
    
    # 3. 不确定样本（置信度低于阈值）
    threshold = 0.6
    uncertain_mask = confidence < threshold
    uncertain_count = np.sum(uncertain_mask)
    print(f"\n置信度低于{threshold}的样本: {uncertain_count} ({uncertain_count/len(confidence)*100:.2f}%)")
    
    # 4. 混淆样本分析（概率接近的类别）
    probs = results['probabilities']
    top2_diff = []
    for prob in probs:
        sorted_probs = np.sort(prob)[::-1]
        diff = sorted_probs[0] - sorted_probs[1]
        top2_diff.append(diff)
    
    top2_diff = np.array(top2_diff)
    confused_samples = np.sum(top2_diff < 0.2)
    print(f"混淆样本（top2概率差<0.2）: {confused_samples} ({confused_samples/len(probs)*100:.2f}%)")
    
    return results

# ===================== 方法5：创建预测报告 =====================
def create_prediction_report():
    """创建详细的预测报告"""
    
    import matplotlib.pyplot as plt
    
    predictor = ModelPredictor('best_model1.pth')
    results = predictor.predict(dataset, return_probs=True)
    
    # 创建DataFrame
    df = pd.DataFrame({
        'prediction': results['predictions'],
        'label': results['labels'],
        'confidence': results['confidence'],
        'prob_A3': results['probabilities'][:, 0],
        'prob_A5': results['probabilities'][:, 1],
        'prob_SE': results['probabilities'][:, 2],
        'prob_RI': results['probabilities'][:, 3]
    })
    
    # 生成报告
    report = f"""
    ========== 预测报告 ==========
    
    总样本数: {len(df)}
    
    类别分布:
    {df['label'].value_counts()}
    
    置信度统计:
    - 平均值: {df['confidence'].mean():.4f}
    - 标准差: {df['confidence'].std():.4f}
    - 最小值: {df['confidence'].min():.4f}
    - 最大值: {df['confidence'].max():.4f}
    
    各类别平均置信度:
    {df.groupby('label')['confidence'].mean()}
    
    低置信度样本（<0.5）:
    {len(df[df['confidence'] < 0.5])} 个
    
    高置信度样本（>0.9）:
    {len(df[df['confidence'] > 0.9])} 个
    """
    
    print(report)
    
    # 保存详细结果
    df.to_csv('prediction_report.csv', index=False)
    
    # 可视化置信度分布
    plt.figure(figsize=(10, 6))
    plt.hist(df['confidence'], bins=50, edgecolor='black')
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.title('Prediction Confidence Distribution')
    plt.savefig('confidence_distribution.png')
    plt.close()
    
    print("\n报告已保存到 prediction_report.csv")
    print("置信度分布图已保存到 confidence_distribution.png")
    
    return df

# ===================== 方法6：处理实时数据流 =====================
def predict_single_sample(predictor, graph_data):
    """
    预测单个样本
    
    Args:
        predictor: ModelPredictor实例
        graph_data: 单个Data对象
    """
    # 将单个样本包装成列表
    results = predictor.predict([graph_data], return_probs=True)
    
    prediction = results['predictions'][0]
    label = results['labels'][0]
    confidence = results['confidence'][0]
    probs = results['probabilities'][0]
    
    return {
        'prediction': prediction,
        'label': label,
        'confidence': confidence,
        'probabilities': probs
    }

# ===================== 方法7：模型评估（需要真实标签） =====================
def evaluate_model():
    """模型评估 - 计算准确率和各类别指标"""
    
    predictor = ModelPredictor(model_path, device=device)
    
    # 进行预测
    results = predictor.predict(dataset, return_probs=True)
    
    # 从数据集中直接获取真实标签
    print("正在从数据集中提取真实标签...")
    try:
        dataset_obj = torch.load(dataset, map_location='cpu')
        y_true_from_dataset = []
        
        for data in dataset_obj:
            if hasattr(data, 'y'):
                label = data.y.item() if torch.is_tensor(data.y) else data.y
                y_true_from_dataset.append(label)
            else:
                print("错误：数据集中的样本没有标签字段 'y'")
                return None
        
        y_true = np.array(y_true_from_dataset)
        print(f"从数据集提取的真实标签数量: {len(y_true)}")
        
    except Exception as e:
        print(f"从数据集提取标签时出错: {e}")
        # 回退到使用预测结果中的标签
        y_true = results['true_labels'] if 'true_labels' in results else results['labels']
        if y_true is None:
            print("警告：无法获取真实标签，无法进行评估")
            return None
        y_true = np.array(y_true)
    
    # 获取预测标签
    y_pred = np.array(results['predictions'])
    
    # 检查数据长度是否匹配
    if len(y_true) != len(y_pred):
        print(f"错误：真实标签数量({len(y_true)})与预测标签数量({len(y_pred)})不匹配")
        return None
    
    # 打印调试信息
    print(f"\n调试信息:")
    print(f"真实标签类型: {type(y_true)}, 形状: {y_true.shape}")
    print(f"预测标签类型: {type(y_pred)}, 形状: {y_pred.shape}")
    print(f"真实标签示例: {y_true[:10]}")
    print(f"预测标签示例: {y_pred[:10]}")
    print(f"真实标签唯一值: {np.unique(y_true)}")
    print(f"预测标签唯一值: {np.unique(y_pred)}")
    
    # 定义标签映射（基于您的数据集生成逻辑）
    LABEL_MAP = {"A3": 0, "A5": 1, "SE": 2, "RI": 3}
    IDX_TO_LABEL = {0: "A3", 1: "A5", 2: "SE", 3: "RI"}
    
    # 处理标签类型不一致的情况
    if y_true.dtype == 'object' or (len(y_true) > 0 and isinstance(y_true[0], str)):
        print("真实标签为字符串，转换为数字...")
        y_true_numeric = np.array([LABEL_MAP.get(label, -1) for label in y_true])
        if -1 in y_true_numeric:
            unknown_labels = set(y_true[y_true_numeric == -1])
            print(f"警告：发现未知真实标签: {unknown_labels}")
        y_true = y_true_numeric
    
    if y_pred.dtype == 'object' or (len(y_pred) > 0 and isinstance(y_pred[0], str)):
        print("预测标签为字符串，转换为数字...")
        y_pred_numeric = np.array([LABEL_MAP.get(label, -1) for label in y_pred])
        if -1 in y_pred_numeric:
            unknown_labels = set(y_pred[y_pred_numeric == -1])
            print(f"警告：发现未知预测标签: {unknown_labels}")
        y_pred = y_pred_numeric
    
    # 过滤掉无效标签
    valid_mask = (y_true >= 0) & (y_true < 4) & (y_pred >= 0) & (y_pred < 4)
    if not np.all(valid_mask):
        invalid_count = np.sum(~valid_mask)
        print(f"警告：过滤掉 {invalid_count} 个无效样本")
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]
    
    if len(y_true) == 0:
        print("错误：没有有效样本进行评估")
        return None
    
    # 使用预定义的类别名称
    class_names = ["A3", "A5", "SE", "RI"]
    
    # 再次检查是否真的不同
    print(f"\n关键检查:")
    print(f"标签完全相同的样本数: {np.sum(y_true == y_pred)}")
    print(f"标签不同的样本数: {np.sum(y_true != y_pred)}")
    print(f"前20个样本比较:")
    for i in range(min(20, len(y_true))):
        status = "✓" if y_true[i] == y_pred[i] else "✗"
        print(f"  样本{i:2d}: 真实={IDX_TO_LABEL[y_true[i]]}({y_true[i]}), 预测={IDX_TO_LABEL[y_pred[i]]}({y_pred[i]}) {status}")
    
    # 计算总体准确率
    overall_accuracy = accuracy_score(y_true, y_pred)
    
    print("=" * 60)
    print("模型评估结果")
    print("=" * 60)
    print(f"总体准确率: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
    print()
    
    # 计算各类别详细指标
    print("详细分类报告:")
    print("-" * 50)
    report = classification_report(
        y_true, y_pred, 
        target_names=class_names,
        digits=4,
        zero_division=0
    )
    print(report)
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    print("\n混淆矩阵:")
    print("-" * 30)
    
    # 创建混淆矩阵的DataFrame用于更好的显示
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    print(cm_df)
    
    # 计算各类别准确率
    print("\n各类别准确率:")
    print("-" * 30)
    class_accuracies = {}
    for i, class_name in enumerate(class_names):
        if cm[i].sum() > 0:  # 避免除零错误
            class_acc = cm[i, i] / cm[i].sum()
            class_accuracies[class_name] = class_acc
            print(f"{class_name}: {class_acc:.4f} ({class_acc*100:.2f}%)")
        else:
            print(f"{class_name}: 无样本")
    
    # 计算各类别样本数
    print("\n各类别样本数:")
    print("-" * 30)
    unique_labels, counts = np.unique(y_true, return_counts=True)
    for label, count in zip(unique_labels, counts):
        class_name = class_names[int(label)]
        percentage = count / len(y_true) * 100
        print(f"{class_name}: {count} ({percentage:.2f}%)")
    
    # 保存评估结果
    eval_results = {
        'overall_accuracy': overall_accuracy,
        'class_accuracies': class_accuracies,
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'predictions': y_pred.tolist(),
        'true_labels': y_true.tolist(),
        'confidence': results['confidence'].tolist(),
        'probabilities': results['probabilities'].tolist()
    }
    
    # 保存到文件
    import json
    with open('evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=2)
    
    # 创建评估报告CSV
    eval_df = pd.DataFrame({
        'true_label': y_true,
        'predicted_label': y_pred,
        'true_label_name': [class_names[int(i)] for i in y_true],
        'predicted_label_name': [class_names[int(i)] for i in y_pred],
        'correct': (y_true == y_pred).astype(int),
        'confidence': results['confidence']
    })
    eval_df.to_csv('evaluation_details.csv', index=False)
    
    # 可视化混淆矩阵
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("\n文件保存完成:")
        print("- evaluation_results.json: 完整评估结果")
        print("- evaluation_details.csv: 详细预测结果")
        print("- confusion_matrix.png: 混淆矩阵图")
        
    except ImportError:
        print("\n注意：matplotlib/seaborn未安装，跳过可视化")
        print("文件保存完成:")
        print("- evaluation_results.json: 完整评估结果") 
        print("- evaluation_details.csv: 详细预测结果")
    
# ===================== 方法8：数据泄漏诊断 =====================
def diagnose_data_leakage():
    """诊断可能的数据泄漏问题"""
    
    print("=" * 60)
    print("数据泄漏诊断")
    print("=" * 60)
    
    # 加载数据集检查真实标签
    try:
        dataset_obj = torch.load(dataset, map_location='cpu')
        print(f"数据集类型: {type(dataset_obj)}")
        print(f"数据集大小: {len(dataset_obj)}")
        
        if hasattr(dataset_obj, '__getitem__'):
            sample = dataset_obj[0]
            print(f"样本类型: {type(sample)}")
            if hasattr(sample, 'y'):
                print(f"标签字段存在: y = {sample.y}")
            else:
                print("警告：样本中没有找到标签字段 'y'")
        
        # 统计真实标签分布
        true_labels = []
        for i, data in enumerate(dataset_obj):
            if hasattr(data, 'y'):
                true_labels.append(data.y.item() if torch.is_tensor(data.y) else data.y)
            if i > 10:  # 只检查前几个样本
                break
        
        if true_labels:
            print(f"前几个真实标签: {true_labels}")
            unique_true, counts_true = np.unique(true_labels, return_counts=True)
            print(f"真实标签分布: {dict(zip(unique_true, counts_true))}")
    except Exception as e:
        print(f"加载数据集时出错: {e}")
    
    # 进行预测并检查
    predictor = ModelPredictor(model_path, device=device)
    results = predictor.predict(dataset, return_probs=True)
    
    # 分析预测结果
    y_pred = np.array(results['predictions'])
    confidence = np.array(results['confidence'])
    probabilities = np.array(results['probabilities'])
    
    print("\n预测结果分析:")
    print("-" * 30)
    print(f"预测标签分布: {dict(zip(*np.unique(y_pred, return_counts=True)))}")
    print(f"平均置信度: {np.mean(confidence):.4f}")
    print(f"置信度标准差: {np.std(confidence):.4f}")
    print(f"最小置信度: {np.min(confidence):.4f}")
    print(f"最大置信度: {np.max(confidence):.4f}")
    
    # 检查是否所有预测都过于确定
    high_conf_count = np.sum(confidence > 0.95)
    very_high_conf_count = np.sum(confidence > 0.99)
    print(f"置信度 > 0.95 的样本数: {high_conf_count} ({high_conf_count/len(confidence)*100:.2f}%)")
    print(f"置信度 > 0.99 的样本数: {very_high_conf_count} ({very_high_conf_count/len(confidence)*100:.2f}%)")
    
    # 检查概率分布
    print(f"\n概率分布分析:")
    print(f"概率矩阵形状: {probabilities.shape}")
    
    # 检查是否有过于极端的概率
    max_probs = np.max(probabilities, axis=1)
    extreme_prob_count = np.sum(max_probs > 0.99)
    print(f"最大概率 > 0.99 的样本数: {extreme_prob_count} ({extreme_prob_count/len(probabilities)*100:.2f}%)")
    
    # 分析每个类别的平均概率
    for i in range(probabilities.shape[1]):
        avg_prob = np.mean(probabilities[:, i])
        print(f"类别 {i} 的平均概率: {avg_prob:.4f}")
    
    # 检查是否模型输出过于集中在某些类别
    predicted_class_dist = np.bincount(y_pred)
    print(f"\n预测类别分布: {predicted_class_dist}")
    if len(predicted_class_dist) < 4:  # 如果没有预测到所有类别
        print("警告：模型没有预测到所有类别！")
    
    return results

# ===================== 方法9：增强的评估（包含诊断） =====================
def evaluate_model_enhanced():
    """增强的模型评估，包含数据泄漏检测"""
    
    print("=" * 60)
    print("增强模型评估")
    print("=" * 60)
    
    # 首先运行诊断
    print("步骤 1: 运行数据泄漏诊断...")
    diagnose_data_leakage()
    
    print("\n" + "=" * 60)
    print("步骤 2: 标准评估...")
    print("=" * 60)
    
    # 运行标准评估
    return evaluate_model()

# ===================== 主函数 =====================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("使用方法:")
        print("python classifyAS.py simple     # 简单预测")
        print("python classifyAS.py batch      # 批量预测")
        print("python classifyAS.py tta        # 使用TTA")
        print("python classifyAS.py analyze    # 分析结果")
        print("python classifyAS.py report     # 生成报告")
        print("python classifyAS.py evaluate   # 模型评估（需要真实标签）")
        print("python classifyAS.py diagnose   # 数据泄漏诊断")
        print("python classifyAS.py enhanced   # 增强评估（评估+诊断）")
        print("\n默认运行简单预测...")
        simple_prediction()
    else:
        mode = sys.argv[1]
        
        if mode == 'simple':
            simple_prediction()
        elif mode == 'batch':
            batch_prediction()
        elif mode == 'tta':
            prediction_with_tta()
        elif mode == 'analyze':
            analyze_predictions()
        elif mode == 'report':
            create_prediction_report()
        elif mode == 'evaluate':
            evaluate_model()
        elif mode == 'diagnose':
            diagnose_data_leakage()
        elif mode == 'enhanced':
            evaluate_model_enhanced()
        else:
            print(f"未知模式: {mode}")
            print("可用模式: simple, batch, tta, analyze, report, evaluate, diagnose, enhanced")