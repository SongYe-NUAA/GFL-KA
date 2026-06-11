#!/usr/bin/env python3
"""
🔬 特征有效性分析工具

用于分析不同特征组合的有效性，帮助找到最优的特征选择策略
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import mutual_info_score
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

class FeatureEffectivenessAnalyzer:
    """🔬 特征有效性分析器"""
    
    def __init__(self, save_dir: str = "analysis_results"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        # 存储分析结果
        self.correlation_history = []
        self.effectiveness_scores = {}
        self.redundancy_analysis = {}
        
    def analyze_feature_effectiveness(self, 
                                    features_dict: Dict[str, torch.Tensor], 
                                    target_iou: torch.Tensor,
                                    step: int = 0) -> Dict[str, float]:
        """🎯 分析特征有效性
        
        Args:
            features_dict: 特征字典 {feature_name: tensor}
            target_iou: 目标IoU值 [N]
            step: 当前步数
            
        Returns:
            Dict[str, float]: 特征有效性分数
        """
        effectiveness_scores = {}
        
        print(f"\n🔬 特征有效性分析 - Step {step}")
        print(f"{'='*60}")
        
        for feature_name, feature_tensor in features_dict.items():
            try:
                # 将特征展平
                feature_flat = self._flatten_feature(feature_tensor)
                target_flat = target_iou.cpu().numpy().flatten()
                
                # 计算多种相关性指标
                effectiveness = self._compute_effectiveness_metrics(
                    feature_flat, target_flat, feature_name
                )
                effectiveness_scores[feature_name] = effectiveness
                
            except Exception as e:
                print(f"❌ 分析特征 {feature_name} 失败: {e}")
                effectiveness_scores[feature_name] = 0.0
        
        # 存储结果
        self.effectiveness_scores[step] = effectiveness_scores
        
        # 显示结果
        self._display_effectiveness_results(effectiveness_scores)
        
        return effectiveness_scores
    
    def analyze_feature_redundancy(self, 
                                 features_dict: Dict[str, torch.Tensor],
                                 step: int = 0) -> Dict[str, Dict[str, float]]:
        """📊 分析特征间冗余度
        
        Args:
            features_dict: 特征字典
            step: 当前步数
            
        Returns:
            Dict: 特征间相关性矩阵
        """
        feature_names = list(features_dict.keys())
        n_features = len(feature_names)
        
        # 初始化相关性矩阵
        correlation_matrix = np.zeros((n_features, n_features))
        correlation_dict = {}
        
        print(f"\n📊 特征冗余度分析 - Step {step}")
        print(f"{'='*60}")
        
        for i, name1 in enumerate(feature_names):
            correlation_dict[name1] = {}
            for j, name2 in enumerate(feature_names):
                if i <= j:
                    if i == j:
                        correlation = 1.0
                    else:
                        # 计算特征间相关性
                        feature1 = self._flatten_feature(features_dict[name1])
                        feature2 = self._flatten_feature(features_dict[name2])
                        correlation = self._compute_feature_correlation(feature1, feature2)
                    
                    correlation_matrix[i, j] = correlation
                    correlation_matrix[j, i] = correlation
                    correlation_dict[name1][name2] = correlation
                    correlation_dict[name2] = correlation_dict.get(name2, {})
                    correlation_dict[name2][name1] = correlation
        
        # 存储结果
        self.redundancy_analysis[step] = correlation_dict
        
        # 可视化相关性矩阵
        self._visualize_correlation_matrix(correlation_matrix, feature_names, step)
        
        # 识别高度冗余的特征对
        self._identify_redundant_features(correlation_dict)
        
        return correlation_dict
    
    def recommend_feature_combination(self, 
                                    effectiveness_scores: Dict[str, float],
                                    redundancy_matrix: Dict[str, Dict[str, float]],
                                    max_features: int = 6,
                                    redundancy_threshold: float = 0.85) -> List[str]:
        """🎯 推荐最优特征组合
        
        Args:
            effectiveness_scores: 特征有效性分数
            redundancy_matrix: 特征冗余度矩阵
            max_features: 最大特征数量
            redundancy_threshold: 冗余度阈值
            
        Returns:
            List[str]: 推荐的特征组合
        """
        print(f"\n🎯 特征组合推荐")
        print(f"{'='*60}")
        
        # 按有效性排序
        sorted_features = sorted(effectiveness_scores.items(), 
                               key=lambda x: x[1], reverse=True)
        
        selected_features = []
        
        for feature_name, effectiveness in sorted_features:
            if len(selected_features) >= max_features:
                break
                
            # 检查是否与已选特征冗余
            is_redundant = False
            for selected_feature in selected_features:
                if redundancy_matrix[feature_name][selected_feature] > redundancy_threshold:
                    print(f"   ⚠️  跳过 {feature_name}: 与 {selected_feature} 冗余度过高 "
                          f"({redundancy_matrix[feature_name][selected_feature]:.3f})")
                    is_redundant = True
                    break
            
            if not is_redundant:
                selected_features.append(feature_name)
                print(f"   ✅ 选择 {feature_name}: 有效性={effectiveness:.3f}")
        
        print(f"\n🏆 推荐特征组合: {selected_features}")
        print(f"   📊 总维度预估: {self._estimate_total_dimensions(selected_features)}")
        
        return selected_features
    
    def _flatten_feature(self, feature_tensor: torch.Tensor) -> np.ndarray:
        """将特征张量展平为1D数组"""
        if isinstance(feature_tensor, torch.Tensor):
            feature_tensor = feature_tensor.detach().cpu()
        
        # 随机采样以减少计算量
        flat = feature_tensor.flatten().numpy()
        if len(flat) > 10000:
            indices = np.random.choice(len(flat), 10000, replace=False)
            flat = flat[indices]
        
        return flat
    
    def _compute_effectiveness_metrics(self, 
                                     feature: np.ndarray, 
                                     target: np.ndarray,
                                     feature_name: str) -> float:
        """计算特征有效性指标"""
        try:
            # 确保数据长度一致
            min_len = min(len(feature), len(target))
            feature = feature[:min_len]
            target = target[:min_len]
            
            # 1. 皮尔逊相关系数
            pearson_corr, _ = pearsonr(feature, target)
            if np.isnan(pearson_corr):
                pearson_corr = 0.0
            
            # 2. 斯皮尔曼相关系数
            spearman_corr, _ = spearmanr(feature, target)
            if np.isnan(spearman_corr):
                spearman_corr = 0.0
            
            # 3. 特征方差（信息量指标）
            feature_var = np.var(feature)
            if np.isnan(feature_var):
                feature_var = 0.0
            
            # 4. 综合有效性分数
            # 权重：相关性占70%，信息量占30%
            effectiveness = (abs(pearson_corr) * 0.4 + 
                           abs(spearman_corr) * 0.3 + 
                           min(feature_var, 1.0) * 0.3)
            
            print(f"   📈 {feature_name:15s}: "
                  f"Pearson={pearson_corr:.3f}, "
                  f"Spearman={spearman_corr:.3f}, "
                  f"Var={feature_var:.3f}, "
                  f"Score={effectiveness:.3f}")
            
            return effectiveness
            
        except Exception as e:
            print(f"   ❌ {feature_name}: 计算失败 {e}")
            return 0.0
    
    def _compute_feature_correlation(self, 
                                   feature1: np.ndarray, 
                                   feature2: np.ndarray) -> float:
        """计算两个特征间的相关性"""
        try:
            # 确保数据长度一致
            min_len = min(len(feature1), len(feature2))
            feature1 = feature1[:min_len]
            feature2 = feature2[:min_len]
            
            # 计算皮尔逊相关系数
            corr, _ = pearsonr(feature1, feature2)
            return abs(corr) if not np.isnan(corr) else 0.0
            
        except Exception as e:
            return 0.0
    
    def _visualize_correlation_matrix(self, 
                                    correlation_matrix: np.ndarray,
                                    feature_names: List[str],
                                    step: int):
        """可视化相关性矩阵"""
        try:
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, 
                       annot=True, 
                       fmt='.3f',
                       xticklabels=feature_names,
                       yticklabels=feature_names,
                       cmap='RdYlBu_r',
                       center=0.5)
            
            plt.title(f'Feature Correlation Matrix - Step {step}')
            plt.tight_layout()
            
            # 保存图片
            save_path = self.save_dir / f'correlation_matrix_step_{step}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"   💾 相关性矩阵已保存: {save_path}")
            
        except Exception as e:
            print(f"   ❌ 可视化失败: {e}")
    
    def _identify_redundant_features(self, correlation_dict: Dict[str, Dict[str, float]]):
        """识别冗余特征对"""
        redundant_pairs = []
        
        for feature1, correlations in correlation_dict.items():
            for feature2, corr in correlations.items():
                if feature1 < feature2 and corr > 0.85:  # 避免重复和自相关
                    redundant_pairs.append((feature1, feature2, corr))
        
        if redundant_pairs:
            print(f"\n   ⚠️  发现高度冗余特征对:")
            for f1, f2, corr in sorted(redundant_pairs, key=lambda x: x[2], reverse=True):
                print(f"      {f1} ↔ {f2}: {corr:.3f}")
        else:
            print(f"\n   ✅ 未发现高度冗余特征对")
    
    def _display_effectiveness_results(self, effectiveness_scores: Dict[str, float]):
        """显示有效性分析结果"""
        sorted_scores = sorted(effectiveness_scores.items(), 
                             key=lambda x: x[1], reverse=True)
        
        print(f"\n🏆 特征有效性排名:")
        for i, (feature_name, score) in enumerate(sorted_scores, 1):
            status = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "📊"
            print(f"   {status} {i:2d}. {feature_name:15s}: {score:.3f}")
    
    def _estimate_total_dimensions(self, selected_features: List[str]) -> int:
        """估算特征组合的总维度"""
        # 基于特征名称估算维度
        dimension_map = {
            'top1': 1, 'top2': 2, 'top3': 3, 'top4': 4, 'top5': 5,
            'mean': 1, 'std': 1, 'entropy': 1, 'max': 1, 'range': 1,
            'top3_centered': 3, 'top4_centered': 4
        }
        
        total_dim = 0
        for feature in selected_features:
            total_dim += dimension_map.get(feature, 1)
        
        return total_dim * 4  # 4个方向
    
    def save_analysis_results(self, filename: str = "feature_analysis_results.json"):
        """保存分析结果到文件"""
        results = {
            'effectiveness_scores': self.effectiveness_scores,
            'redundancy_analysis': self.redundancy_analysis,
            'correlation_history': self.correlation_history
        }
        
        save_path = self.save_dir / filename
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 分析结果已保存: {save_path}")
    
    def generate_report(self, step: int) -> str:
        """生成分析报告"""
        if step not in self.effectiveness_scores:
            return "❌ 没有找到分析数据"
        
        effectiveness = self.effectiveness_scores[step]
        redundancy = self.redundancy_analysis.get(step, {})
        
        # 推荐特征组合
        recommended = self.recommend_feature_combination(effectiveness, redundancy)
        
        report = f"""
🔬 特征有效性分析报告 - Step {step}
{'='*60}

📊 特征有效性排名:
"""
        sorted_scores = sorted(effectiveness.items(), key=lambda x: x[1], reverse=True)
        for i, (feature, score) in enumerate(sorted_scores[:10], 1):
            report += f"{i:2d}. {feature:15s}: {score:.3f}\n"
        
        report += f"""
🎯 推荐特征组合:
{', '.join(recommended)}

📈 预估总维度: {self._estimate_total_dimensions(recommended)}

💡 优化建议:
1. 优先使用高有效性特征
2. 避免冗余度>0.85的特征组合
3. 控制总维度在合理范围内(<=32)
4. 定期重新评估特征有效性
"""
        
        return report

def main():
    """示例用法"""
    analyzer = FeatureEffectivenessAnalyzer()
    
    # 模拟特征数据
    batch_size, height, width = 2, 64, 64
    
    features_dict = {
        'top4': torch.randn(batch_size, 4, 4, height, width),
        'top4_centered': torch.randn(batch_size, 4, 4, height, width),
        'mean': torch.randn(batch_size, 4, 1, height, width),
        'entropy': torch.randn(batch_size, 4, 1, height, width),
        'std': torch.randn(batch_size, 4, 1, height, width),
    }
    
    target_iou = torch.rand(batch_size * height * width)
    
    # 分析特征有效性
    effectiveness = analyzer.analyze_feature_effectiveness(features_dict, target_iou, step=1)
    
    # 分析特征冗余度
    redundancy = analyzer.analyze_feature_redundancy(features_dict, step=1)
    
    # 生成报告
    report = analyzer.generate_report(step=1)
    print(report)
    
    # 保存结果
    analyzer.save_analysis_results()

if __name__ == "__main__":
    main()