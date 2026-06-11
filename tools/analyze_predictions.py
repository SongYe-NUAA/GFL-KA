import json
import numpy as np
from pycocotools.coco import COCO
import pickle
import torch
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings

# 设置OpenMP环境变量以避免冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 设置Matplotlib后端，避免潜在的GUI问题
import matplotlib
matplotlib.use('Agg')

class NumpyEncoder(json.JSONEncoder):
    """处理NumPy数据类型的JSON编码"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return float(obj)
        if isinstance(obj, np.int64) or isinstance(obj, np.int32):
            return int(obj)
        return super(NumpyEncoder, self).default(obj)

def plot_score_iou_distribution(scores, ious, save_path=None):
    """绘制预测分数和IoU的散点图"""
    try:
        # 导入必要的库
        from scipy import stats
        from scipy.stats import gaussian_kde
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.metrics import r2_score
        
        # 设置图表样式 (更新样式使用方式，避免弃用警告)
        plt.style.use('seaborn-v0_8')
        plt.figure(figsize=(12, 10))
        
        # 设置全局字体大小
        plt.rcParams.update({'font.size': 14,
                            'axes.labelsize': 16,
                            'axes.titlesize': 20,
                            'xtick.labelsize': 14,
                            'ytick.labelsize': 14})
        
        # 创建颜色映射
        colors = plt.cm.viridis(np.linspace(0, 1, 256))
        
        # 计算点密度用于颜色映射
        xy = np.vstack([scores, ious])
        z = gaussian_kde(xy)(xy)
        
        # 根据密度排序点，使得高密度的点绘制在上层
        idx = z.argsort()
        scores, ious, z = scores[idx], ious[idx], z[idx]
        
        # 绘制散点图
        plt.scatter(scores, ious, 
                   c=z, 
                   s=20,      # 增大点的大小
                   alpha=0.6,  # 调整透明度
                   cmap='viridis',
                   rasterized=True)  # 光栅化以减小文件大小
        
        # 添加颜色条 - 修复fontsize参数错误
        cbar = plt.colorbar(label='Point Density')
        cbar.set_label('Point Density', fontsize=16)
        cbar.ax.tick_params(labelsize=14)  # 增大colorbar刻度字体
        
        # 添加对角线
        plt.plot([0, 1], [0, 1], 'r--', label='Ideal Line', linewidth=2)
        
        # 添加高级拟合
        # 线性拟合作为参考
        slope, intercept, r_value, p_value, std_err = stats.linregress(scores, ious)
        fit_line = slope * np.array([0, 1]) + intercept
        
        # 多项式拟合 - 尝试不同阶数并选择最佳
        degrees = [1, 2, 3]
        best_degree = 1
        best_r2 = 0
        best_model = None
        x_pred = np.linspace(0, 1, 100).reshape(-1, 1)
        
        # 数据范围
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        for degree in degrees:
            model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
            model.fit(scores.reshape(-1, 1), ious)
            y_pred = model.predict(x_pred)
            r2 = r2_score(ious, model.predict(scores.reshape(-1, 1)))
            
            if r2 > best_r2:
                best_r2 = r2
                best_degree = degree
                best_model = model
        
        # 使用最佳模型在有数据的区域进行预测
        y_pred = best_model.predict(x_pred)
        
        # 在数据外区域使用线性外推
        # 使用线性回归获取线性外推的斜率和截距
        linear_model = LinearRegression()
        linear_model.fit(scores.reshape(-1, 1), ious)
        linear_slope = linear_model.coef_[0]
        linear_intercept = linear_model.intercept_
        
        # 创建组合预测：在数据范围内使用多项式，在范围外使用线性
        y_pred_combined = np.copy(y_pred)
        
        # 对于小于最小数据点的区域，使用线性外推
        low_mask = x_pred.ravel() < min_score
        if np.any(low_mask):
            # 获取多项式在最小数据点的值和斜率
            min_score_point = np.array([[min_score]])
            min_poly_value = best_model.predict(min_score_point)[0]
            
            # 在最小数据点左侧使用线性外推
            for i in np.where(low_mask)[0]:
                y_pred_combined[i] = min_poly_value - linear_slope * (min_score - x_pred[i, 0])
        
        # 对于大于最大数据点的区域，使用线性外推
        high_mask = x_pred.ravel() > max_score
        if np.any(high_mask):
            # 获取多项式在最大数据点的值和斜率
            max_score_point = np.array([[max_score]])
            max_poly_value = best_model.predict(max_score_point)[0]
            
            # 在最大数据点右侧使用线性外推
            for i in np.where(high_mask)[0]:
                y_pred_combined[i] = max_poly_value + linear_slope * (x_pred[i, 0] - max_score)
        
        # 确保预测值在[0,1]范围内
        y_pred_combined = np.clip(y_pred_combined, 0, 1)
        
        # 计算置信区间
        # 预测值
        y_pred_scores = best_model.predict(scores.reshape(-1, 1))
        
        # 残差
        residuals = ious - y_pred_scores
        
        # 残差标准差
        residual_std = np.std(residuals)
        
        # 计算95%置信区间
        conf_interval = 1.96 * residual_std
        
        # 获取数据范围内的索引
        data_range_mask = (x_pred.ravel() >= min_score) & (x_pred.ravel() <= max_score)
        
        # 只在数据范围内绘制拟合曲线和置信区间
        plt.plot(x_pred[data_range_mask], y_pred[data_range_mask], 'g-', 
                linewidth=3.5,  # 增加拟合线粗细 
                label=f'Polynomial Fit')
        
        # 可视化数据范围
        plt.axvline(x=min_score, color='blue', linestyle=':', alpha=0.6, linewidth=3)  # 增加可见度
        plt.axvline(x=max_score, color='blue', linestyle=':', alpha=0.6, linewidth=3)  # 增加可见度
        
        # 只在数据范围内显示置信区间
        plt.fill_between(x_pred.ravel()[data_range_mask], 
                         np.maximum(0, y_pred.ravel()[data_range_mask] - conf_interval), 
                         np.minimum(1, y_pred.ravel()[data_range_mask] + conf_interval), 
                         color='g', alpha=0.25,  # 稍微增加透明度
                         label='95% Confidence Interval')
        
        # 设置网格
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 设置坐标轴范围和标签
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel('Top 1 value of General Distribution', fontsize=18, fontweight='bold')
        plt.ylabel('real localization quality (IoU)', fontsize=18, fontweight='bold')
        
        # 增大坐标轴数字大小
        plt.tick_params(axis='both', which='major', labelsize=16)
        
        # 添加标题
        plt.title('Distribution of Prediction Scores vs IoU', 
                 fontsize=22, 
                 fontweight='bold', 
                 pad=20)
        
        # 计算统计信息（保留计算但不显示）
        correlation = np.corrcoef(scores, ious)[0, 1]
        mean_iou = np.mean(ious)
        median_iou = np.median(ious)
        
        # 添加图例到左上角
        plt.legend(fontsize=12, loc='upper left', framealpha=0.9)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图形
        if save_path:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            # 保存高质量图片
            plt.savefig(save_path, 
                       dpi=300, 
                       bbox_inches='tight',
                       facecolor='white',
                       edgecolor='none')
            print(f"图表已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()
        return True
        
    except Exception as e:
        warnings.warn(f"绘图时出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def load_predictions(pred_file):
    """加载预测结果文件"""
    print(f"加载预测结果: {pred_file}")
    if pred_file.endswith('.pkl'):
        with open(pred_file, 'rb') as f:
            predictions = pickle.load(f)
    else:
        with open(pred_file, 'r') as f:
            predictions = json.load(f)
    return predictions

def bbox_iou(box1, box2):
    """计算两个边界框的IoU
    box格式: [x1, y1, x2, y2] 或 [x, y, w, h]
    """
    # 确保box格式统一：转换为[x1, y1, x2, y2]格式
    if len(box1) == 4:
        if box1[2] < box1[0] or box1[3] < box1[1]:  # 说明是[x, y, w, h]格式
            box1 = [box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]]
    if len(box2) == 4:
        if box2[2] < box2[0] or box2[3] < box2[1]:  # 说明是[x, y, w, h]格式
            box2 = [box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]]
    
    # 计算交集区域的坐标
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # 计算交集面积
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # 计算两个框的面积
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # 计算IoU
    union = area1 + area2 - intersection
    iou = intersection / union if union > 0 else 0
    
    return iou

def analyze_predictions(pred_file, anno_file, output_dir='.'):
    """分析预测结果，计算Top1预测框与真实标注的IoU"""
    try:
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载COCO格式的标注文件
        coco_gt = COCO(anno_file)
        
        # 加载预测结果
        predictions = load_predictions(pred_file)
        
        # 初始化结果存储
        results = {
            'image_results': [],
            'total_predictions': 0,
            'total_matched': 0,
            'iou_scores': [],
            'score_iou_correlation': 0
        }
        
        print("开始分析预测结果...")
        
        # 检查预测结果的格式
        if isinstance(predictions, list):
            if len(predictions) > 0:
                if isinstance(predictions[0], dict):
                    if 'img_id' in predictions[0]:  # MMDetection新格式
                        # 按图片ID组织预测结果
                        predictions_by_image = {}
                        for pred in predictions:
                            img_id = pred['img_id']
                            if 'pred_instances' in pred:
                                instances = pred['pred_instances']
                                scores = instances.get('scores', [])
                                if isinstance(scores, torch.Tensor):
                                    scores = scores.cpu().numpy()
                                bboxes = instances.get('bboxes', [])
                                if isinstance(bboxes, torch.Tensor):
                                    bboxes = bboxes.cpu().numpy()
                                
                                if len(scores) > 0:
                                    # 获取最高分数的预测框
                                    max_score_idx = np.argmax(scores)
                                    predictions_by_image[img_id] = {
                                        'bbox': bboxes[max_score_idx].tolist(),  # 转换为Python列表
                                        'score': float(scores[max_score_idx])
                                    }
                    else:  # COCO格式
                        # 按图片ID组织预测结果
                        predictions_by_image = {}
                        for pred in predictions:
                            img_id = pred['image_id']
                            if img_id not in predictions_by_image or pred['score'] > predictions_by_image[img_id]['score']:
                                predictions_by_image[img_id] = {
                                    'bbox': pred['bbox'],
                                    'score': pred['score']
                                }
        
        # 收集所有预测分数和IoU值
        prediction_scores = []
        iou_values = []
        
        # 遍历每张图片
        for img_id in tqdm(coco_gt.getImgIds()):
            # 获取图片的标注
            ann_ids = coco_gt.getAnnIds(imgIds=img_id)
            annotations = coco_gt.loadAnns(ann_ids)
            
            if img_id in predictions_by_image:
                pred = predictions_by_image[img_id]
                results['total_predictions'] += 1
                
                # 计算与所有真实标注框的IoU
                max_iou = 0
                matched_gt = None
                
                for ann in annotations:
                    gt_bbox = ann['bbox']  # COCO格式: [x, y, width, height]
                    pred_bbox = pred['bbox']
                    
                    iou = bbox_iou(pred_bbox, gt_bbox)
                    if iou > max_iou:
                        max_iou = iou
                        matched_gt = ann
                
                # 收集预测分数和IoU值
                prediction_scores.append(float(pred['score']))
                iou_values.append(float(max_iou))
                
                # 记录结果
                image_result = {
                    'image_id': int(img_id),  # 确保是Python int
                    'prediction_score': float(pred['score']),  # 确保是Python float
                    'max_iou': float(max_iou),  # 确保是Python float
                    'prediction_bbox': [float(x) for x in pred['bbox']]  # 确保是Python float列表
                }
                
                if matched_gt:
                    image_result['matched_gt_bbox'] = [float(x) for x in matched_gt['bbox']]  # 确保是Python float列表
                    if max_iou >= 0.5:  # IoU阈值0.5
                        results['total_matched'] += 1
                
                results['image_results'].append(image_result)
                results['iou_scores'].append(float(max_iou))  # 确保是Python float
        
        # 转换为numpy数组
        prediction_scores = np.array(prediction_scores)
        iou_values = np.array(iou_values)
        
        # 绘制散点图
        plot_success = plot_score_iou_distribution(
            prediction_scores,
            iou_values,
            save_path=os.path.join(output_dir, 'score_iou_distribution_initial.png')
        )
        
        if not plot_success:
            print("警告：绘图过程中出现错误，但分析将继续进行")
        
        # 添加分数-IoU相关性分析
        results['score_iou_correlation'] = float(np.corrcoef(prediction_scores, iou_values)[0, 1])
        
        # 计算统计信息
        results['mean_iou'] = float(np.mean(results['iou_scores']))  # 确保是Python float
        results['median_iou'] = float(np.median(results['iou_scores']))  # 确保是Python float
        results['match_rate'] = float(results['total_matched'] / results['total_predictions'] if results['total_predictions'] > 0 else 0)
        
        # 打印统计信息
        print("\n分析结果:")
        print(f"总预测数量: {results['total_predictions']}")
        print(f"匹配数量 (IoU >= 0.5): {results['total_matched']}")
        print(f"匹配率: {results['match_rate']:.2%}")
        print(f"平均IoU: {results['mean_iou']:.4f}")
        print(f"中位数IoU: {results['median_iou']:.4f}")
        
        # 保存详细结果
        output_file = os.path.join(output_dir, 'prediction_analysis.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
        print(f"\n详细结果已保存到: {output_file}")
        
        return results
        
    except Exception as e:
        print(f"分析过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='分析目标检测预测结果')
    parser.add_argument('pred_file', help='预测结果文件路径 (.pkl 或 .json)')
    parser.add_argument('anno_file', help='COCO格式的标注文件路径')
    parser.add_argument('--output-dir', default='analysis_results', help='输出目录路径')
    parser.add_argument('--no-plot', action='store_true', help='不生成散点图')
    args = parser.parse_args()
    
    # 设置输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 运行分析
    results = analyze_predictions(args.pred_file, args.anno_file, args.output_dir)
    
    if results is not None:
        print(f"\n预测分数与IoU的相关系数: {results['score_iou_correlation']:.4f}")
    else:
        print("\n分析失败") 