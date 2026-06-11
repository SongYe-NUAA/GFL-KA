import pickle
import json
import numpy as np
import os
import torch

def convert_to_serializable(obj):
    """将不可序列化的对象转换为可JSON序列化的对象"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, (set, tuple)):
        return list(obj)
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    else:
        # 对于其他类型，尝试转换为字符串
        try:
            return str(obj)
        except:
            return "Non-serializable object"

class NumpyEncoder(json.JSONEncoder):
    """处理NumPy数据类型的JSON编码器"""
    def default(self, obj):
        if isinstance(obj, (np.ndarray, np.number)):
            return convert_to_serializable(obj)
        return super(NumpyEncoder, self).default(obj)

# 确保输出目录存在
output_dir = os.path.dirname('runs/windturbine_GFLv2/nobackground_4090_train/addsoftkurtosisfactorattention_state/results.json')
os.makedirs(output_dir, exist_ok=True)

# COCO格式必须包含的字段: image_id, category_id, bbox, score
# COCO格式的检测结果示例:
# [
#   {
#     "image_id": 1,
#     "category_id": 1,
#     "bbox": [100.0, 100.0, 50.0, 50.0],  # [x, y, width, height]
#     "score": 0.9
#   },
#   ...
# ]

# 加载标注文件以获取正确的image_id映射（如果存在）
def load_image_ids_from_annotation(ann_file):
    try:
        print(f"尝试从标注文件加载图像ID: {ann_file}")
        with open(ann_file, 'r') as f:
            annotations = json.load(f)
        
        image_ids = {}
        for i, img in enumerate(annotations['images']):
            image_ids[i] = img['id']
        
        print(f"成功加载 {len(image_ids)} 个图像ID")
        return image_ids
    except Exception as e:
        print(f"加载标注文件失败: {e}")
        return None

# 尝试加载标注文件
annotation_file = '../WindBlade-30K/annotations/nobackground/test.json'
image_ids = load_image_ids_from_annotation(annotation_file)

try:
    print("正在读取 pkl 文件...")
    with open('runs/windturbine_GFLv2/nobackground_4090_train/addsoftkurtosisfactorattention_state/results.pkl', 'rb') as f:
        data = pickle.load(f)

    print(f"读取成功，数据类型: {type(data)}")
    
    # 直接转换为COCO格式
    coco_results = []
    
    # 检查数据结构
    if isinstance(data, list):
        print(f"数据是列表，长度: {len(data)}")
        
        # 检查第一个元素以确定数据结构
        if len(data) > 0:
            first_item = data[0]
            print(f"第一个元素类型: {type(first_item)}")
            
            # 检查是否是MMDetection的结果格式
            if isinstance(first_item, dict) and 'img_id' in first_item and 'pred_instances' in first_item:
                print("检测到MMDetection结果格式，包含img_id和pred_instances")
                
                for item in data:
                    image_id = item.get('img_id')
                    pred_instances = item.get('pred_instances', {})
                    
                    # 提取预测信息
                    labels = pred_instances.get('labels', [])
                    scores = pred_instances.get('scores', [])
                    bboxes = pred_instances.get('bboxes', [])
                    
                    # 转换为Python列表
                    if isinstance(labels, torch.Tensor):
                        labels = labels.cpu().numpy().tolist()
                    if isinstance(scores, torch.Tensor):
                        scores = scores.cpu().numpy().tolist()
                    if isinstance(bboxes, torch.Tensor):
                        bboxes = bboxes.cpu().numpy().tolist()
                    
                    # 检查数据长度是否一致
                    if not (len(labels) == len(scores) == len(bboxes)):
                        print(f"警告: 图像ID {image_id} 的数据长度不一致 - labels: {len(labels)}, scores: {len(scores)}, bboxes: {len(bboxes)}")
                        continue
                    
                    # 创建COCO格式的检测结果
                    for i in range(len(labels)):
                        # 确保边界框格式正确
                        if len(bboxes[i]) != 4:
                            print(f"警告: 图像ID {image_id} 的边界框 {i} 格式不正确: {bboxes[i]}")
                            continue
                        
                        # 提取边界框坐标
                        x1, y1, x2, y2 = bboxes[i]
                        width = x2 - x1
                        height = y2 - y1
                        
                        # 创建COCO格式的检测结果
                        detection = {
                            'image_id': int(image_id),
                            'category_id': int(labels[i]) + 1,  # COCO类别ID从1开始
                            'bbox': [float(x1), float(y1), float(width), float(height)],
                            'score': float(scores[i])
                        }
                        coco_results.append(detection)
                
                print(f"从MMDetection结果格式中提取了 {len(coco_results)} 个检测结果")
            
            # 检查是否是分类别组织的列表
            elif isinstance(first_item, list):
                print("数据结构似乎是按类别组织的列表，这是老版本MMDetection的格式")
                
                # MMDetection 通常按类别组织结果
                # 每个类别的结果列表中包含该类别检测到的所有边界框
                # 每个检测结果通常为: [image_idx, x1, y1, x2, y2, score]
                
                # 遍历每个类别的结果
                for category_id, category_results in enumerate(data):
                    print(f"处理类别 {category_id+1}，结果数量: {len(category_results)}")
                    
                    # 样本结果，帮助调试
                    if len(category_results) > 0:
                        print(f"样本结果: {category_results[0]}")
                    
                    for detection in category_results:
                        # 尝试确定结果格式
                        if len(detection) >= 5:  # 至少需要[x1,y1,x2,y2,score]
                            # 最常见的格式: [image_idx, x1, y1, x2, y2, score]
                            if len(detection) > 5:
                                img_idx = int(detection[0])
                                bbox = detection[1:5]
                                score = float(detection[5])
                                
                                # 使用正确的图像ID
                                image_id = img_idx
                            # 或者: [x1, y1, x2, y2, score]
                            else:
                                # 无法获取图像索引，使用默认值
                                image_id = 1  # 默认值
                                bbox = detection[:4]
                                score = float(detection[4])
                            
                            # 转换[x1,y1,x2,y2]到COCO格式[x,y,width,height]
                            try:
                                x1, y1, x2, y2 = map(float, bbox)
                                width = x2 - x1
                                height = y2 - y1
                                
                                coco_detection = {
                                    'image_id': image_id,
                                    'category_id': category_id + 1,  # COCO类别ID从1开始
                                    'bbox': [x1, y1, width, height],
                                    'score': float(score)
                                }
                                coco_results.append(coco_detection)
                            except Exception as e:
                                print(f"处理边界框时出错: {e}, bbox: {bbox}")
            else:
                # 如果数据已经是COCO格式
                print("检查数据是否已经是COCO格式")
                
                # 验证第一个元素是否包含必要的COCO字段
                if isinstance(first_item, dict) and 'image_id' in first_item and 'category_id' in first_item:
                    print("数据已经是COCO格式，直接使用")
                    coco_results = data
                else:
                    print("数据结构无法识别，尝试手动解析")
                    # 尝试理解数据结构
                    for i, item in enumerate(data[:10]):
                        print(f"元素 {i}: {item}")
    else:
        print(f"数据不是列表，而是 {type(data)}，无法处理")
        raise ValueError("无法识别的数据格式")
        
    # 检查结果
    if not coco_results:
        raise ValueError("无法生成COCO格式的结果")
    
    print(f"共生成 {len(coco_results)} 个COCO格式的检测结果")
    print(f"样本结果: {coco_results[0]}")
    
    # 统计检测结果中使用的图像ID
    image_ids_in_results = set(item['image_id'] for item in coco_results)
    print(f"检测结果中包含 {len(image_ids_in_results)} 个不同的图像ID")
    if len(image_ids_in_results) > 0:
        print(f"图像ID范围: {min(image_ids_in_results)} 到 {max(image_ids_in_results)}")
    
    # 写入 json 文件
    print("正在写入 JSON 文件...")
    with open('runs/windturbine_GFLv2/nobackground_4090_train/addsoftkurtosisfactorattention_state/results.json', 'w') as f:
        json.dump(coco_results, f)
    
    print("转换完成!")

except Exception as e:
    print(f"转换过程中出现错误: {e}")
    import traceback
    traceback.print_exc()
    
    print("尝试读取和打印原始数据结构...")
    try:
        with open('runs/windturbine_GFLv2/nobackground_4090_train/addsoftkurtosisfactorattention_state/results.pkl', 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, list):
            print(f"数据是长度为 {len(data)} 的列表")
            for i, item in enumerate(data[:3]):  # 只打印前3个元素
                print(f"元素 {i} 类型: {type(item)}")
                if isinstance(item, dict):
                    print(f"  字典键: {item.keys()}")
                    if 'pred_instances' in item:
                        pred = item['pred_instances']
                        print(f"  pred_instances键: {pred.keys() if isinstance(pred, dict) else type(pred)}")
                elif isinstance(item, list):
                    print(f"  子列表长度: {len(item)}")
                    if len(item) > 0:
                        print(f"  第一个子元素: {item[0]}")
                        if isinstance(item[0], list) and len(item[0]) > 0:
                            print(f"    第一个子元素的第一个元素: {item[0][0]}")
        else:
            print(f"数据类型: {type(data)}")
    
    except Exception as e2:
        print(f"无法读取原始数据: {e2}")
    
    print("请手动检查数据结构并相应调整脚本")