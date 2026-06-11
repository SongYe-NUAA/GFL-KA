import os
import json
from collections import defaultdict

def count_labelme_annotations(json_path):
    """统计单个labelme格式标注文件"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        shapes = data.get('shapes', [])
        per_class_count = defaultdict(int)
        
        # 统计每个类别
        for shape in shapes:
            label = shape['label']
            # 处理复合标签
            labels = label.split(';')
            for l in labels:
                per_class_count[l.strip()] += 1
            # 统计完整标签
            per_class_count[label] += 1
            
        return len(shapes), per_class_count
    except Exception as e:
        print(f"处理文件 {json_path} 时出错: {str(e)}")
        return 0, defaultdict(int)

def count_coco_annotations(json_path):
    """统计COCO格式标注文件"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 创建类别ID到名称的映射
        categories = {cat['id']: cat['name'] for cat in data.get('categories', [])}
        per_class_count = defaultdict(int)
        
        # 统计每个类别的标注数量
        for ann in data.get('annotations', []):
            category_id = ann['category_id']
            category_name = categories.get(category_id, f"unknown_{category_id}")
            per_class_count[category_name] += 1
            
        return len(data.get('annotations', [])), per_class_count
    except Exception as e:
        print(f"处理文件 {json_path} 时出错: {str(e)}")
        return 0, defaultdict(int)

def count_txt_annotations(txt_path):
    """统计txt格式标注文件"""
    try:
        per_class_count = defaultdict(int)
        annotation_count = 0
        
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            if line:
                # YOLO格式: class_id x y w h
                # 或自定义格式: class_name x1 y1 x2 y2
                parts = line.split()
                if len(parts) >= 5:  # 确保至少有类别和坐标信息
                    annotation_count += 1
                    # 假设第一个值是类别ID或名称
                    class_name = parts[0]
                    per_class_count[class_name] += 1
                    
        return annotation_count, per_class_count
    except Exception as e:
        print(f"处理文件 {txt_path} 时出错: {str(e)}")
        return 0, defaultdict(int)

def detect_annotation_type(file_path):
    """检测标注文件类型"""
    if file_path.endswith('.txt'):
        return 'txt'
    
    if not file_path.endswith('.json'):
        return 'unknown'
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if all(key in data for key in ['images', 'annotations', 'categories']):
            return 'coco'
        elif 'shapes' in data and 'imagePath' in data:
            return 'labelme'
        else:
            return 'unknown'
    except:
        return 'unknown'

def count_annotations(folder_path):
    """统计文件夹中的标注"""
    total_annotations = 0
    per_class_count = defaultdict(int)
    file_count = {'labelme': 0, 'coco': 0, 'txt': 0, 'unknown': 0}
    per_file_annotations = {}
    
    # 遍历文件夹
    for filename in os.listdir(folder_path):
        if filename.endswith(('.json', '.txt')):
            file_path = os.path.join(folder_path, filename)
            
            # 检测文件类型
            ann_type = detect_annotation_type(file_path)
            file_count[ann_type] += 1
            
            # 根据类型统计标注
            if ann_type == 'labelme':
                annotations_count, class_counts = count_labelme_annotations(file_path)
            elif ann_type == 'coco':
                annotations_count, class_counts = count_coco_annotations(file_path)
            elif ann_type == 'txt':
                annotations_count, class_counts = count_txt_annotations(file_path)
            else:
                continue
            
            total_annotations += annotations_count
            per_file_annotations[filename] = annotations_count
            
            # 更新类别计数
            for label, count in class_counts.items():
                per_class_count[label] += count
    
    # 打印统计结果
    print(f"\n=== 标注统计结果 ===")
    print(f"文件总数: {sum(file_count.values())}")
    print(f"- Labelme格式: {file_count['labelme']}")
    print(f"- COCO格式: {file_count['coco']}")
    print(f"- TXT格式: {file_count['txt']}")
    print(f"- 未知格式: {file_count['unknown']}")
    print(f"\n标注总数: {total_annotations}")
    if sum(file_count.values()) > 0:
        print(f"平均每文件标注数: {total_annotations/sum(file_count.values()):.2f}")
    
    print("\n=== 各类别数量 ===")
    for label, count in sorted(per_class_count.items(), key=lambda x: x[1], reverse=True):
        print(f"{label}: {count}")
    
    if per_file_annotations:
        max_file = max(per_file_annotations.items(), key=lambda x: x[1])
        min_file = min(per_file_annotations.items(), key=lambda x: x[1])
        print(f"\n标注最多的文件: {max_file[0]} ({max_file[1]}个标注)")
        print(f"标注最少的文件: {min_file[0]} ({min_file[1]}个标注)")
    
    return total_annotations, per_class_count, file_count

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    else:
        folder_path = input("请输入标注文件夹路径: ")
    
    if not os.path.exists(folder_path):
        print(f"错误：文件夹 {folder_path} 不存在")
    else:
        count_annotations(folder_path)