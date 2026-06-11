import os
import json
import numpy as np
import cv2
from glob import glob

# GT数据集路径
gt_annotation_file = "D:/project/WTGFL/WindBlade-30K/annotations/nobackground/test.json"  # COCO格式
# 或者
gt_folder = "path/to/your/labels"  # YOLO/VOC格式

# 检测结果路径 - 修改为上级目录以处理所有结果
results_folder = "D:/project/WTGFL/mmdetection-main/kresult"

# 加载GT数据（以COCO格式为例）
def load_gt_annotations(gt_file):
    """加载COCO格式的GT标注"""
    print(f"正在加载GT标注文件: {gt_file}")
    with open(gt_file, 'r') as f:
        annotations = json.load(f)
    
    # 创建图像id到标注的映射
    img_to_anns = {}
    for img in annotations['images']:
        img_to_anns[img['file_name']] = {'image_id': img['id'], 'annotations': []}
        img_to_anns[os.path.basename(img['file_name'])] = {'image_id': img['id'], 'annotations': []}  # 也添加仅文件名的映射
    
    print(f"找到 {len(annotations['images'])} 张图像")
    print(f"找到 {len(annotations['annotations'])} 个标注")
    
    for ann in annotations['annotations']:
        img_id = ann['image_id']
        # 找到对应的图像文件名
        for img in annotations['images']:
            if img['id'] == img_id:
                file_name = img['file_name']
                base_name = os.path.basename(file_name)
                
                if file_name in img_to_anns:
                    img_to_anns[file_name]['annotations'].append(ann)
                
                if base_name in img_to_anns:
                    img_to_anns[base_name]['annotations'].append(ann)
                break
    
    return img_to_anns

# 计算IoU
def compute_iou(box1, box2):
    """计算两个边界框之间的IoU"""
    # box格式 [x1, y1, x2, y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    inter = w * h
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0

# 为每个检测结果找到对应的GT框
def match_detections_with_gt(gt_annotations):
    result_dirs = glob(os.path.join(results_folder, "*"))
    print(f"找到 {len(result_dirs)} 个结果目录")
    
    processed = 0
    matched = 0
    
    for result_dir in result_dirs:
        img_info_file = os.path.join(result_dir, "img_info.json")
        if not os.path.exists(img_info_file):
            print(f"跳过目录 {result_dir} (没有img_info.json)")
            continue
            
        processed += 1
        
        # 读取检测结果信息
        with open(img_info_file, 'r') as f:
            img_info = json.load(f)
            
        # 获取图片文件名
        img_path = img_info.get('img_path', '')
        img_name = os.path.basename(img_path) if img_path else img_info.get('img_name', '')
        
        print(f"处理: {img_name} (来自目录 {os.path.basename(result_dir)})")
        
        # 检查是否有对应的GT标注
        if img_name in gt_annotations:
            print(f"  找到对应的GT标注, 共 {len(gt_annotations[img_name]['annotations'])} 个")
            # 获取检测框
            if 'detection_bbox' not in img_info:
                print(f"  警告: 没有找到detection_bbox字段")
                continue
                
            det_bbox = img_info['detection_bbox']
            
            # 查找最佳匹配的GT框
            best_iou = 0
            best_gt = None
            
            for ann in gt_annotations[img_name]['annotations']:
                # 将COCO格式 [x,y,w,h] 转为 [x1,y1,x2,y2]
                gt_box = ann['bbox']
                if len(gt_box) == 4:
                    # COCO格式默认是xywh
                    gt_box = [gt_box[0], gt_box[1], 
                             gt_box[0] + gt_box[2], 
                             gt_box[1] + gt_box[3]]
                
                # 计算IoU
                iou = compute_iou(det_bbox, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt_box
            
            # 更新结果文件
            if best_gt is not None:
                matched += 1
                print(f"  找到匹配的GT框, IoU = {best_iou:.2f}")
                # 读取现有的info dict
                info_file = os.path.join(result_dir, "best_info.json")
                if os.path.exists(info_file):
                    with open(info_file, 'r') as f:
                        info_dict = json.load(f)
                        
                    # 添加GT信息
                    info_dict['matched_gt_bbox'] = best_gt
                    info_dict['iou'] = best_iou
                    
                    # 保存更新后的信息
                    with open(info_file, 'w') as f:
                        json.dump(info_dict, f, indent=2)
                
                # 重新生成可视化结果
                # 尝试多种可能的图片路径
                img = None
                possible_paths = [
                    img_path,
                    os.path.join("D:/project/WTGFL/WindBlade-30K/images/nobackground/test", img_name),
                    os.path.join("D:/project/WTGFL/WindBlade-30K/images/nobackground/test", os.path.basename(img_name))
                ]
                
                for path in possible_paths:
                    if path and os.path.exists(path):
                        print(f"  尝试读取图像: {path}")
                        img = cv2.imread(path)
                        if img is not None:
                            print(f"  成功读取图像, 形状: {img.shape}")
                            break
                
                if img is None:
                    print(f"  警告: 无法加载图像, 尝试了以下路径: {possible_paths}")
                    continue
                
                # 绘制检测框
                x1, y1, x2, y2 = map(int, det_bbox)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4)  # 绿色框，粗线条(4px)
                score = img_info['detection_score']
                cv2.putText(img, f'{score:.2f}', (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # 绘制GT框 - 使用白色细线条
                gx1, gy1, gx2, gy2 = map(int, best_gt)
                cv2.rectangle(img, (gx1, gy1), (gx2, gy2), (255, 255, 255), 2)  # 白色框，细线条(2px)
                
                # 画一条连接线 - 使用虚线
                center_pred = ((x1 + x2) // 2, (y1 + y2) // 2)
                center_gt = ((gx1 + gx2) // 2, (gy1 + gy2) // 2)
                # 使用较细的黄色虚线
                for i in range(0, int(np.linalg.norm(np.array(center_pred) - np.array(center_gt))), 10):
                    # 使用参数方程计算线段上的点
                    ratio = i / np.linalg.norm(np.array(center_pred) - np.array(center_gt))
                    if i % 20 < 10:  # 画虚线
                        pt1 = (int((1-ratio)*center_pred[0] + ratio*center_gt[0]), 
                               int((1-ratio)*center_pred[1] + ratio*center_gt[1]))
                        cv2.circle(img, pt1, 1, (0, 255, 255), -1)  # 黄色小点
                
                # 保存图片
                save_path = os.path.join(result_dir, "best_det_result_with_gt.jpg")
                cv2.imwrite(save_path, img)
                print(f"  已保存可视化结果到: {save_path}")
        else:
            print(f"  未找到对应的GT标注")
    
    print(f"处理完成. 总共处理了 {processed} 个目录, 其中 {matched} 个成功匹配到GT框")

# 主函数
def main():
    # 检查GT标注文件是否存在
    if not os.path.exists(gt_annotation_file):
        print(f"错误: GT标注文件不存在: {gt_annotation_file}")
        return
        
    # 检查结果目录是否存在
    if not os.path.exists(results_folder):
        print(f"错误: 结果目录不存在: {results_folder}")
        return
    
    # 加载GT标注
    gt_annotations = load_gt_annotations(gt_annotation_file)
    
    # 匹配检测结果和GT
    match_detections_with_gt(gt_annotations)

if __name__ == "__main__":
    main()