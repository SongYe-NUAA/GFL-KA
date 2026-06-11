import os
import cv2
import json
import numpy as np
from tqdm import tqdm


def create_mask_from_polygon(polygon, height, width):
    """
    从多边形点创建掩码
    Args:
        polygon: 多边形点列表
        height: 图像高度
        width: 图像宽度
    Returns:
        mask: 二值掩码图像
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    polygon = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(mask, [polygon], 1)
    return mask


def seg_to_det(seg_folder, img_folder, output_folder, class_name="windturbine"):
    """
    将分割标注转换为检测标注
    Args:
        seg_folder: 分割标注文件夹路径
        img_folder: 原始图像文件夹路径
        output_folder: 输出文件夹路径
        class_name: 类别名称
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    json_files = [f for f in os.listdir(seg_folder) if f.endswith('.json')]

    for json_file in tqdm(json_files):
        # 读取原始JSON标注
        json_path = os.path.join(seg_folder, json_file)
        with open(json_path, 'r', encoding='utf-8') as f:
            anno = json.load(f)

        # 读取图像获取尺寸
        img_name = os.path.splitext(json_file)[0] + '.jpg'
        img_path = os.path.join(img_folder, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Cannot find image for {json_file}")
            continue

        height, width = img.shape[:2]

        # 创建新的JSON标注
        new_anno = {
            "version": "5.0.1",
            "flags": {},
            "shapes": [],
            "imagePath": anno.get("imagePath", img_name),
            "imageData": anno.get("imageData", None),
            "imageHeight": int(height),
            "imageWidth": int(width)
        }

        # 创建空掩码
        mask = np.zeros((height, width), dtype=np.uint8)

        # 解析原始JSON中的标注信息
        if 'shapes' in anno:
            shapes = anno['shapes']
            for shape in shapes:
                if shape['shape_type'] == 'polygon':
                    points = shape['points']
                    obj_mask = create_mask_from_polygon(points, height, width)
                    mask = cv2.bitwise_or(mask, obj_mask)
        else:
            print(f"Unsupported JSON format in {json_file}")
            continue

        # 找到所有连通区域
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

        # 为每个目标创建矩形标注
        for i in range(1, num_labels):
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            area = int(stats[i, cv2.CC_STAT_AREA])

            if area < 100:  # 过滤小目标
                continue

            # 只需要左上角和右下角两个点
            rect_points = [
                [int(x), int(y)],  # 左上角
                [int(x + w), int(y + h)]  # 右下角
            ]

            shape_dict = {
                "label": class_name,
                "points": rect_points,
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {}
            }

            new_anno["shapes"].append(shape_dict)

        # 保存新的JSON标注
        output_path = os.path.join(output_folder, json_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(new_anno, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    # 设置路径
    seg_folder = r"C:\Users\Administrator\Desktop\wind_dataset\Blade30\labels"
    img_folder = r"C:\Users\Administrator\Desktop\wind_dataset\Blade30\images"
    output_folder = r"C:\Users\Administrator\Desktop\wind_dataset\Blade30\labels1"

    # 转换标注
    seg_to_det(seg_folder, img_folder, output_folder)

if __name__ == "__main__":
    # 设置路径
    seg_folder = r"C:\Users\Administrator\Desktop\wind_dataset\Blade30\labels"  # JSON标注文件夹
    img_folder = r"C:\Users\Administrator\Desktop\wind_dataset\Blade30\images"  # 原始图像文件夹
    output_folder = r"C:\Users\Administrator\Desktop\wind_dataset\Blade30\labels1"  # 输出文件夹

    # 转换标注
    seg_to_det(seg_folder, img_folder, output_folder)