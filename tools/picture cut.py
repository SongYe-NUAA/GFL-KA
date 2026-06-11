#本代码是用于裁剪最大内接矩形的
import os
from PIL import Image, ImageOps
import numpy as np
import cv2



def find_largest_inscribed_rectangle(image_path):
    try:
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")

        # 添加padding
        padded_image = add_padding(image, padding_size=50)
        
        gray = cv2.cvtColor(padded_image, cv2.COLOR_BGR2GRAY)
        
        # 二值化处理
        _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        
        # 形态学操作处理毛刺和离散点
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 边缘检测
        edges = cv2.Canny(binary, 50, 150)

        # 找到轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # 找到最大轮廓
        largest_contour = max(contours, key=cv2.contourArea)

        # 创建掩码并填充最大轮廓
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [largest_contour], -1, 255, -1)

        # 计算距离变换
        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

        # 找到距离变换的最大值点
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(dist_transform)

        # 使用最大值点作为内接矩形的中心
        radius = int(max_val)
        x, y = max_loc

        # 创建矩形
        rect = (x - radius, y - radius, 2 * radius, 2 * radius)
        x, y, w, h = rect

        # 验证矩形是否完全在掩码内
        rect_mask = mask[y:y+h, x:x+w]
        if not np.all(rect_mask == 255):
            # 如果矩形不完全在掩码内，缩小半径直到找到合适的大小
            while radius > 0 and not np.all(mask[y:y+h, x:x+w] == 255):
                radius -= 1
                x = max_loc[0] - radius
                y = max_loc[1] - radius
                w = h = 2 * radius
                if x < 0 or y < 0 or x+w >= mask.shape[1] or y+h >= mask.shape[0]:
                    continue

        # 裁剪图像
        cropped = padded_image[y:y+h, x:x+w]

        # 保存调试图像
        debug_image = padded_image.copy()
        cv2.rectangle(debug_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.drawContours(debug_image, [largest_contour], -1, (0, 0, 255), 2)
        cv2.imwrite('debug_rect.jpg', debug_image)
        cv2.imwrite('debug_mask.jpg', mask)
        
        return cropped
        
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None


def add_padding(image, padding_size=50, padding_color=(0, 0, 0)):
    """
    为图像添加padding

    Args:
        image: 输入图像
        padding_size: padding的大小（像素）
        padding_color: padding的颜色，默认白色

    Returns:
        添加padding后的图像
    """
    height, width = image.shape[:2]

    # 创建新的图像（带padding）
    padded_image = np.full((
        height + 2 * padding_size,
        width + 2 * padding_size,
        3 if len(image.shape) == 3 else 1
    ), padding_color, dtype=np.uint8)

    # 将原图复制到padding后的图像中心
    padded_image[padding_size:padding_size + height,
    padding_size:padding_size + width] = image

    return padded_image
def process_folder(folder_path, output_folder):
    """
    Process all images in a folder, save the largest inscribed rectangle for each image.

    Args:
        folder_path (str): Path to the folder containing images.
        output_folder (str): Path to the folder where cropped images will be saved.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            try:
                cropped_image = find_largest_inscribed_rectangle(image_path)
                
                # 使用cv2.imwrite保存图像
                output_path = os.path.join(output_folder, f"cropped_{filename}")
                cv2.imwrite(output_path, cropped_image)
                
                print(f"Processed {filename} and saved cropped image as {output_path}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")


# Example usage
# Input folder containing images
input_folder = r'D:\dataset\windturbine-detection\turbine\testyp-10700-output'
# Output folder where cropped images will be saved
output_folder = r'D:\dataset\windturbine-detection\turbine\testyp-10700output'


process_folder(input_folder, output_folder)
