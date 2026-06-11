#本代码是用于裁剪最大内接旋转矩形框的
import os
from PIL import Image, ImageOps
import numpy as np
import cv2
import time


def find_largest_inscribed_rectangle(image_path, timeout=30):
    try:
        start_time = time.time()

        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")

        # 添加padding
        padded_image = add_padding(image, padding_size=50)
        gray = cv2.cvtColor(padded_image, cv2.COLOR_BGR2GRAY)

        # 二值化处理
        _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        # 找到轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return image

        # 找到最大轮廓
        largest_contour = max(contours, key=cv2.contourArea)

        # 创建掩码
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [largest_contour], -1, 255, -1)

        # 计算掩码的中心点
        M = cv2.moments(mask)
        if M["m00"] == 0:
            return image
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])

        # 初始化最佳结果
        max_rect_area = 0
        best_rect = None
        best_rotated_image = None
        best_angle = 0

        # 在180度范围内每10度搜索一次
        for angle in range(0, 180, 1):
            if time.time() - start_time > timeout:
                break

            # 旋转掩码
            M = cv2.getRotationMatrix2D((mask.shape[1] / 2, mask.shape[0] / 2), angle, 1.0)
            h, w = mask.shape
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            new_w = int((h * sin) + (w * cos))
            new_h = int((h * cos) + (w * sin))

            M[0, 2] += (new_w / 2) - w / 2
            M[1, 2] += (new_h / 2) - h / 2

            rotated_mask = cv2.warpAffine(mask, M, (new_w, new_h))

            # 计算旋转后的中心点
            rotated_center = np.dot(M, [center_x, center_y, 1])
            center_x_rot = int(rotated_center[0])
            center_y_rot = int(rotated_center[1])

            # 从中心向外扩展找最大矩形
            max_radius = min(
                center_x_rot,
                center_y_rot,
                new_w - center_x_rot,
                new_h - center_y_rot
            )

            # 二分查找最大有效半径
            left, right = 0, max_radius
            valid_radius = 0

            while left <= right:
                radius = (left + right) // 2
                rect_x = center_x_rot - radius
                rect_y = center_y_rot - radius
                rect_w = 2 * radius
                rect_h = 2 * radius

                if (rect_x >= 0 and rect_y >= 0 and
                        rect_x + rect_w < new_w and rect_y + rect_h < new_h):
                    rect_mask = rotated_mask[rect_y:rect_y + rect_h, rect_x:rect_x + rect_w]
                    if np.all(rect_mask == 255):
                        valid_radius = radius
                        left = radius + 1
                    else:
                        right = radius - 1
                else:
                    right = radius - 1

            rect_area = 4 * valid_radius * valid_radius
            if rect_area > max_rect_area:
                max_rect_area = rect_area
                best_angle = angle
                best_rect = (center_x_rot - valid_radius, center_y_rot - valid_radius,
                             2 * valid_radius, 2 * valid_radius)
                best_rotated_image = cv2.warpAffine(padded_image, M, (new_w, new_h))

        if best_rect is None:
            return image

        # 裁剪最终图像
        x, y, w, h = best_rect
        cropped = best_rotated_image[y:y + h, x:x + w]

        # 检查裁剪结果
        if cropped is None or cropped.size == 0 or min(cropped.shape[:2]) < 10:
            return image

        return cropped

    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return image


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

                if cropped_image is not None and cropped_image.size > 0:
                    output_path = os.path.join(output_folder, f"cropped_{filename}")
                    cv2.imwrite(output_path, cropped_image)
                    print(f"Processed {filename} and saved cropped image as {output_path}")
                else:
                    print(f"Failed to process {filename}: Invalid result")

            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")


# Example usage
# Input folder containing images
input_folder = r'D:\dataset\windturbine-detection\turbine\testyp-10700-output'
# Output folder where cropped images will be saved
output_folder = r'D:\dataset\windturbine-detection\turbine\testyp-10700output1'

process_folder(input_folder, output_folder)
