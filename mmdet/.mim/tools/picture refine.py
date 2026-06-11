

import cv2
import numpy as np
import os


def remove_black_padding(image):
    # 将图片转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用 Otsu 方法进行阈值处理，得到二值图像
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 使用形态学操作（膨胀和腐蚀）来处理噪声
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 查找非零像素（即非黑色像素）
    coords = cv2.findNonZero(binary)

    # 如果没有找到非零像素，则返回原始图片
    if coords is None:
        return image

    # 获取包围非零像素的最小矩形边界
    x, y, w, h = cv2.boundingRect(coords)

    # 裁剪图片
    cropped = image[y:y + h, x:x + w]

    # 再次检查裁剪后的图像是否包含黑色像素
    gray_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    _, binary_cropped = cv2.threshold(gray_cropped, 1, 255, cv2.THRESH_BINARY)

    # 设置最大迭代次数
    max_iterations =40
    iteration = 40

    # 循环裁剪，直到裁剪后的图像不包含任何黑色像素
    while np.any(binary_cropped == 0) and iteration < max_iterations:
        # 找到新的非零像素
        coords = cv2.findNonZero(binary_cropped)
        if coords is None:
            break
        x, y, w, h = cv2.boundingRect(coords)
        new_cropped = cropped[y:y + h, x:x + w]

        # 检查裁剪前后图像大小变化
        if new_cropped.shape == cropped.shape:
            break

        cropped = new_cropped
        gray_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        _, binary_cropped = cv2.threshold(gray_cropped, 1, 255, cv2.THRESH_BINARY)
        iteration += 1

    return cropped


def process_images(input_folder, output_folder):
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取输入文件夹中的所有图片文件
    image_files = [f for f in os.listdir(input_folder) if
                   f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    for image_file in image_files:
        # 构建完整的图片路径
        image_path = os.path.join(input_folder, image_file)

        # 读取图片
        image = cv2.imread(image_path)

        if image is None:
            print(f"无法读取图像: {image_path}")
            continue

        # 移除黑色填充
        cropped_image = remove_black_padding(image)

        # 构建输出图片路径
        output_path = os.path.join(output_folder, image_file)

        # 保存处理后的图片
        if cv2.imwrite(output_path, cropped_image):
            print(f"已保存处理后的图片: {output_path}")
        else:
            print(f"无法保存图片: {output_path}")


if __name__ == "__main__":
    input_folder = 'D:/dataset/windturbine-detection/turbine/testyp-10700'  # 输入图片文件夹路径
    output_folder = 'D:/dataset/windturbine-detection/turbine/testyp-10700-output'  # 输出图片文件夹路径

    process_images(input_folder, output_folder)