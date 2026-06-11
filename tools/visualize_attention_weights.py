import os
import sys
from pathlib import Path

# 添加当前目录到路径以确保模块可以被找到
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

# 导入必要的函数
try:
    from visualize_attention_on_image import process_single_image
except ImportError:
    print("警告: 无法导入visualize_attention_on_image模块，尝试使用绝对导入")
    try:
        # 尝试使用绝对导入
        from tools.visualize_attention_on_image import process_single_image
    except ImportError:
        print("错误: 无法导入process_single_image函数！")
        print("请确保visualize_attention_on_image.py文件在同一目录下或在Python路径中")
        raise

def process_images(image_paths, attention_weights, output_dir, kurtosis=None, gpu_id=None, data_dir=None, start_idx=0, kurtosis_weight=None):
    """处理多个图像，为每个图像创建注意力可视化"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置CUDA设备，如果提供了GPU ID
    if gpu_id is not None:
        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            print(f"已设置CUDA设备为GPU {gpu_id}")
        except Exception as e:
            print(f"设置CUDA设备时出错: {e}")
    
    # 确保注意力权重是列表
    if not isinstance(attention_weights, list):
        attention_weights = [attention_weights]
    
    # 确保kurtosis是列表或None
    if kurtosis is not None and not isinstance(kurtosis, list):
        kurtosis = [kurtosis]
    
    # 确保kurtosis_weight是列表或None
    if kurtosis_weight is not None and not isinstance(kurtosis_weight, list):
        kurtosis_weight = [kurtosis_weight]
    
    # 处理每个图像
    results = []
    for img_idx, image_path in enumerate(image_paths):
        print(f"\n处理图像 {img_idx+1}/{len(image_paths)}: {os.path.basename(image_path)}")
        
        # 为当前图像创建输出目录
        img_output_dir = os.path.join(output_dir, f"image_{img_idx+1}")
        os.makedirs(img_output_dir, exist_ok=True)
        
        # 处理当前图像
        try:
            result = process_single_image(
                image_path, 
                img_output_dir, 
                attention_weights, 
                kurtosis=kurtosis, 
                data_dir=data_dir,
                start_idx=start_idx,
                kurtosis_weight=kurtosis_weight
            )
            results.append(result)
        except Exception as e:
            print(f"处理图像 {image_path} 时出错: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    # 返回处理结果
    return results

# 添加主函数，避免被其他模块导入时自动运行
if __name__ == "__main__":
    print("请提供必要的参数来调用process_images函数")
    # 这里可以添加简单的命令行参数解析示例代码 