import json
import copy


def adjust_categories_without_background(input_json_path, output_json_path):
    """
    移除 background 类别并重新调整其他类别的 ID

    Args:
        input_json_path: 输入的标注文件路径
        output_json_path: 输出的标注文件路径
    """
    # 读取原始标注文件
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 创建新的数据结构
    new_data = copy.deepcopy(data)

    # 获取类别列表
    categories = data['categories']

    # 创建新的类别列表，移除background并调整ID
    new_categories = []
    id_mapping = {}  # 用于存储原始ID到新ID的映射
    new_id = 0

    for cat in categories:
        if cat['name'].lower() != 'background':
            # 记录ID映射关系
            id_mapping[cat['id']] = new_id
            # 创建新的类别对象
            new_cat = copy.deepcopy(cat)
            new_cat['id'] = new_id
            new_categories.append(new_cat)
            new_id += 1

    # 更新类别列表
    new_data['categories'] = new_categories

    # 更新标注中的类别ID
    if 'annotations' in new_data:
        for ann in new_data['annotations']:
            ann['category_id'] = id_mapping[ann['category_id']]

    # 保存新的标注文件
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)

    # 打印统计信息
    print(f"处理文件: {input_json_path}")
    print(f"原始类别数量: {len(data['categories'])}")
    print(f"处理后类别数量: {len(new_data['categories'])}")
    print(f"类别ID映射: {id_mapping}")
    print("完成!\n")


if __name__ == '__main__':
    # 处理训练集
    adjust_categories_without_background(
        '../WindBlade-30K/annotations/train.json',
        '../WindBlade-30K/annotations/train_no_bg.json'
    )

    # 处理验证集
    adjust_categories_without_background(
        '../WindBlade-30K/annotations/val.json',
        '../WindBlade-30K/annotations/val_no_bg.json'
    )

    # 处理测试集
    adjust_categories_without_background(
        '../WindBlade-30K/annotations/test.json',
        '../WindBlade-30K/annotations/test_no_bg.json'
    )