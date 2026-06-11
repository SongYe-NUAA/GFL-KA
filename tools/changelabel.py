import json

def update_coco_categories(input_file, output_file, category_map, compact=False):
    """
    更新COCO类别并优化输出文件大小
    
    Args:
        input_file: 输入JSON文件路径
        output_file: 输出JSON文件路径
        category_map: 类别映射字典，格式：
            - 删除类别: {old_name: None}
            - 重命名/合并类别: {old_name: (new_name, new_id)}
        compact: 是否使用紧凑格式输出
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 处理类别和标注
    id_map = {}
    new_categories = []
    used_names = set()  # 记录已使用的类别名称
    name_to_id = {}    # 记录类别名称到ID的映射
    
    # 按原始顺序处理类别
    for category in data['categories']:
        current_id = category['id']
        current_name = category['name']
        
        if current_name in category_map:
            mapping = category_map[current_name]
            if mapping is None:
                continue
            new_name, new_id = mapping
            
            # 如果这个新类别名称已经存在，使用已存在的ID
            if new_name in used_names:
                id_map[current_id] = name_to_id[new_name]
            else:
                # 新类别第一次出现
                id_map[current_id] = new_id
                name_to_id[new_name] = new_id
                used_names.add(new_name)
                new_categories.append({
                    'id': new_id,
                    'name': new_name
                })
        else:
            # 保持原有类别
            id_map[current_id] = current_id
            if current_name not in used_names:
                new_categories.append(category)
                used_names.add(current_name)
                name_to_id[current_name] = current_id
    
    # 只保留有效的标注
    new_annotations = [
        {k: v for k, v in anno.items() if k != 'segment_info'}
        for anno in data['annotations']
        if anno['category_id'] in id_map
    ]
    
    # 更新类别ID
    for anno in new_annotations:
        anno['category_id'] = id_map[anno['category_id']]
    
    # 构建优化后的数据
    processed_data = {
        'images': data['images'],
        'categories': new_categories,
        'annotations': new_annotations
    }
    
    # 保存
    with open(output_file, 'w', encoding='utf-8') as f:
        if compact:
            json.dump(processed_data, f, ensure_ascii=False, separators=(',', ':'))
        else:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)

# Example usage
if __name__ == "__main__":
    # Define the mapping of old category names to new ones
    # Format: {old_name: (new_name, new_id)} or {old_name: None} to remove
    category_map = {
        'swell': None,  # 删除 swell 类
        'teeth': None,  # 删除 teeth 类
        'demould': ('hole', 0),  # demould 合并到 hole 类
        'oil': ('stain', 3),  # oil 合并到 stain 类
    }

    # Paths to the input and output files
    input_file = r'C:\Users\Administrator\Desktop\learning\dataset\WindBlade-30K（refine）\annotations\nobackground\train.json'
    output_file = r'C:\Users\Administrator\Desktop\learning\dataset\WindBlade-30K（refine）\annotations\noswell_oil_demould_teeth\train.json'

    # Update the categories and save the result
    update_coco_categories(input_file, output_file, category_map)


