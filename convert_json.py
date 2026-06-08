import json
import os

def convert_labels_to_yolo_data(json_path: str) -> dict:
    """
    读取 label.json 文件，并将其转换为 main.py 所需的 yolo_detected_data 格式。

    - 根据 category_id 映射到积木名称 (code1_1, code2_1, etc.)。
    - 将坐标从毫米(mm)转换为米(m)。
    - 提取 z_angle_rad 作为旋转角度。

    :param json_path: labels.json 文件的路径。
    :return: 格式化后的字典。
    """
    # 检查文件是否存在
    if not os.path.exists(json_path):
        print(f"错误：文件未找到 -> {json_path}")
        return {}

    with open(json_path, 'r') as f:
        labels = json.load(f)

    # 用于为每个类别生成序号 (e.g., code1_1, code1_2)
    category_counters = {1: 1, 2: 1, 3: 1, 4: 1}
    
    yolo_detected_data = {}

    # 按 "id" 排序，以确保积木名称的分配顺序是可预测的
    sorted_labels = sorted(labels, key=lambda item: item['id'])

    for item in sorted_labels:
        cat_id = item.get("category_id")
        if cat_id not in category_counters:
            print(f"警告：跳过未知的 category_id: {cat_id}")
            continue

        # 1. 生成积木名称
        instance_num = category_counters[cat_id]
        block_name = f"code{cat_id}_{instance_num}"
        
        # 特殊情况：如果 category_id 为 4，则名称就是 "code4"
        if cat_id == 4:
            block_name = "code4"

        # 2. 提取坐标并从 mm 转换为 m
        # JSON中的 t_mm 是 [x, y, z] in mm
        coords_mm = item.get("t_mm")
        if not coords_mm:
            # 如果 t_mm 不存在，则使用 x, y, z 字段
            coords_mm = [item.get("x", 0) * 1000, item.get("y", 0) * 1000, item.get("z", 0) * 1000]

        coords_m = [c / 1000.0 for c in coords_mm]

        # 3. 提取旋转角度
        angle_rad = item.get("z_angle_rad", 0.0)

        # 4. 存入字典
        yolo_detected_data[block_name] = [coords_m, angle_rad]

        # 5. 更新计数器
        category_counters[cat_id] += 1
        
    return yolo_detected_data

if __name__ == "__main__":
    # 获取当前脚本所在目录，并构建 labels.json 的路径
    # 假设 labels.json 与 block_model_12.6_V2 目录在同一级
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir) # -> block_model_12.6_V2/
    json_file_path = os.path.join(project_root, "block_model_12.6_V2/labels(2).json")

    # --- 执行转换 ---
    detected_data = convert_labels_to_yolo_data(json_file_path)

    # --- 打印结果，方便复制粘贴 ---
    print("="*80)
    print("请将以下内容复制到 main.py 中：")
    print("="*80)
    
    # 使用 pprint 模块进行格式化输出，更美观
    import pprint
    
    # 为了让输出的格式完全符合你的要求，我们手动构建字符串
    print("yolo_detected_data = {")
    for i, (key, value) in enumerate(detected_data.items()):
        pos_list = value[0]
        angle = value[1]
        # 格式化坐标列表，保留5位小数
        pos_str = f"[{pos_list[0]:.5f}, {pos_list[1]:.5f}, {pos_list[2]:.5f}]"
        # 格式化角度，保留4位小数
        angle_str = f"{angle:.4f}"
        
        # 在行尾添加逗号
        print(f"    '{key}': [{pos_str}, {angle_str}],")
    print("}")
    
    print("\n" + "="*80)
