#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
像素坐标到桌面平面坐标的 2D 映射脚本。
使用单应性矩阵进行平面变换。
"""

import numpy as np
import cv2
import json

class PlaneMapper:
    """
    像素坐标与桌面平面坐标之间的 2D 映射类。
    使用单应性矩阵进行变换。
    """
    
    def __init__(self, pixel_points, world_points):
        """
        初始化映射器。
        
        Args:
            pixel_points: 像素坐标点列表，格式 [(x1, y1), (x2, y2), ...]
            world_points: 对应的桌面平面坐标点列表，格式 [(x1, y1), (x2, y2), ...]
                          单位：米或毫米（保持一致）
        """
        if len(pixel_points) != len(world_points) or len(pixel_points) < 4:
            raise ValueError("需要至少 4 对对应点")
        
        # 转换为 numpy 数组
        self.pixel_points = np.array(pixel_points, dtype=np.float32)
        self.world_points = np.array(world_points, dtype=np.float32)
        
        # 计算单应性矩阵（像素到世界）
        self.H_pixel_to_world, _ = cv2.findHomography(self.pixel_points, self.world_points)
        
        # 计算逆矩阵（世界到像素）
        self.H_world_to_pixel = np.linalg.inv(self.H_pixel_to_world)
        
        print("单应性矩阵计算完成")
        print("像素到世界矩阵 H_pixel_to_world:")
        print(self.H_pixel_to_world)
        print("世界到像素矩阵 H_world_to_pixel:")
        print(self.H_world_to_pixel)
    
    def pixel_to_world(self, pixel_x, pixel_y):
        """
        将像素坐标转换为桌面平面坐标。
        
        Args:
            pixel_x, pixel_y: 像素坐标
        
        Returns:
            (world_x, world_y): 桌面平面坐标
        """
        pixel_point = np.array([[pixel_x, pixel_y]], dtype=np.float32)
        world_point = cv2.perspectiveTransform(pixel_point.reshape(1, 1, 2), self.H_pixel_to_world)
        return world_point[0, 0]
    
    def world_to_pixel(self, world_x, world_y):
        """
        将桌面平面坐标转换为像素坐标。
        
        Args:
            world_x, world_y: 桌面平面坐标
        
        Returns:
            (pixel_x, pixel_y): 像素坐标
        """
        world_point = np.array([[world_x, world_y]], dtype=np.float32)
        pixel_point = cv2.perspectiveTransform(world_point.reshape(1, 1, 2), self.H_world_to_pixel)
        return pixel_point[0, 0]


def convert_json_to_desktop_format(json_path, desktop_z=0.014):
    """
    从 JSON 文件读取积木数据，将基座坐标转换为桌面坐标格式。
    桌面坐标假设为 (x, y)，z 设置为桌面高度。
    
    Args:
        json_path: JSON 文件路径
        desktop_z: 桌面 Z 坐标（默认 0.014 米）
    
    Returns:
        desktop_data: 转换后的字典，格式与 main.py 中的 yolo_detected_data 相同
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    desktop_data = {}
    for key, value in data.items():
        x, y, _ = value[0]  # 提取基座 x, y，忽略 z
        angle = value[1]    # 旋转角度不变
        desktop_data[key] = [[x, y, desktop_z], angle]
    
    return desktop_data

def print_desktop_data(desktop_data):
    """
    打印桌面数据，格式与 main.py 中的 yolo_detected_data 相同。
    """
    print("yolo_detected_data = {")
    for key, value in desktop_data.items():
        coords, angle = value
        print(f"    '{key}': [{coords}, {angle}],")
    print("}")
# 示例使用
def main():
    # 示例对应点（需要根据实际标定替换）
    # 像素坐标（图像中的点）
    pixel_points = [
        (333.4, 183.5),  # 左上
        (293.5, 313.6),  # 右上
        (219.8, 75.8),  # 右下
        (115.2, 383.7),  # 左下
    ]
    
    # 对应的桌面平面坐标（单位：米）
    world_points = [
        (0.30, 0.02),   # 左上
        (0.20, 0.05),   # 右上
        (0.40, 0.12),   # 右下
        (0.15, 0.17),   # 左下
    ]

         
    
    # 创建映射器
    mapper = PlaneMapper(pixel_points, world_points)
    
    # 测试转换
    print("\n测试转换:")
    
    # 像素到世界
    pixel_test = (402.7, 294.5)
    world_result = mapper.pixel_to_world(*pixel_test)
    print(f"像素坐标 {pixel_test} -> 桌面坐标 {world_result}")
    
    # 世界到像素
    world_test = (0.25, 0.0279)
    pixel_result = mapper.world_to_pixel(*world_test)
    print(f"桌面坐标 {world_test} -> 像素坐标 {pixel_result}")
    
    # 验证逆变换
    back_pixel = mapper.world_to_pixel(*world_result)
    print(f"逆变换验证: {pixel_test} -> {world_result} -> {back_pixel}")

    # 新增：转换 JSON 到桌面格式并打印
    json_path = "yolo_output_20251209_184721.json"  # 替换为实际路径
    desktop_data = convert_json_to_desktop_format(json_path)
    print_desktop_data(desktop_data)



if __name__ == "__main__":
    main()