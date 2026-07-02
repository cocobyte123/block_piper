'''
好的，但是我希望现在这个变量定义，pick的部分和place的部分分开，或者说我们现在不知道place需要到达的位置，然后你可以看到，一共有10块积木，但是有四个类型，这个也要标出来，place的地方是我们计算出来的。现在我们重新初始化变量的定义。1、积木的信息，积木id，积木类型，积木的旋转姿态，积木的中心点。2、积木的预制体，就是类型1的积木，它的长宽高是多少，3、一个关于积木怎么搭建起来的数据解构，存放具体积木id，那些积木在第一层，那些积木第二层，第三层，第四层。（一共就四层）4、定义夹爪的固定向量，用于后续预抓取点的操作。
另外可能还会有一些变量，你自行决定。我可以先跟你说各个点位的计算，如果你有不理解的请告诉我：
期望抓取位置=物体中心+夹爪的固定偏差
修正位置=期望位置+误差修正向量
预抓取位置=修正位置+预留高度
同理放置的情况
'''
import numpy as np
import os
import matplotlib
# --- 【核心修改】强制使用 'Agg' 后端，确保在任何环境下都能正确保存文件 ---
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from visual_block import BlockVisualizer
from typing import Union

class TaskScheduler:
    """
    根据分离的“原型”、“实例”和“架构”数据，动态计算所有任务参数。
    """
    def __init__(self, first_block_target_pos: list = [0.0, 0.0, 0.0]):
        """
        初始化规划器。
        :param first_block_target_pos: 第一个积木的精确放置位置 [x, y, z]。默认 [0.0, 0.0, 0.0]。
        """
        print("[TaskScheduler] 初始化...")
        # --- 新增：保存构建原点，用于相对坐标可视化 ---
        self.build_origin = np.array(first_block_target_pos)

        # 1. 定义夹爪、预设高度等固定参数
        self._define_gripper_and_build_params()
        
        # 2. 定义积木的“预制体”或“原型” (尺寸信息)
        self._define_block_prototypes()
        
        # 3. 定义10个积木“实例”的初始状态 (来自 block_challenge.xml)
        self._define_block_instances()
        
        # 4. 定义最终成品的“搭建架构” (分层结构)
        self._define_build_architecture()
        
        # 5. 核心计算：根据架构和原型，计算所有积木的最终放置位置
        self._calculate_all_placements(np.array(first_block_target_pos))
        
        print(f"[TaskScheduler] 所有积木的放置位置已计算完毕。")

        # --- 【核心修改】初始化可视化器时，传入 gripper_offset ---
        self.visualizer = BlockVisualizer(
            instances=self.instances,
            prototypes=self.prototypes,
            architecture=self.architecture,
            build_origin=self.build_origin,
            gripper_offset=self.gripper_offset  # 将夹爪偏移传递给可视化工具
        )

    def _define_gripper_and_build_params(self):
        """1. 定义夹爪的固定向量及其他构建参数"""
        self.gripper_offset = np.array([0.0, 0.0, 0.01])  # 示例：夹爪中心在末端法兰盘Z轴上方x cm
         # --- 新增：定义放置区域裁剪边界 ---
        self._define_placement_region()
        
        # 移除全局默认高度和距离，现在它们是每个积木的个性化参数
        print(f"  -> 夹爪偏移: {self.gripper_offset}")

    def _define_placement_region(self):
        """【新增】定义放置区域的裁剪边界 (预留接口，用户可修改)"""
        self.placement_region = {
            "x_min": -0.20,   # X轴最小值 (米)
            "x_max": 0.32,   # X轴最大值 (米)
            "z_min": 0.015,   # Z轴最小值 (米)
            "z_max": 0.40  # Z轴最大值 (米)
        }
        # 注意：Y轴不裁剪，因为所有积木的Y坐标都是一样的


    def _clip_to_placement_region(self, pos: np.ndarray) -> np.ndarray:
        """【新增】对位置进行放置区域裁剪 (只裁剪X和Z轴)"""
        original_pos_str = f"[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]"
        clipped_pos = pos.copy()
        clipped_pos[0] = np.clip(clipped_pos[0], self.placement_region["x_min"], self.placement_region["x_max"])
        clipped_pos[2] = np.clip(clipped_pos[2], self.placement_region["z_min"], self.placement_region["z_max"])
        
        if not np.array_equal(pos, clipped_pos):
            clipped_pos_str = f"[{clipped_pos[0]:.3f}, {clipped_pos[1]:.3f}, {clipped_pos[2]:.3f}]"
            print(f"    -> \033[93m[裁剪] 位置 {original_pos_str} 已被裁剪为 {clipped_pos_str}\033[0m")
            
        return clipped_pos

    def _define_block_prototypes(self):
        """2. 定义积木的预制体 (类型 -> 尺寸)"""
        self.prototypes = {
            "type1": {"size": np.array([0.06, 0.03, 0.03])},    # 对应 code1
            "type2": {"size": np.array([0.07, 0.03, 0.03])},   # 对应 code2 (梯形近似为长方体)
            "type3": {"size": np.array([0.03, 0.03, 0.06])},   # 对应 code3
            "type4": {"size": np.array([0.05, 0.03, 0.03])},   # 对应 code4
        }
        print("  -> 已定义4种积木原型 (尺寸)。")

    def _define_block_instances(self):
        """
        【核心修改】为每个积木实例添加 "descent_distance" (下放距离) 超参数。
        """

        # --- 新增：定义统一的初始位置和旋转角度 ---
        offset_z=0.11
        unified_initial_pos = np.array([0.300, 0.003, offset_z])  # 统一的初始位置，所有积木使用这个
        unified_initial_angle = 0.0  # 统一的初始旋转角度（弧度），所有积木使用这个
        place_angle = -np.pi/2  # 统一的放置旋转角度（弧度）
        
        self.instances = {
            # 积木ID: {..., pre_place_height, descent_distance, ...}
            "code1_1": {"type": "type1", "initial_pos": unified_initial_pos, "initial_angle_rad": unified_initial_angle, "place_angle_rad": place_angle, "pre_grasp_height": 0.08, "pre_place_height": 0.08, "descent_distance": 0.075, "lift_direction": "up", "lift_distance": 0.05, "placement_error_x": 0.0},
            "code1_2": {"type": "type1", "initial_pos": unified_initial_pos, "initial_angle_rad": unified_initial_angle, "place_angle_rad": place_angle, "pre_grasp_height": 0.08, "pre_place_height": 0.08, "descent_distance": 0.08, "lift_direction": "up", "lift_distance": 0.05, "placement_error_x": 0.0},
            "code1_3": {"type": "type1", "initial_pos": unified_initial_pos, "initial_angle_rad": unified_initial_angle, "place_angle_rad": place_angle, "pre_grasp_height": 0.08, "pre_place_height": 0.08, "descent_distance": 0.08, "lift_direction": "up", "lift_distance": 0.05, "placement_error_x": 0.0},
            "code1_4": {"type": "type1", "initial_pos": unified_initial_pos, "initial_angle_rad": unified_initial_angle, "place_angle_rad": place_angle, "pre_grasp_height": 0.08, "pre_place_height": 0.08, "descent_distance": 0.08, "lift_direction": "up", "lift_distance": 0.02, "placement_error_x": 0.0},
            "code2_1": {"type": "type2", "initial_pos": unified_initial_pos, "initial_angle_rad": unified_initial_angle, "place_angle_rad": place_angle, "pre_grasp_height": 0.08, "pre_place_height": 0.08, "descent_distance": 0.07, "lift_direction": "up", "lift_distance": 0.05, "placement_error_x": 0.0},
            
            # --- 特殊积木：定义放置旋转角度 --- 0.02 是为了避免与桌面碰撞
            "code3_1": {"type": "type3", "initial_pos": unified_initial_pos+np.array([0.00, 0.00, 0.02]), "initial_angle_rad": unified_initial_angle, "place_angle_rad": place_angle,"pre_grasp_height": 0.08, "pre_place_height": 0.08, "descent_distance": 0.06, "lift_direction": "up", "lift_distance": 0.05, "placement_error_x": 0.0},
            "code3_2": {"type": "type3", "initial_pos": unified_initial_pos+np.array([0.00, 0.00, 0.02]), "initial_angle_rad": unified_initial_angle, "place_angle_rad": place_angle,"pre_grasp_height": 0.08, "pre_place_height": 0.08, "descent_distance": 0.06, "lift_direction": "right", "lift_distance": 0.05, "placement_error_x": 0.0},
            
            "code2_2": {"type": "type2", "initial_pos": unified_initial_pos, "initial_angle_rad": unified_initial_angle, "place_angle_rad": place_angle, "pre_grasp_height": 0.08, "pre_place_height": 0.05, "descent_distance": 0.05, "lift_direction": "up", "lift_distance": 0.02, "placement_error_x": 0.0},
            "code2_3": {"type": "type2", "initial_pos": unified_initial_pos, "initial_angle_rad": unified_initial_angle, "place_angle_rad": place_angle, "pre_grasp_height": 0.08, "pre_place_height": 0.08, "descent_distance": 0.08, "lift_direction": "right", "lift_distance": 0.06, "placement_error_x": 0.0},
            "code4":   {"type": "type4", "initial_pos": unified_initial_pos, "initial_angle_rad": unified_initial_angle, "place_angle_rad": place_angle, "pre_grasp_height": 0.08, "pre_place_height": 0.03, "descent_distance": 0.03, "lift_direction": "right", "lift_distance": 0.08, "placement_error_x": 0.0},
        }
        # self.instances = {
        #     # 积木ID: {..., pre_place_height, descent_distance, ...}
        #     "code1_1": {"type": "type1", "initial_pos": np.array([0.32, -0.187, offset_z+0.017]), "initial_angle_rad": 0.0, "place_angle_rad": 0.0, "pre_grasp_height": 0.08, "pre_place_height": 0.08, "descent_distance": 0.08, "lift_direction": "up", "lift_distance": 0.05, "placement_error_x": 0.0},
        #     "code1_2": {"type": "type1", "initial_pos": np.array([0.41, -0.163, offset_z+0.017]), "initial_angle_rad": 0.5, "place_angle_rad": 0.0, "pre_grasp_height": 0.08, "pre_place_height": 0.08, "descent_distance": 0.08, "lift_direction": "up", "lift_distance": 0.05, "placement_error_x": 0.0},
        #     "code1_3": {"type": "type1", "initial_pos": np.array([0.300, 0.003, offset_z+0.017]), "initial_angle_rad": 0.0, "place_angle_rad": 0.0, "pre_grasp_height": 0.08, "pre_place_height": 0.08, "descent_distance": 0.08, "lift_direction": "up", "lift_distance": 0.05, "placement_error_x": 0.0},
        #     "code1_4": {"type": "type1", "initial_pos": np.array([0.340, -0.117, offset_z+0.017]), "initial_angle_rad": -0.9, "place_angle_rad": 0.0, "pre_grasp_height": 0.08, "pre_place_height": 0.08, "descent_distance": 0.08, "lift_direction": "up", "lift_distance": 0.02, "placement_error_x": 0.0},
        #     "code2_1": {"type": "type2", "initial_pos": np.array([0.260, -0.117, offset_z+0.017]), "initial_angle_rad": 0.0, "place_angle_rad": 0.0, "pre_grasp_height": 0.08, "pre_place_height": 0.08, "descent_distance": 0.08, "lift_direction": "up", "lift_distance": 0.05, "placement_error_x": 0.0},
            
        #     # --- 特殊积木：定义放置旋转角度 --- 0.02 是为了避免与桌面碰撞
        #     "code3_1": {"type": "type3", "initial_pos": np.array([0.25, 0.093, offset_z+0.025]), "initial_angle_rad": np.pi / 2, "place_angle_rad": np.pi, "pre_grasp_height": 0.08, "pre_place_height": 0.08, "descent_distance": 0.08-0.045, "lift_direction": "up", "lift_distance": 0.05, "placement_error_x": 0.0},
        #     "code3_2": {"type": "type3", "initial_pos": np.array([0.400, 0.113, offset_z+0.025]), "initial_angle_rad": -np.pi / 2, "place_angle_rad": -np.pi, "pre_grasp_height": 0.08, "pre_place_height": 0.08, "descent_distance": 0.08-0.04, "lift_direction": "right", "lift_distance": 0.05, "placement_error_x": 0.0},
            
        #     "code2_2": {"type": "type2", "initial_pos": np.array([0.415, -0.030, offset_z+0.017]), "initial_angle_rad": -1.0, "place_angle_rad": 0.0, "pre_grasp_height": 0.08, "pre_place_height": 0.05, "descent_distance": 0.05, "lift_direction": "up", "lift_distance": 0.02, "placement_error_x": 0.0},
        #     "code2_3": {"type": "type2", "initial_pos": np.array([0.410, 0.038, offset_z+0.017]), "initial_angle_rad": 0.0, "place_angle_rad": 0.0, "pre_grasp_height": 0.08, "pre_place_height": 0.08, "descent_distance": 0.08, "lift_direction": "right", "lift_distance": 0.06, "placement_error_x": 0.0},
        #     "code4":   {"type": "type4", "initial_pos": np.array([0.320, 0.123, offset_z+0.017-0.01]), "initial_angle_rad": 0.0, "place_angle_rad": 0.0, "pre_grasp_height": 0.08, "pre_place_height": 0.03, "descent_distance": 0.03, "lift_direction": "right", "lift_distance": 0.08, "placement_error_x": 0.0},
        # }


        self.code4_x_offset=0.06  # code4 特殊偏移量
        print(f"  -> 已定义 {len(self.instances)} 个积木实例的初始状态（含个性化参数）。")

    def set_active_blocks_from_detection(self, detected_dict: dict):
        """
        【新方法】根据YOLO检测结果，从空列表开始设置活跃积木，只添加检测到的积木，并更新它们的初始状态。
        :param detected_dict: 字典，格式为 {"积木ID": [[x, y, z], angle_rad], ...}
        """
        print("\n--- 正在根据YOLO检测设置活跃积木（从空列表开始） ---")
        self.active_blocks = set()  # 从空列表开始
        updated_count = 0
        for block_id, data in detected_dict.items():
            if block_id in self.instances:
                if isinstance(data, list) and len(data) == 2:
                    pos, angle = data
                    # 只更新X, Y坐标，保持原有的Z坐标不变
                    current_z = self.instances[block_id]["initial_pos"][2]
                    self.instances[block_id]["initial_pos"] = np.array([pos[0], pos[1], current_z])
                    self.instances[block_id]["initial_angle_rad"] = angle
                    self.active_blocks.add(block_id)  # 添加到活跃列表
                    print(f"  -> 已添加并更新 '{block_id}': XY位置 -> [{pos[0]:.3f}, {pos[1]:.3f}], Z保持 -> {current_z:.3f}, 角度 -> {angle:.2f} rad")
                    updated_count += 1
                else:
                    print(f"  -> \033[93m警告：'{block_id}' 的数据格式不正确，已跳过。\033[0m")
            else:
                print(f"  -> \033[93m警告：在任务实例中未找到积木ID '{block_id}'，已跳过。\033[0m")
        
        if updated_count > 0:
            print(f"--- 共设置 {updated_count} 个积木为活跃状态 ---\n")
        else:
            print("--- 未设置任何积木为活跃状态 ---\n")

    def update_initial_states_from_dict(self, initial_states_dict: dict):
        """
        【新方法】根据外部传入的字典，更新一个或多个积木的初始位置和旋转角度。
        :param initial_states_dict: 字典，格式为 {"积木ID": [[x, y, z], angle_rad], ...}
        """
        print("\n--- 正在根据外部数据更新积木初始状态 ---")
        updated_count = 0
        for block_id, data in initial_states_dict.items():
            if block_id in self.instances:
                if isinstance(data, list) and len(data) == 2:
                    pos, angle = data
                    # 【核心修改】只更新X, Y坐标，保持原有的Z坐标不变
                    current_z = self.instances[block_id]["initial_pos"][2]
                    self.instances[block_id]["initial_pos"] = np.array([pos[0], pos[1], current_z])
                    self.instances[block_id]["initial_angle_rad"] = angle
                    print(f"  -> 已更新 '{block_id}': XY位置 -> [{pos[0]:.3f}, {pos[1]:.3f}], Z保持 -> {current_z:.3f}, 角度 -> {angle:.2f} rad")
                    updated_count += 1
                else:
                    print(f"  -> \033[93m警告：'{block_id}' 的数据格式不正确，应为 [[x,y,z], angle]，已跳过。\033[0m")
            else:
                print(f"  -> \033[93m警告：在任务实例中未找到积木ID '{block_id}'，已跳过。\033[0m")
        
        if updated_count > 0:
            print(f"--- 共更新了 {updated_count} 个积木的状态 ---\n")
        else:
            print("--- 未更新任何积木状态 ---\n")


    def update_placement_error(self, block_id: str, error_x: float):
            """【新方法】由执行器调用，用于更新一个积木完成放置后的最终误差。"""
            if block_id in self.instances:
                self.instances[block_id]["placement_error_x"] = error_x
                print(f"  -> [误差记录] 已更新 '{block_id}' 的最终放置误差X为: {error_x * 1000:.2f} mm")
            else:
                print(f"  -> \033[91m[误差记录] 错误：尝试更新一个不存在的积木ID '{block_id}'\033[0m")

    def _define_build_architecture(self):
        """4. 定义积木的搭建结构 (分层)"""
        self.architecture = {
            "layer_1": ["code1_1", "code1_2", "code1_3", "code1_4"],
            "layer_2": ["code2_2", "code2_1"],  #"code3_2", "code3_1", 
            "layer_3": ["code2_3"],
            "layer_4": ["code4"],
        }
        
        # 【新增】定义哪些层需要执行最后的“整体推紧”动作
        self.COMPACTION_LAYERS = ["layer_1", "layer_2"]
        print(f"  -> 将对以下层执行整体推紧: {self.COMPACTION_LAYERS}")

        # 空隙参数：layer_1 和 layer_2 的积木间空隙 (用户可手动调整)
        self.layer_1_gaps = [0.001, 0.001, 0.001]  # layer_1 有4个积木，3个空隙
        self.layer_2_gaps = [0.005 ]  # layer_2 有4个积木，3个空隙
        
        print("  -> 已定义4层搭建架构。")

    def _calculate_all_placements(self, first_block_target_pos):
        """
        【核心逻辑修正】移除对角度的覆盖，只计算位置。
        """
        print("  -> [核心计算] 正在根据中心支撑逻辑计算所有放置位置...")
        
        # --- 第1层计算：以第一个积木为锚点，向右排列 ---
        layer_1_blocks = self.architecture["layer_1"]
        self._place_layer_horizontally_from_anchor(
            block_ids=layer_1_blocks, 
            anchor_block_id=layer_1_blocks[0], 
            anchor_block_center_pos=first_block_target_pos, 
            gaps=self.layer_1_gaps
        )
        
        # --- 第2层计算：找到第1层的几何中心，在其上方放置第2层 ---
        layer_2_blocks = self.architecture["layer_2"]
        l1_positions = [self.instances[bid]["place_pos"] for bid in layer_1_blocks]
        l1_min_x = min(p[0] for p in l1_positions)
        l1_max_x = max(p[0] for p in l1_positions)
        l1_center_x = (l1_min_x + l1_max_x) / 2.0
        top_z_layer1 = max(
            self.instances[bid]["place_pos"][2] + self.prototypes[self.instances[bid]["type"]]["size"][2] / 2.0
            for bid in layer_1_blocks
        )
        layer2_anchor_pos = np.array([l1_center_x, first_block_target_pos[1], top_z_layer1])
        self._place_layer_horizontally_around_center(
            block_ids=layer_2_blocks, 
            center_pos=layer2_anchor_pos, 
            gaps=self.layer_2_gaps
        )

        # --- 第3层计算：放在第2层的几何中心之上 ---
        layer_3_blocks = self.architecture["layer_3"]
        l2_positions = [self.instances[bid]["place_pos"] for bid in layer_2_blocks]
        l2_center_x = (min(p[0] for p in l2_positions) + max(p[0] for p in l2_positions)) / 2.0
        top_z_layer2 = max(
            self.instances[bid]["place_pos"][2] + self.prototypes[self.instances[bid]["type"]]["size"][2] / 2.0
            for bid in layer_2_blocks
        )
        layer3_pos = np.array([l2_center_x, first_block_target_pos[1], top_z_layer2])
        block_proto_l3 = self.prototypes[self.instances[layer_3_blocks[0]]["type"]]
        self.instances[layer_3_blocks[0]]["place_pos"] = layer3_pos + np.array([0, 0, block_proto_l3["size"][2] / 2.0])
        # 此处不再需要设置角度，因为它已在 _define_block_instances 中定义

        # --- 第4层计算：放在第3层的顶部 ---
        layer_4_blocks = self.architecture["layer_4"]
        center_block_l3_instance = self.instances[layer_3_blocks[0]]
        top_z_layer3 = max(
            self.instances[bid]["place_pos"][2] + self.prototypes[self.instances[bid]["type"]]["size"][2] / 2.0
            for bid in layer_3_blocks
        )
        layer4_pos = center_block_l3_instance["place_pos"].copy()
        layer4_pos[2] = top_z_layer3
        block_proto_l4 = self.prototypes[self.instances[layer_4_blocks[0]]["type"]]
        self.instances[layer_4_blocks[0]]["place_pos"] = layer4_pos + np.array([0, 0, block_proto_l4["size"][2] / 2.0])
        # 此处不再需要设置角度    def _place_layer_horizontally(self, block_ids, center_pos, gaps, angle_rad, z_offset):

        code3_y_offset = -0.01  # Y轴正方向偏移 1cm
        for block_id in ["code3_1", "code3_2"]:
            if block_id in self.instances and "place_pos" in self.instances[block_id]:
                original_pos = self.instances[block_id]["place_pos"]
                self.instances[block_id]["place_pos"][1] += code3_y_offset
                print(f"  -> \033[96m[硬编码偏移] '{block_id}' 的Y轴位置从 {original_pos[1]:.4f} 偏移到 {self.instances[block_id]['place_pos'][1]:.4f}\033[0m")

        

    def _place_layer_horizontally(self, block_ids, center_pos, gaps, angle_rad, z_offset):
        """
        辅助函数：将一层的积木横向排列，中线对齐，考虑空隙。
        【修改】现在正确处理Z坐标，将其作为支撑面。
        """
        if not block_ids:
            return
        
        total_width = sum(self.prototypes[self.instances[bid]["type"]]["size"][0] for bid in block_ids)
        total_gaps = sum(gaps)
        total_width += total_gaps
        
        current_x = center_pos[0] - total_width / 2.0
        
        for i, block_id in enumerate(block_ids):
            proto = self.prototypes[self.instances[block_id]["type"]]
            block_width = proto["size"][0]
            block_height = proto["size"][2]
            
            block_center_x = current_x + block_width / 2.0
            # 积木中心Z坐标 = 支撑面Z (center_pos[2]) + 自身高度的一半
            block_center_z = center_pos[2] + block_height / 2.0
            place_pos = np.array([block_center_x, center_pos[1], block_center_z])
            
            self.instances[block_id]["place_pos"] = place_pos
            self.instances[block_id]["place_angle_rad"] = angle_rad
            
            current_x += block_width
            if i < len(gaps):
                current_x += gaps[i]

    


    def _place_layer_horizontally_from_anchor(self, block_ids, anchor_block_id, anchor_block_center_pos, gaps):
        """【新方法】以一个指定的锚点积木为基准，向右排列一层积木。"""
        if anchor_block_id not in block_ids: return

        # 1. 放置锚点积木
        self.instances[anchor_block_id]["place_pos"] = np.array(anchor_block_center_pos)
        # 此处不再修改 place_angle_rad
        
        # 2. 计算后续积木的位置
        anchor_proto = self.prototypes[self.instances[anchor_block_id]["type"]]
        current_x_edge = anchor_block_center_pos[0] + anchor_proto["size"][0] / 2.0

        anchor_index = block_ids.index(anchor_block_id)
        for i in range(anchor_index + 1, len(block_ids)):
            block_id = block_ids[i]
            gap = gaps[i - 1]
            current_x_edge += gap
            
            proto = self.prototypes[self.instances[block_id]["type"]]
            block_width = proto["size"][0]
            
            block_center_x = current_x_edge + block_width / 2.0
            place_pos = np.array([block_center_x, anchor_block_center_pos[1], anchor_block_center_pos[2]])
            
            self.instances[block_id]["place_pos"] = place_pos
            # 此处不再修改 place_angle_rad
            
            current_x_edge += block_width

    def _place_layer_horizontally_around_center(self, block_ids, center_pos, gaps):
        """【原方法的修正版】将一层积木围绕一个中心点对称排列。"""
        if not block_ids: return
        
        total_width = sum(self.prototypes[self.instances[bid]["type"]]["size"][0] for bid in block_ids) + sum(gaps)
        current_x_edge = center_pos[0] - total_width / 2.0
        
        for i, block_id in enumerate(block_ids):
            proto = self.prototypes[self.instances[block_id]["type"]]
            block_width = proto["size"][0]
            block_height = proto["size"][2]
            
            block_center_x = current_x_edge + block_width / 2.0
            block_center_z = center_pos[2] + block_height / 2.0 # Z坐标是基于支撑面
            
            place_pos = np.array([block_center_x, center_pos[1], block_center_z])
            self.instances[block_id]["place_pos"] = place_pos
            # 此处不再修改 place_angle_rad
            
            current_x_edge += block_width
            if i < len(gaps):
                current_x_edge += gaps[i]

   
    def get_task_for_block(self, block_id: str, build_order: list, projected_grasp_error: float) -> Union[dict, None]:
        """
        【V3 核心重构】根据用户反馈，彻底修正推紧方向和最终补偿逻辑。
        1. slide_direction: 始终朝向“上一个”已放置的积木。
        2. is_final_block_in_compaction_layer: 正确识别“层末积木”（搭建顺序中的最后一个）。
        """
        if block_id not in self.instances:
            print(f"错误：积木 '{block_id}' 未在实例中定义。")
            return None
        
        instance = self.instances[block_id]
        proto = self.prototypes[instance['type']]
        
        # --- 计算抓取、放置、撤离点位 ---
        pick_pos = (instance["initial_pos"] + self.gripper_offset).tolist()
        pre_grasp_pos = (np.array(pick_pos) + np.array([0, 0, instance["pre_grasp_height"]])).tolist()
        place_pos = instance["place_pos"]
        pre_place_pos = (place_pos + np.array([0, 0, instance["pre_place_height"]])).tolist()
        final_lift_pos = np.array([pre_place_pos[0], pre_place_pos[1], place_pos[2]+0.03]).tolist()

        # --- 初始化任务标志和方向 ---
        slide_direction = 0.0
        is_final_block_in_compaction_layer = False
        
        current_block_layer_name = None
        current_block_layer_list = None
        current_block_index_in_build_order = -1

        # 在完整的 build_order 中找到当前积木，以确定其在层内的搭建顺序
        if block_id in build_order:
            current_block_index_in_build_order = build_order.index(block_id)

        # 找到当前积木所在的层定义
        for layer_name, block_list in self.architecture.items():
            if block_id in block_list:
                current_block_layer_name = layer_name
                current_block_layer_list = block_list
                break
        
        if current_block_layer_list:
            # 1. 判断是否为“层末积木”，触发后续的额外推紧
            # 条件: a) 当前层需要被推紧 b) 当前积木是该层中定义的最后一个
            if current_block_layer_name in self.COMPACTION_LAYERS and block_id == current_block_layer_list[-1]:
                is_final_block_in_compaction_layer = True
                print(f"  -> '{block_id}' 被识别为层 '{current_block_layer_name}' 的【层末积木】，将触发额外推紧。")
            
            # 2. 计算普通滑动方向 (如果不是该层搭建的第一个积木)
            # 找到层内第一个被搭建的积木
            first_built_block_in_layer = current_block_layer_list[0]
            if block_id != first_built_block_in_layer:
                # “上一个”搭建的积木，在 build_order 中的索引-1
                prev_block_id = build_order[current_block_index_in_build_order - 1]
                
                # 确保上一个积木和当前积木在同一层
                if prev_block_id in current_block_layer_list:
                    prev_block_pos_x = self.instances[prev_block_id]['place_pos'][0]
                    current_block_pos_x = place_pos[0]
                    
                    # 如果上一个在右边(X更大)，我们向右推(slide_direction=-1.0)
                    # 如果上一个在左边(X更小)，我们向左推(slide_direction=+1.0)
                    slide_direction = -1.0 if prev_block_pos_x > current_block_pos_x else 1.0

        if current_block_layer_name == "layer_1":
            final_lift_pos = np.array([pre_place_pos[0], pre_place_pos[1], place_pos[2] + 0.10]).tolist()
            print("  -> 第一层撤离点提高到放置高度上方10cm，减少低位回观察点的IK压力。")

        # 强制为 code3 积木设置为垂直放置
        if 'code3' in block_id:
            slide_direction = 0.0
            is_final_block_in_compaction_layer = False
            print(f"  -> 检测到 '{block_id}'，强制设置为垂直放置。")
        
        print(f"  -> 计算出的滑动方向 (1:向左推, -1:向右推): {slide_direction}")

        # --- 组装最终的任务包 ---
        task_package = {
            "id": block_id,
            "block_size": proto["size"],
            "pick_pos": pick_pos,
            "pre_grasp_pos": pre_grasp_pos,
            "pick_angle_rad": instance["initial_angle_rad"],
            "place_pos": place_pos.tolist(),
            "pre_place_pos": pre_place_pos,
            "place_angle_rad": instance["place_angle_rad"],
            "final_lift_pos": final_lift_pos,
            "slide_direction": slide_direction,
            "is_final_block_in_compaction_layer": is_final_block_in_compaction_layer,
        }

        # 如果是层末积木，则添加该层“层首积木”的位置信息
        if is_final_block_in_compaction_layer:
            # 层首积木是列表中的第一个
            first_block_in_layer_id = current_block_layer_list[0]
            first_block_pos = self.instances[first_block_in_layer_id]['place_pos']
            task_package['layer_first_block_pos'] = first_block_pos.tolist()
            print(f"  -> 为额外推紧任务添加信息：层首积木 '{first_block_in_layer_id}' 的位置。")

        return task_package

# =====================================================================================
# --- 测试主函数 ---
# =====================================================================================
if __name__ == "__main__":
    # 测试规划功能
    print("="*60)
    print("测试 TaskScheduler 规划功能")
    print("="*60)
    
    actual_first_block_pos = [0.1115, 0.380, 0.0152]
    print(f"\n使用真实的初始目标点位: {actual_first_block_pos}\n")
    
    scheduler = TaskScheduler(first_block_target_pos=actual_first_block_pos)
    
    print("\n--- 所有积木的放置位置和姿态 ---")
    for block_id, instance in scheduler.instances.items():
        if "place_pos" in instance and "place_angle_rad" in instance:
            pos = instance["place_pos"]
            angle = instance["place_angle_rad"]
            print(f"{block_id}: 位置 [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}], 角度 {angle:.3f} rad ({np.degrees(angle):.1f}°)")
        else:
            print(f"{block_id}: 放置信息未计算")
    
    # --- 【新增】打印抓取位置和姿态（结构与放置相同） ---
    print("\n--- 所有积木的抓取位置和姿态 ---")
    for block_id, instance in scheduler.instances.items():
        pos = instance["initial_pos"]
        angle = instance["initial_angle_rad"]
        print(f"{block_id}: 位置 [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}], 角度 {angle:.3f} rad ({np.degrees(angle):.1f}°)")
    
    # --- 【核心修改】调用新的可视化方法 ---
    print("\n--- 开始生成可视化图像 ---")
    # 创建一个文件夹来存放图像
    output_dir = os.path.join(os.path.dirname(__file__), "visualizations")
    os.makedirs(output_dir, exist_ok=True)
    print(f"图像将保存到: {output_dir}")

    # 1. 生成最终蓝图
    blueprint_path = os.path.join(output_dir, "0_final_blueprint.png")
    scheduler.visualizer.visualize_blueprint(save_path=blueprint_path)

    # 2. 为每一层生成策略图
    for i in range(1, 5): # 遍历 layer 1 到 4
        strategy_path = os.path.join(output_dir, f"{i}_layer_{i}_strategy.png")
        scheduler.visualizer.visualize_placement_strategy_for_layer(layer_index=i, save_path=strategy_path)

    print("\n测试完成！")
