import os
import sys
import time
import numpy as np
from scipy.spatial.transform import Rotation

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from task_scheduler import TaskScheduler
# 改为 Piper SDK 的导入
from piper_sdk import C_PiperInterface_V2

class CommandExecutor:
    def __init__(self, robot: C_PiperInterface_V2):
        self.robot = robot
        
        # 设置 Piper 为位置控制模式，速度设为 50%（中等速度）
        # 注意：Piper 的 MotionCtrl_2 参数：
        # ctrl_mode=0x01 (CAN 指令控制模式)
        # move_mode=0x00 (MOVE P, 位置模式)
        # move_spd_rate_ctrl=50 (速度百分比，0-100)
        # is_mit_mode=0x00 (位置速度模式)
        self.robot.MotionCtrl_2(ctrl_mode=0x01, move_mode=0x00, move_spd_rate_ctrl=50, is_mit_mode=0x00)
        
        self.last_projected_grasp_error = 0.0
        self.last_place_error_x = 0.0

        # 【新增】为特定层定义的硬编码推正点位
        self.HARDCODED_COMPACTION_POINTS = {
            "layer_1": {
                "approach_pos": [0.15, -0.317, 0.10], # 推正前的准备点 (X, Y, Z) 单位：米
                "push_end_pos": [0.10, -0.317, 0.0245]  # 推正动作的结束点 (X, Y, Z) 单位：米
            },
        }
        print(f"[Executor] 已初始化硬编码推正点位: {list(self.HARDCODED_COMPACTION_POINTS.keys())}")

    def _quat_to_euler_degrees(self, quat):
        """
        将四元数转换为欧拉角（度），用于 Piper 的 EndPoseCtrl。
        Piper 使用 XYZ 欧拉角，单位为 0.001 度，所以需要乘以 1000。
        输入 quat: [x, y, z, w] 四元数
        输出: [RX, RY, RZ] 单位 0.001 度
        """
        rot = Rotation.from_quat(quat)
        euler_deg = rot.as_euler('xyz', degrees=True)
        # Piper 需要 0.001 度单位，所以乘以 1000
        return [int(round(euler_deg[0] * 1000)), int(round(euler_deg[1] * 1000)), int(round(euler_deg[2] * 1000))]

    def _convert_pos_to_piper_units(self, pos):
        """
        将位置从米转换为 Piper 单位（0.001 mm）。
        输入 pos: [x, y, z] 单位：米
        输出: [X, Y, Z] 单位：0.001 mm
        """
        # 米 -> mm -> 0.001 mm: * 1000 * 1000 = * 1000000
        return [int(round(pos[0] * 1000000)), int(round(pos[1] * 1000000)), int(round(pos[2] * 1000000))]

    def _get_current_pose(self):
        """
        获取机械臂当前位姿
        
        Returns:
            tuple: (position, euler_angles)
                - position: [x, y, z] 单位：米
                - euler_angles: [rx, ry, rz] 单位：度
        """
        end_pose_msg = self.robot.GetArmEndPoseMsgs()
        
        current_pos = [
            end_pose_msg.end_pose.X_axis / 1000000.0,  # 转换为米
            end_pose_msg.end_pose.Y_axis / 1000000.0,
            end_pose_msg.end_pose.Z_axis / 1000000.0
        ]
        
        current_euler = [
            end_pose_msg.end_pose.RX_axis / 1000.0,    # 转换为度
            end_pose_msg.end_pose.RY_axis / 1000.0,
            end_pose_msg.end_pose.RZ_axis / 1000.0
        ]
        
        return current_pos, current_euler

    def _wait_for_position_reached(self, target_pos, target_quat, 
                                 position_tolerance=0.005, angle_tolerance=5.0, 
                                 max_wait_time=15.0, check_interval=0.5):
        """
        等待机械臂运动到指定位置，带误差容差检测
        
        Args:
            target_pos: 目标位置 [x, y, z] 单位：米
            target_quat: 目标四元数 [x, y, z, w]
            position_tolerance: 位置误差容差，单位：米（默认5mm）
            angle_tolerance: 角度误差容差，单位：度（默认5度）
            max_wait_time: 最大等待时间，单位：秒
            check_interval: 检测间隔，单位：秒
            
        Returns:
            tuple: (is_reached, final_error)
                - is_reached: bool，是否到达目标位置
                - final_error: dict，最终误差信息
        """
        print(f"  [位置检测] 等待到达目标位置...")
        print(f"    目标位置: [{target_pos[0]:.6f}, {target_pos[1]:.6f}, {target_pos[2]:.6f}]")
        
        # 将目标四元数转换为欧拉角用于比较
        target_euler = Rotation.from_quat(target_quat).as_euler('xyz', degrees=True)
        print(f"    目标姿态: [{target_euler[0]:.1f}°, {target_euler[1]:.1f}°, {target_euler[2]:.1f}°]")
        print(f"    容差设置: 位置±{position_tolerance*1000:.1f}mm, 角度±{angle_tolerance:.1f}°")
        
        start_time = time.time()
        check_count = 0
        
        while True:
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            # 超时检查
            if elapsed_time > max_wait_time:
                print(f"  ⚠ [位置检测] 超时！最大等待时间 {max_wait_time:.1f}s 已达到")
                break
            
            # 获取当前位姿
            current_pos, current_euler = self._get_current_pose()
            check_count += 1
            
            # 计算位置误差
            pos_error = np.array(current_pos) - np.array(target_pos)
            pos_error_magnitude = np.linalg.norm(pos_error)
            
            # 计算角度误差
            angle_errors = np.array(current_euler) - np.array(target_euler)
            # 处理角度跨越±180°的情况
            for i in range(3):
                while angle_errors[i] > 180:
                    angle_errors[i] -= 360
                while angle_errors[i] < -180:
                    angle_errors[i] += 360
            max_angle_error = np.max(np.abs(angle_errors))
            
            print(f"  [检测 {check_count:2d}] 当前位置: [{current_pos[0]:.6f}, {current_pos[1]:.6f}, {current_pos[2]:.6f}]")
            print(f"           当前姿态: [{current_euler[0]:.1f}°, {current_euler[1]:.1f}°, {current_euler[2]:.1f}°]")
            print(f"           位置误差: {pos_error_magnitude*1000:.1f}mm, 最大角度误差: {max_angle_error:.1f}°", end="")
            
            # 检查是否在容差范围内
            position_ok = pos_error_magnitude <= position_tolerance
            angle_ok = max_angle_error <= angle_tolerance
            
            if position_ok and angle_ok:
                print(" ✓")
                print(f"  ✓ [位置检测] 成功到达目标位置！用时 {elapsed_time:.2f}s")
                return True, {
                    'position_error_mm': pos_error_magnitude * 1000,
                    'max_angle_error_deg': max_angle_error,
                    'time_taken_s': elapsed_time,
                    'checks_performed': check_count
                }
            else:
                status = []
                if not position_ok:
                    status.append(f"位置差{pos_error_magnitude*1000:.1f}mm>{position_tolerance*1000:.1f}mm")
                if not angle_ok:
                    status.append(f"角度差{max_angle_error:.1f}°>{angle_tolerance:.1f}°")
                print(f" ✗ ({', '.join(status)})")
            
            # 等待下次检测
            time.sleep(check_interval)
        
        # 超时或失败
        current_pos, current_euler = self._get_current_pose()
        pos_error = np.array(current_pos) - np.array(target_pos)
        pos_error_magnitude = np.linalg.norm(pos_error)
        angle_errors = np.array(current_euler) - np.array(target_euler)
        max_angle_error = np.max(np.abs(angle_errors))
        
        print(f"  ✗ [位置检测] 未能到达目标位置")
        print(f"    最终位置误差: {pos_error_magnitude*1000:.1f}mm")
        print(f"    最终角度误差: {max_angle_error:.1f}°")
        
        return False, {
            'position_error_mm': pos_error_magnitude * 1000,
            'max_angle_error_deg': max_angle_error,
            'time_taken_s': elapsed_time,
            'checks_performed': check_count,
            'timeout': True
        }

    def _move_to_cart_pose_piper(self, pos, quat, wait_for_completion=False, 
                               position_tolerance=0.005, angle_tolerance=5.0,speed=40):
        """
        Piper 的移动到笛卡尔位姿方法（增强版）
        
        Args:
            pos: [x, y, z] 单位：米
            quat: [x, y, z, w] 四元数
            wait_for_completion: 是否等待运动完成
            position_tolerance: 位置误差容差，单位：米
            angle_tolerance: 角度误差容差，单位：度
        """
        # 确保在位置控制模式
        self.robot.MotionCtrl_2(ctrl_mode=0x01, move_mode=0x00, move_spd_rate_ctrl=speed, is_mit_mode=0x00)
        
        piper_pos = self._convert_pos_to_piper_units(pos)
        piper_euler = self._quat_to_euler_degrees(quat)
        
        print(f"Moving to Cartesian Pose (Piper units): Pos {piper_pos}, Euler {piper_euler}")
        self.robot.EndPoseCtrl(piper_pos[0], piper_pos[1], piper_pos[2], piper_euler[0], piper_euler[1], piper_euler[2])
        
        if wait_for_completion:
            # 使用位置检测功能
            is_reached, error_info = self._wait_for_position_reached(
                pos, quat, position_tolerance, angle_tolerance
            )
            
            if not is_reached:
                print(f"  ⚠ 警告：运动未完全到达目标位置，但继续执行...")
            time.sleep(0.5)
            return is_reached, error_info
        else:
            # 传统方式：固定等待时间
            time.sleep(0.8)     # 非等待模式移动后等待（s）；预抓取/抓取/抬升等大距离移动共用
            return True, {}

    def _get_orientation_quat(self, task_package: dict, pose_type: str) -> list:
        base_down = Rotation.from_euler('x', 179, degrees=True).as_matrix()
        angle_rad = task_package.get(f"{pose_type}_angle_rad", 0.0) or 0.0
        rot_z = Rotation.from_euler('z', angle_rad, degrees=False).as_matrix()
        final_rot_mat = rot_z @ base_down
        return Rotation.from_matrix(final_rot_mat).as_quat().tolist()

    def _get_current_gripper_distance(self):
        """
        获取当前夹爪距离
        
        Returns:
            float: 当前夹爪距离，单位：mm
        """
        try:
            gripper_msg = self.robot.GetArmGripperMsgs()
            # grippers_angle 的单位是 0.001mm（微米），需要除以1000转换为mm
            current_distance_mm = gripper_msg.grippers_angle / 1000.0
            return current_distance_mm
        except Exception as e:
            return 5.0

    def _gripper_open(self,flag=0):
        """
        Piper 夹爪安全张开：先多开1mm，等0.2s，再完全打开
        """
      
        if flag==1:
            # 获取当前距离
            current_distance_mm = self._get_current_gripper_distance()
            # 步骤1：多开1mm
            step1_distance_mm = 30 + 1.0
            step1_angle = int(round(step1_distance_mm * 1000))  # 转换为微米单位
            self.robot.GripperCtrl(gripper_angle=step1_angle, gripper_effort=500, gripper_code=0x01, set_zero=0x00)
            time.sleep(0.2)
        # 步骤2：完全打开
        final_distance_mm = 80.0
        final_angle = int(round(final_distance_mm * 1000))  # 转换为微米单位
        self.robot.GripperCtrl(gripper_angle=final_angle, gripper_effort=500, gripper_code=0x01, set_zero=0x00)
        time.sleep(0.5)
        

    def _gripper_close(self):
        """
        Piper 夹爪闭合。
        gripper_angle=0 为闭合。
        """
        step1_distance_mm = 30 + 1.0
        step1_angle = int(round(step1_distance_mm * 1000))  # 转换为微米单位
        self.robot.GripperCtrl(gripper_angle=step1_angle, gripper_effort=3000, gripper_code=0x01, set_zero=0x00)
        time.sleep(0.1)
        self.robot.GripperCtrl(gripper_angle=0, gripper_effort=3000, gripper_code=0x01, set_zero=0x00)
        time.sleep(0.5)

    def run_mission(self, scheduler: TaskScheduler, build_order: list):
        print("="*80 + "\n任务流启动 (Piper 机械臂模式)\n" + "="*80)
        global_start_time = time.time()

        for task_idx, block_id in enumerate(build_order):
            print(f"\n{'='*30} 请求处理积木 '{block_id}' ({task_idx + 1}/{len(build_order)}) {'='*30}")
            
            current_task = scheduler.get_task_for_block(block_id, build_order, self.last_projected_grasp_error)
            if current_task is None:
                continue
            
            print(f"\n--- 任务详情 ---\n{current_task}")
            self.execute_task(current_task)
            scheduler.update_placement_error(block_id, self.last_place_error_x)

            elapsed_time = time.time() - global_start_time
            print(f"\n\033[92m[计时] 积木 {current_task['id']} 完成。累计总用时: {elapsed_time:.2f} 秒。\033[0m")

    def execute_task(self, task: dict):
        try:
            block_id = task['id']
            pick_orientation_quat = self._get_orientation_quat(task, 'pick')
            place_orientation_quat = self._get_orientation_quat(task, 'place')

            # --- 抓取阶段 ---
            print(f"\n--- Executing: {block_id}: Grasping ---")
            self._move_to_cart_pose_piper(task['pre_grasp_pos'], pick_orientation_quat)
            self._gripper_open(); time.sleep(1.0)      # 张开夹爪延迟（s）
            self._move_to_cart_pose_piper(task['pick_pos'], pick_orientation_quat)
            self._gripper_close(); time.sleep(1.0)       # 闭合夹爪后等待（s）
            self._move_to_cart_pose_piper(task['pre_grasp_pos'], pick_orientation_quat)

            # --- 放置阶段 ---
            print(f"\n--- Executing: {block_id}: Placing ---")
            place_pos = np.array(task['place_pos'])
            pre_place_pos = np.array(task['pre_place_pos'])
            slide_direction = task.get('slide_direction', 0.0)
            
            if slide_direction != 0.0:
                print(f"  -> 执行推入式放置 (朝向: {'左' if slide_direction > 0 else '右'})")
                slide_offset_x = +slide_direction * task['block_size'][0] / 2.0
                slide_start_pos = place_pos + np.array([slide_offset_x, 0, 0.02])
                speed=30
                self._move_to_cart_pose_piper(slide_start_pos, place_orientation_quat, speed=speed)
                # ============ 关键修改：最终放置位置需要精确到达 ============
                is_reached, error_info = self._move_to_cart_pose_piper(
                    place_pos, place_orientation_quat, 
                    wait_for_completion=True,      # 等待运动完成
                    position_tolerance=0.0005,      # 3mm位置容差
                    angle_tolerance=3.0,           # 3度角度容差
                    speed=10
                )
           
            else:
                print("  -> 执行垂直放置")
                self._move_to_cart_pose_piper(pre_place_pos, place_orientation_quat)
                # ============ 关键修改：最终放置位置需要精确到达 ============
                is_reached, error_info = self._move_to_cart_pose_piper(
                    place_pos, place_orientation_quat,
                    wait_for_completion=True,      # 等待运动完成
                    position_tolerance=0.003,      # 3mm位置容差  
                    angle_tolerance=3.0           # 3度角度容差
                )
            
            # 记录到达精度
            if is_reached:
                print(f"  ✓ [精确放置] 成功到达，位置误差:{error_info['position_error_mm']:.1f}mm, "
                      f"角度误差:{error_info['max_angle_error_deg']:.1f}°")
            else:
                print(f"  ⚠ [精确放置] 未完全到达，位置误差:{error_info['position_error_mm']:.1f}mm, "
                      f"角度误差:{error_info['max_angle_error_deg']:.1f}°")
    

            # --- 释放与撤离 ---
            print(f"  -> 9. 释放积木")
            self._gripper_open(1); time.sleep(0.5)     # 张开夹爪释放
            self._move_to_cart_pose_piper(task['final_lift_pos'], place_orientation_quat, speed=60)
            time.sleep(0.1)      # 撤离后短暂等待（s）

        except Exception as e:
            print(f"\033[91m[任务执行失败]: {e}\033[0m"); import traceback; traceback.print_exc(); raise

    def _euler_degrees_to_quat(self, euler_deg):
        """
        将欧拉角（度）转换为四元数，用于兼容性。
        输入 euler_deg: [RX, RY, RZ] 度
        输出: [x, y, z, w] 四元数
        """
        rot = Rotation.from_euler('xyz', euler_deg, degrees=True)
        return rot.as_quat().tolist()
