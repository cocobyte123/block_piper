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

    def _move_to_cart_pose_piper(self, pos, quat):
        """
        Piper 的移动到笛卡尔位姿方法。
        输入 pos: [x, y, z] 单位：米
        输入 quat: [x, y, z, w] 四元数
        Piper EndPoseCtrl 需要：
        - X, Y, Z: 0.001 mm
        - RX, RY, RZ: 0.001 度
        """
        # 确保在位置控制模式
        self.robot.MotionCtrl_2(ctrl_mode=0x01, move_mode=0x00, move_spd_rate_ctrl=40, is_mit_mode=0x00)
        
        piper_pos = self._convert_pos_to_piper_units(pos)
        piper_euler = self._quat_to_euler_degrees(quat)
        print(f"Moving to Cartesian Pose (Piper units): Pos {piper_pos}, Euler {piper_euler}")
        self.robot.EndPoseCtrl(piper_pos[0], piper_pos[1], piper_pos[2], piper_euler[0], piper_euler[1], piper_euler[2])
        time.sleep(1.2)  # 短暂等待，确保命令发送

    def _get_orientation_quat(self, task_package: dict, pose_type: str) -> list:
        base_down = Rotation.from_euler('x', 179, degrees=True).as_matrix()
        angle_rad = task_package.get(f"{pose_type}_angle_rad", 0.0) or 0.0
        rot_z = Rotation.from_euler('z', angle_rad, degrees=False).as_matrix()
        final_rot_mat = rot_z @ base_down
        return Rotation.from_matrix(final_rot_mat).as_quat().tolist()

    def _gripper_open(self):
        """
        Piper 夹爪张开。
        GripperCtrl 参数：
        - gripper_angle: 0.001 mm，假设 10000 = 10 mm 张开
        - gripper_effort: 0.001 N·m，5000 = 5 N·m
        - gripper_code: 0x01 使能
        - set_zero: 0x00 不设置零点
        """
        self.robot.GripperCtrl(gripper_angle=80000, gripper_effort=1000, gripper_code=0x01, set_zero=0x00)

    def _gripper_close(self):
        """
        Piper 夹爪闭合。
        gripper_angle=0 为闭合。
        """
        self.robot.GripperCtrl(gripper_angle=0, gripper_effort=5000, gripper_code=0x01, set_zero=0x00)

    def run_mission(self, scheduler: TaskScheduler, build_order: list):
        print("="*80 + "\n任务流启动 (Piper 机械臂模式)\n" + "="*80)
        global_start_time = time.time()

        for task_idx, block_id in enumerate(build_order):
            print(f"\n{'='*30} 请求处理积木 '{block_id}' ({task_idx + 1}/{len(build_order)}) {'='*30}")
            
            current_task = scheduler.get_task_for_block(block_id, build_order, self.last_projected_grasp_error)
            if current_task is None:
                continue
            
            print(f"\n--- 任务详情 ---\n{current_task}")
            self._execute_task(current_task)
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
            self._gripper_open(); time.sleep(1.0)
            self._move_to_cart_pose_piper(task['pick_pos'], pick_orientation_quat)
            self._gripper_close(); time.sleep(1.0)
            self._move_to_cart_pose_piper(task['pre_grasp_pos'], pick_orientation_quat)

            # --- 放置阶段 ---
            print(f"\n--- Executing: {block_id}: Placing ---")
            place_pos = np.array(task['place_pos'])
            pre_place_pos = np.array(task['pre_place_pos'])
            slide_direction = task.get('slide_direction', 0.0)
            
            if slide_direction != 0.0:
                print(f"  -> 执行推入式放置 (朝向: {'左' if slide_direction > 0 else '右'})")
                slide_offset_x = +slide_direction * task['block_size'][0] / 4.0
                
                slide_start_pos = place_pos + np.array([slide_offset_x, 0, 0.005])
     
                self._move_to_cart_pose_piper(slide_start_pos, place_orientation_quat)
              
                self._move_to_cart_pose_piper(place_pos, place_orientation_quat)
           
            else:
                print("  -> 执行垂直放置")
                self._move_to_cart_pose_piper(pre_place_pos, place_orientation_quat)
              
                self._move_to_cart_pose_piper(place_pos, place_orientation_quat)
           
            
            # 获取实际放置位姿
            end_pose_msg = self.robot.GetArmEndPoseMsgs()
            actual_place_pose = [
                [end_pose_msg.end_pose.X_axis / 1000000.0, end_pose_msg.end_pose.Y_axis / 1000000.0, end_pose_msg.end_pose.Z_axis / 1000000.0],
                self._euler_degrees_to_quat([end_pose_msg.end_pose.RX_axis / 1000.0, end_pose_msg.end_pose.RY_axis / 1000.0, end_pose_msg.end_pose.RZ_axis / 1000.0])
            ]

            print("实际防取的位置与姿势",actual_place_pose)
            # 误差记录（假设有相关逻辑）

            # --- 释放与撤离 ---
            print(f"  -> 9. 释放积木")
            self._gripper_open(); time.sleep(1)
            self._move_to_cart_pose_piper(task['final_lift_pos'], place_orientation_quat)
            time.sleep(2)

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