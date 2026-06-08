# 预留的一个程序import os
import sys
import time
import threading
import numpy as np
import rclpy
import matplotlib.pyplot as plt
import os

# 动态添加父目录到Python路径，以导入 CommandExecutor
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from command_executor import CommandExecutor

class PoseLimitTester(CommandExecutor):
    """
    一个用于扫描和确定机器人在特定Y平面上可达工作空间的工具。
    """
    def __init__(self):
        super().__init__()
        print("[PoseLimitTester] 初始化...")

        # --- 扫描参数定义 ---
        # 1. 固定的Y轴坐标 (与搭建任务保持一致)
        self.fixed_y = 0.380
        
        # 2. Z轴扫描范围和步长
        self.z_start = 0.02   # Z轴起始扫描高度 (米)
        self.z_end = 0.25     # Z轴最大扫描高度 (米)
        self.z_step = 0.02    # 每次Z轴上升的高度 (米)

        # 3. X轴扫描范围和步长
        self.x_scan_min = 0.05  # X轴扫描的左边界
        self.x_scan_max = 0.35  # X轴扫描的右边界
        self.x_step = 0.01    # 每次X轴移动的距离 (米)
        self.x_preset = 0.20  # 每次水平扫描后返回的预设X位置 (接近中点)

        # 4. 边界判断条件
        self.timeout_threshold = 2 # 连续超时2次则判定为边界

        # 5. 结果存储
        self.reachable_points = []
        self.output_file = os.path.join(os.path.dirname(__file__), "reachable_space_mask.npy")

    def _attempt_move(self, target_pos: np.ndarray) -> bool:
        """
        尝试移动到目标点，并根据超时情况判断是否成功。
        连续超时 N 次则认为失败。
        """
        start_q = self.sensor_arm_qpos.copy()
        target_pose_mat = np.eye(4)
        target_pose_mat[:3, 3] = target_pos
        target_pose_mat[:3, :3] = self.grasp_rotation(0.0) # 使用默认0度姿态

        consecutive_timeouts = 0
        for _ in range(self.timeout_threshold):
            path = self.plan_path_joint_space(target_pose_mat, start_q, flag=0)
            if path:
                if self.execute_blocking_path(path, flag=0):
                    # 路径规划和执行都成功
                    print(f"  -> \033[92m成功到达: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]\033[0m")
                    return True
            
            # 规划失败或执行超时，计为一次超时
            consecutive_timeouts += 1
            print(f"  -> \033[93m移动尝试失败 ({consecutive_timeouts}/{self.timeout_threshold})...\033[0m")
        
        # 达到超时阈值，判定为不可达
        print(f"  -> \033[91m判定为边界！连续 {self.timeout_threshold} 次失败。\033[0m")
        return False

    def run_scan(self):
        """
        【重构版】执行完整的扫描任务。
        严格遵循“左扫->回中->右扫->回中->上升”的流程，并以探索到Z轴上限为结束条件。
        """
        print("="*80)
        print(f"开始扫描可达空间，固定 Y = {self.fixed_y}")
        print(f"流程: 左扫 -> 回中 -> 右扫 -> 回中 -> 上升")
        print(f"输出文件: {self.output_file}")
        print("="*80)

        current_z = self.z_start
        while rclpy.ok():
            print(f"\n{'='*20} 尝试扫描高度 Z = {current_z:.3f} {'='*20}")

            # 核心步骤 0: 尝试移动到本层的中心点。如果失败，则意味着已达Z轴上限，扫描结束。
            center_pos = np.array([self.x_preset, self.fixed_y, current_z])
            print(f"\n[Phase 0] 检查本层中心点可达性...")
            if not self._attempt_move(center_pos):
                print(f"\n\033[91m[扫描终止] 无法到达中心点 Z={current_z:.3f}。已确定Z轴可达上限。\033[0m")
                break # 结束 while 循环
            
            # 如果中心点可达，记录它并开始本层的扫描
            self.reachable_points.append(center_pos)

            # 核心步骤 1: 向左扫描
            print(f"\n[Phase 1] 向左扫描...")
            for x in np.arange(self.x_preset - self.x_step, self.x_scan_min - 1e-5, -self.x_step):
                if not rclpy.ok(): break
                target_pos = np.array([x, self.fixed_y, current_z])
                if self._attempt_move(target_pos):
                    self.reachable_points.append(target_pos)
                else:
                    print("  -> 已达左边界。")
                    break 

            # 核心步骤 2: 返回中心点
            print(f"\n[Phase 2] 返回中心点...")
            self._attempt_move(center_pos)

            # 核心步骤 3: 向右扫描
            print(f"\n[Phase 3] 向右扫描...")
            for x in np.arange(self.x_preset + self.x_step, self.x_scan_max + 1e-5, self.x_step):
                if not rclpy.ok(): break
                target_pos = np.array([x, self.fixed_y, current_z])
                if self._attempt_move(target_pos):
                    self.reachable_points.append(target_pos)
                else:
                    print("  -> 已达右边界。")
                    break
            
            # 核心步骤 4: 再次返回中心点，为下一轮上升做准备 (关键步骤)
            print(f"\n[Phase 4] 完成本层扫描，返回中心点准备上升...")
            self._attempt_move(center_pos)

            # 核心步骤 5: 增加Z高度，准备下一轮循环
            current_z += self.z_step

        # 扫描结束后保存结果
        self.save_results()

    def save_results(self):
        """将扫描到的可达点位保存到文件，并生成可视化预览。"""
        if not self.reachable_points:
            print("没有扫描到任何可达点，不保存文件。")
            return

        points_array = np.array(self.reachable_points)
        np.save(self.output_file, points_array)
        print(f"\n\033[92m扫描完成！共记录 {len(points_array)} 个可达点。")
        print(f"结果已保存到: {self.output_file}\033[0m")

        # 生成一个简单的2D散点图作为预览
        plt.figure(figsize=(10, 8))
        plt.scatter(points_array[:, 0], points_array[:, 2], s=10, c='green', marker='.')
        plt.title(f'Reachable Workspace Scan (Y = {self.fixed_y})')
        plt.xlabel('X-axis (m)')
        plt.ylabel('Z-axis (m)')
        plt.grid(True)
        plt.axis('equal')
        preview_path = self.output_file.replace('.npy', '.png')
        plt.savefig(preview_path)
        print(f"可视化预览图已保存到: {preview_path}")
        plt.close()


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    rclpy.init()
    
    tester = PoseLimitTester()

    # 启动ROS2的spin线程，以便节点能持续接收传感器数据
    spin_thread = threading.Thread(target=lambda: rclpy.spin(tester))
    spin_thread.start()

    print("正在等待机器人节点连接...")
    while rclpy.ok():
        if tester.recv_joint_states_:
            print("机器人节点已就绪！")
            break
        time.sleep(0.5)
    
    try:
        # 移动到一个安全的初始姿态
        initial_q = np.array([0.0, 0.0, -0.005, 0.0, 0.0, 0.0])
        tester.hold_position(initial_q, duration=2.0)
        
        # 开始执行扫描任务
        tester.run_scan()

    except KeyboardInterrupt:
        print("\n用户通过 Ctrl+C 退出程序。")
        tester.save_results() # 尝试在退出前保存已扫描的点
    finally:
        rclpy.shutdown()
        spin_thread.join()
        print("所有线程已结束，程序退出。")
