import numpy as np
import time
import traceback
import cv2
import threading
from scipy.spatial.transform import Rotation

from command_executor import CommandExecutor
from task_scheduler import TaskScheduler
from piper_sdk import C_PiperInterface_V2
from test_yolo import DetectionSystem

# ================= 全局变量 =================
GLOBAL_PIXEL_TO_MM_RATIO = {
    'x': 0.67,  # X方向：初始值 0.067 mm/pixel
    'y': 0.67,  # Y方向：初始值 0.067 mm/pixel
    'direction_x': 1,  # 方向系数
    'direction_y': 1,
    'update_count': 0
}

# 全局观察点位置（作为后备观察点）
GLOBAL_OBSERVATION_CONFIG = {
    'pos': [0.105, 0.00, 0.32],
    'quat': [-160.0, 0.0, -90.0]
}

CAMERA_COORDINATE_CONFIG = {
    'base_rz_deg': -90.0,      # 基准RZ角度（度）
    'rotation_direction': 1,    # 旋转方向：1=逆时针，-1=顺时针
    'pixel_x_to_arm': 'Y',     # 像素X+对应机械臂哪个轴（RZ=-90°时）
    'pixel_y_to_arm': 'X',     # 像素Y+对应机械臂哪个轴（RZ=-90°时）
}


# ================= 相机可视化线程 =================
class CameraVisualizationThread(threading.Thread):
    """独立线程显示相机画面"""
    def __init__(self, detector):
        super().__init__(daemon=True)
        self.detector = detector
        self.running = True
        
    def run(self):
        cv2.namedWindow('Camera Live View', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Camera Live View', 1280, 720)
        
        while self.running:
            color_frame, _ = self.detector.get_frames()
            if color_frame:
                color_image = np.asanyarray(color_frame.get_data())
                cv2.imshow('Camera Live View', color_image)
                
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
    
    def stop(self):
        self.running = False

# ================= 相机管理类 =================
# ================= 修改相机管理类，增加稳定性检测 =================
class CameraManager:
    """相机管理类：保持相机持续开启，按需获取检测结果"""
    def __init__(self, enable_visualization=True):
        self.detector = DetectionSystem()
        self.enable_visualization = enable_visualization
        self.viz_thread = None
        
        if enable_visualization:
            self.viz_thread = CameraVisualizationThread(self.detector)
            self.viz_thread.start()
            print("✓ 相机可视化线程已启动（按 'q' 关闭窗口）")
        
        print("✓ 相机系统已初始化（持续运行模式）")
    
    def update_robot_pose(self, robot):
        """更新机器人位姿到检测系统"""
        end_pose_msg = robot.GetArmEndPoseMsgs()
        position = [
            end_pose_msg.end_pose.X_axis / 1000000.0,
            end_pose_msg.end_pose.Y_axis / 1000000.0,
            end_pose_msg.end_pose.Z_axis / 1000000.0
        ]
        quaternion = Rotation.from_euler('xyz', [
            end_pose_msg.end_pose.RX_axis / 1000.0,
            end_pose_msg.end_pose.RY_axis / 1000.0,
            end_pose_msg.end_pose.RZ_axis / 1000.0
        ], degrees=True).as_quat().tolist()
        
        self.detector.set_robot_pose(position, quaternion)
        return position
    
    def get_detection(self, stabilize_time=1.5):
        """获取检测结果（原版，单次检测）"""
        time.sleep(stabilize_time)
        return self.detector.run_single_detection()
    
    def get_stable_detection(self, num_samples=2, skip_frames=2, sample_interval=0.02, 
                           angle_tolerance_deg=8.0, position_tolerance_mm=5.0):
        """
        获取稳定的检测结果（多帧采样平均）
        
        Args:
            num_samples: 采样帧数
            skip_frames: 跳过前几帧（避免不稳定数据）
            sample_interval: 采样间隔（秒）
            angle_tolerance_deg: 角度容差（度）
            position_tolerance_mm: 位置容差（毫米）
        
        Returns:
            稳定的检测结果，如果不稳定则返回None
        """
        print(f"  [稳定性检测] 跳过{skip_frames}帧，采样{num_samples}次...")
        
        # 跳过初始不稳定帧
        for i in range(skip_frames):
            print(f"    跳过第{i+1}帧...", end="")
            detection = self.detector.run_single_detection()
            if detection:
                print(f" 发现{len(detection)}个物体")
            else:
                print(" 无检测结果")
            time.sleep(0.6)
        
        # 采集稳定样本
        samples = []
        
        for i in range(num_samples):
            print(f"    采样 {i+1}/{num_samples}...", end="")
            
            detection = self.detector.run_single_detection()
            
            if detection:
                samples.append(detection)
                print(f" ✓ 发现{len(detection)}个物体")
            else:
                print(" ✗ 无检测结果")
            
            if i < num_samples - 1:  # 最后一次不等待
                time.sleep(sample_interval)
        
        if len(samples) < 2:
            print(f"  ⚠ 有效采样不足（{len(samples)}/{num_samples}）")
            return samples[0] if samples else None
        
        # ============ 稳定性分析和平均 ============
        print(f"  [稳定性分析] 分析{len(samples)}个样本...")
        
        # 统计每个积木在各个样本中的出现情况
        block_data = {}
        
        for sample_idx, sample in enumerate(samples):
            for block_id, data in sample.items():
                if block_id not in block_data:
                    block_data[block_id] = {
                        'positions': [],
                        'angles': [],
                        'pixels': [],
                        'sample_indices': []
                    }
                
                # 存储数据
                block_data[block_id]['positions'].append(data[0][:3])  # [x, y, z]
                block_data[block_id]['angles'].append(np.degrees(data[1]))  # 角度（度）
                if len(data) >= 3 and data[2] is not None:
                    block_data[block_id]['pixels'].append(data[2])  # [px, py]
                else:
                    block_data[block_id]['pixels'].append(None)
                block_data[block_id]['sample_indices'].append(sample_idx)
        
        # 筛选稳定的积木并计算平均值
        stable_blocks = {}
        
        for block_id, data in block_data.items():
            count = len(data['positions'])
            
            # 要求至少在一半以上的样本中出现
            if count < max(2, num_samples // 2):
                print(f"    {block_id}: 出现次数不足 ({count}/{num_samples})")
                continue
            
            # 计算位置稳定性
            positions = np.array(data['positions'])
            pos_mean = np.mean(positions, axis=0)
            pos_std = np.std(positions, axis=0)
            pos_max_std = np.max(pos_std) * 1000  # 转换为毫米
            
            # 计算角度稳定性
            angles = np.array(data['angles'])
            angle_mean = np.mean(angles)
            angle_std = np.std(angles)
            
            # 计算像素稳定性
            valid_pixels = [p for p in data['pixels'] if p is not None]
            pixel_mean = None
            pixel_max_std = 0.0
            
            if valid_pixels:
                pixels = np.array(valid_pixels)
                pixel_mean = np.mean(pixels, axis=0)
                pixel_std = np.std(pixels, axis=0)
                pixel_max_std = np.max(pixel_std)
            
            print(f"    {block_id}: 出现{count}次, 位置±{pos_max_std:.1f}mm, 角度±{angle_std:.1f}°", end="")
            
            # 检查稳定性
            is_stable = (pos_max_std <= position_tolerance_mm and 
                        angle_std <= angle_tolerance_deg)
            
            if is_stable:
                print(" ✓")
                
                # 构建稳定的数据项
                stable_position = pos_mean.tolist()
                stable_angle = np.radians(angle_mean)
                stable_pixel = pixel_mean.tolist() if pixel_mean is not None else None
                
                stable_blocks[block_id] = [stable_position, stable_angle, stable_pixel]
                
            else:
                print(f" ✗ (位置±{pos_max_std:.1f}mm>{position_tolerance_mm}, 角度±{angle_std:.1f}°>{angle_tolerance_deg})")
        
        if stable_blocks:
            print(f"  ✓ 稳定检测完成，{len(stable_blocks)}个物体通过稳定性测试")
            print(f"    稳定积木: {list(stable_blocks.keys())}")
            return stable_blocks
        else:
            print(f"  ⚠ 无稳定物体，使用最后一次检测结果")
            return samples[-1] if samples else None
    
    def close(self):
        """关闭相机"""
        if self.viz_thread:
            self.viz_thread.stop()
            self.viz_thread.join(timeout=2)
        if self.detector:
            self.detector.close()
            print("✓ 相机系统已关闭")
# ================= 夹爪角度计算函数（完全修复版）=================
def calculate_gripper_angle(detected_angle_rad, current_rz_deg, avoid_forbidden_zone=True):
    """
    计算夹爪旋转角度（简化版）
    
    原理：

    Returns:
        gripper_angle_rad: 夹爪应该旋转到的角度（弧度）
        gripper_angle_deg: 夹爪应该旋转到的角度（度）
    """
    # 步骤1: 将检测角度转换为度数 (0° ~ 180°)
    detected_angle_deg = np.degrees(detected_angle_rad)
   
    
    print(f"     [角度计算] 积木角度: {detected_angle_deg:.1f}°")
    print(f"     [当前状态] 夹爪RZ: {current_rz_deg:.1f}°")
    
    
    # 步骤3: 计算目标夹爪角度
    # 夹爪目标角度 = 当前RZ - 等效值
    if detected_angle_deg > 90.0:
        target_rz_deg = -90.0 - detected_angle_deg+180
    else:
        target_rz_deg = -90.0 - detected_angle_deg
    
    print(f"********888计算: {current_rz_deg:.1f}° - ({detected_angle_deg:.1f}°) = {target_rz_deg:.1f}°")
    
    # 步骤4: 标准化到 [-180°, 180°]
    while target_rz_deg > 180:
        target_rz_deg -= 360.0
    while target_rz_deg < -180.0:
        target_rz_deg += 360.0
    
    print(f"       标准化后: {target_rz_deg:.1f}°")
    
    # 步骤5: 检查限位区并利用180°对称性
    if avoid_forbidden_zone and 60.0 <= target_rz_deg <= 80.0:
        # 尝试180°对称角度
        target_rz_deg = target_rz_deg - 180.0
      
    
    gripper_angle_rad = np.radians(target_rz_deg)
    
    print(f"     [最终结果] 夹爪角度: RZ={target_rz_deg:.1f}° ({gripper_angle_rad:.4f} rad)")
    
    return gripper_angle_rad, target_rz_deg


# ================= 坐标转换辅助函数 =================
def calculate_pixel_to_arm_transform(current_rz_deg):
    """
    根据末端RZ角度计算像素坐标到机械臂坐标的变换矩阵
    
    Args:
        current_rz_deg: 当前末端RZ角度（度）
    
    Returns:
        2x2旋转矩阵，用于将像素偏移转换为机械臂XY偏移
    """
    # 计算相对于基准角度的旋转差值
    base_rz = CAMERA_COORDINATE_CONFIG['base_rz_deg']
    rotation_dir = CAMERA_COORDINATE_CONFIG['rotation_direction']
    
    # 相对旋转角（弧度）
    relative_angle_rad = np.radians((current_rz_deg - base_rz) * rotation_dir)
    
    # 构建旋转矩阵
    # 在RZ=-90°基准下：像素X+ → 机械臂Y-, 像素Y+ → 机械臂X-
    # 基准变换矩阵（硬编码的映射关系）
    base_transform = np.array([
        [0, -1],   # 像素Y → 机械臂X（负方向）
        [-1, 0]    # 像素X → 机械臂Y（负方向）
    ])
    
    # 应用相对旋转
    rotation_matrix = np.array([
        [np.cos(relative_angle_rad), -np.sin(relative_angle_rad)],
        [np.sin(relative_angle_rad), np.cos(relative_angle_rad)]
    ])
    
    # 最终变换 = 旋转矩阵 × 基准变换
    final_transform = rotation_matrix @ base_transform
    
    return final_transform


# ================= 移动与观察函数 =================
def move_to_observation_point(robot, pos, quat, speed=50):
    """移动到指定观察点"""
    piper_pos = [int(round(p * 1000000)) for p in pos]
    piper_euler = [int(round(e * 1000)) for e in quat]
    
    robot.MotionCtrl_2(0x01, 0x00, speed, 0x00)
    robot.EndPoseCtrl(piper_pos[0], piper_pos[1], piper_pos[2], 
                      piper_euler[0], piper_euler[1], piper_euler[2])
    time.sleep(2.5)
    
    # 验证是否到达
    end_pose_msg = robot.GetArmEndPoseMsgs()
    current_pos = [
        end_pose_msg.end_pose.X_axis / 1000000.0,
        end_pose_msg.end_pose.Y_axis / 1000000.0,
        end_pose_msg.end_pose.Z_axis / 1000000.0
    ]
    
    print(f"  -> 移动后实际位置: [{current_pos[0]:.6f}, {current_pos[1]:.6f}, {current_pos[2]:.6f}]")
    
    if np.allclose(current_pos, pos, atol=0.01):
        print(f"  ✓ 已到达观察点")
        return True
    else:
        print(f"  ✗ 未能精确到达 (误差: {np.linalg.norm(np.array(current_pos) - np.array(pos))*1000:.2f}mm)")
        return False

def observe_from_global_view(robot, camera_manager):
    """从全局观察点进行检测（使用稳定性检查）"""
    print("\n  -> 【全局观察】移动到全局视角...")
    success = move_to_observation_point(
        robot, 
        GLOBAL_OBSERVATION_CONFIG['pos'], 
        GLOBAL_OBSERVATION_CONFIG['quat']
    )
    
    if not success:
        return None
    
    camera_manager.update_robot_pose(robot)
    time.sleep(0.5)  # 短暂稳定
    
    # 使用稳定检测：跳过2帧，采样4次
    return camera_manager.get_stable_detection(
        num_samples=2, 
        skip_frames=2, 
        sample_interval=0.02,
        angle_tolerance_deg=6.0,      # 全局观察角度容差稍严格
        position_tolerance_mm=3.0     # 全局观察位置容差稍严格
    )

def observe_from_local_view(robot, camera_manager, rough_pos, observation_z=0.30):
    """
    从局部位置上方进行精细检测（使用稳定性检查）
    """
    local_obs_pos = [rough_pos[0] - 0.05, rough_pos[1], observation_z]
    local_obs_quat = [-160, 0, -90]
    
    print(f"  -> 【局部观察】移动到 [{local_obs_pos[0]:.3f}, {local_obs_pos[1]:.3f}, {local_obs_pos[2]:.3f}]...")
    success = move_to_observation_point(robot, local_obs_pos, local_obs_quat)
    
    if not success:
        print("  ✗ 局部观察位置移动失败")
        return None, False
    
    camera_manager.update_robot_pose(robot)
    time.sleep(0.3)  # 短暂稳定
    
    # 使用稳定检测：跳过1帧，采样3次（局部观察要求稍低）
    detection = camera_manager.get_stable_detection(
        num_samples=3, 
        skip_frames=5, 
        sample_interval=0.02,
        angle_tolerance_deg=10.0,     # 局部观察角度容差稍宽松
        position_tolerance_mm=8.0     # 局部观察位置容差稍宽松
    )
    
    if not detection:
        print("  ✗ 局部未检测到稳定积木")
        return None, False
    
    return detection, True

def rotate_gripper_to_angle(robot, target_angle_rad):
    """旋转夹爪到指定角度（保持当前XYZ位置和RX、RY）"""
    end_pose_msg = robot.GetArmEndPoseMsgs()
    current_pos = [
        end_pose_msg.end_pose.X_axis,
        end_pose_msg.end_pose.Y_axis,
        end_pose_msg.end_pose.Z_axis
    ]
    
    # 保持当前的RX、RY不变，只修改RZ
    current_euler = [
        end_pose_msg.end_pose.RX_axis,
        end_pose_msg.end_pose.RY_axis,
        int(round(np.degrees(target_angle_rad) * 1000))
    ]
    
    print(f"  -> 旋转夹爪: RZ从 {end_pose_msg.end_pose.RZ_axis/1000:.1f}° 到 {np.degrees(target_angle_rad):.1f}°...")
    
    print(current_euler,current_pos)
    robot.MotionCtrl_2(0x01, 0x00, 50, 0x00)
    robot.EndPoseCtrl(current_pos[0], current_pos[1], current_pos[2],
                     current_euler[0], current_euler[1], current_euler[2])
    time.sleep(1.5)
    
    # 验证旋转结果
    end_pose_msg = robot.GetArmEndPoseMsgs()
    actual_rz = end_pose_msg.end_pose.RZ_axis / 1000.0
    print(f"  -> 实际RZ角度: {actual_rz:.1f}°")

# ================= PID控制器类 =================
class PixelPIDController:
    """像素对中的PID控制器"""
    def __init__(self, kp=0.08, ki=0.0, kd=0.01):
        """
        Args:
            kp: 比例系数（主要控制响应速度）
            ki: 积分系数（消除稳态误差，暂时不用）
            kd: 微分系数（抑制震荡）
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        self.prev_error_x = 0.0
        self.prev_error_y = 0.0
        self.integral_x = 0.0
        self.integral_y = 0.0
    
    def compute(self, error_x, error_y, dt=1.0):
        """
        计算PID输出
        
        Args:
            error_x: X方向像素误差（目标 - 当前）
            error_y: Y方向像素误差
            dt: 时间步长（这里固定为1）
        
        Returns:
            (move_x_mm, move_y_mm): 机械臂应移动的距离（毫米）
        """
        # 积分项
        self.integral_x += error_x * dt
        self.integral_y += error_y * dt
        
        # 微分项
        derivative_x = (error_x - self.prev_error_x) / dt
        derivative_y = (error_y - self.prev_error_y) / dt
        
        # PID输出（像素单位）
        output_px_x = (self.kp * error_x + 
                       self.ki * self.integral_x + 
                       self.kd * derivative_x)
        
        output_px_y = (self.kp * error_y + 
                       self.ki * self.integral_y + 
                       self.kd * derivative_y)
        
        # 保存当前误差
        self.prev_error_x = error_x
        self.prev_error_y = error_y
        
        # 转换为毫米（使用全局比例尺和方向）
        move_x_mm = (GLOBAL_PIXEL_TO_MM_RATIO['direction_y'] * 
                     output_px_y * GLOBAL_PIXEL_TO_MM_RATIO['y'])
        
        move_y_mm = (GLOBAL_PIXEL_TO_MM_RATIO['direction_x'] * 
                     output_px_x * GLOBAL_PIXEL_TO_MM_RATIO['x'])
        
        return move_x_mm, move_y_mm
    
    def reset(self):
        """重置PID状态"""
        self.prev_error_x = 0.0
        self.prev_error_y = 0.0
        self.integral_x = 0.0
        self.integral_y = 0.0

# ================= 精确对中函数（PID版本）=================
def refine_position_to_center_with_tracking(robot, camera_manager, current_pos, initial_pixel_offset, 
                                           target_yolo_prefix,
                                           img_center_x=320, img_center_y=240,
                                           max_iterations=6, tolerance_pixels=15):
    """
    RZ=-90°固定角度下的像素中心化（选择像素X最小的积木）
    """
    global GLOBAL_PIXEL_TO_MM_RATIO
    
    print(f"\n  -> 【精确对中-RZ固定】开始迭代...")
    print(f"     目标类型: {target_yolo_prefix}")
    print(f"     图像中心: ({img_center_x}, {img_center_y})")
    
    # 获取当前夹爪姿态
    end_pose_msg = robot.GetArmEndPoseMsgs()
    current_gripper_euler = [
        end_pose_msg.end_pose.RX_axis,
        end_pose_msg.end_pose.RY_axis,
        end_pose_msg.end_pose.RZ_axis
    ]
    
    current_rz_deg = current_gripper_euler[2] / 1000.0
    print(f"     当前夹爪RZ: {current_rz_deg:.1f}°")
    
    # RZ=-90°时的固定映射关系
    direction_x = +1  # 固定
    direction_y = +1  # 固定
    ratio_mm_per_px = 0.67  # ← 修正比例尺！应该是0.067而不是0.67
    
    print(f"     固定参数: 比例尺={ratio_mm_per_px:.3f} mm/px, dir_x={direction_x:+d}, dir_y={direction_y:+d}")
    
    refined_pos = current_pos.copy()
    px, py = initial_pixel_offset
    
    for iteration in range(1, max_iterations + 1):
        offset_distance = np.sqrt((px - img_center_x)**2 + (py - img_center_y)**2)
        
        print(f"\n     [迭代 {iteration}/{max_iterations}]")
        print(f"       当前像素: ({px:.1f}, {py:.1f}), 距中心: {offset_distance:.1f}px")
        print(f"       当前位置: [{refined_pos[0]:.6f}, {refined_pos[1]:.6f}, {refined_pos[2]:.6f}]")
        
        if offset_distance <= tolerance_pixels:
            print(f"       ✓ 已达精度要求 ({offset_distance:.1f} ≤ {tolerance_pixels})")
            break
        
        # 计算像素误差
        error_px = img_center_x - px  # 需要向左移动多少像素
        error_py = img_center_y - py  # 需要向下移动多少像素
        
        # 直接计算机械臂移动量（mm）
        # RZ=-90°时：像素X → 机械臂Y（反向），像素Y → 机械臂X（正向）
        move_x_mm = direction_y * error_py * ratio_mm_per_px  # 像素Y误差 → 机械臂X移动
        move_y_mm = direction_x * error_px * ratio_mm_per_px  # 像素X误差 → 机械臂Y移动
        
        print(f"       像素误差: ΔPx={error_px:.1f}, ΔPy={error_py:.1f}")
        print(f"       移动指令: ΔX={move_x_mm:.2f}mm, ΔY={move_y_mm:.2f}mm")
        
        # 限制移动量
        max_move = 50.0  # ← 减小限制，避免过大移动
        if abs(move_x_mm) > max_move:
            move_x_mm = max_move if move_x_mm > 0 else -max_move
        if abs(move_y_mm) > max_move:
            move_y_mm = max_move if move_y_mm > 0 else -max_move
        
        # 保存移动前状态
        prev_px, prev_py = px, py
        prev_pos = refined_pos.copy()
        
        # 执行移动
        refined_pos[0] += move_x_mm / 1000.0
        refined_pos[1] += move_y_mm / 1000.0
        
        piper_pos = [int(round(p * 1000000)) for p in refined_pos]
        
        robot.MotionCtrl_2(0x01, 0x00, 30, 0x00)
        robot.EndPoseCtrl(piper_pos[0], piper_pos[1], piper_pos[2], 
                         current_gripper_euler[0], current_gripper_euler[1], current_gripper_euler[2])
        time.sleep(1.8)
        
        # 获取实际位置
        end_pose_msg = robot.GetArmEndPoseMsgs()
        actual_pos = [
            end_pose_msg.end_pose.X_axis / 1000000.0,
            end_pose_msg.end_pose.Y_axis / 1000000.0,
            end_pose_msg.end_pose.Z_axis / 1000000.0
        ]
        
        actual_move_x = actual_pos[0] - prev_pos[0]
        actual_move_y = actual_pos[1] - prev_pos[1]
        
        print(f"       实际位置: [{actual_pos[0]:.6f}, {actual_pos[1]:.6f}, {actual_pos[2]:.6f}]")
        print(f"       实际移动: ΔX={actual_move_x*1000:.2f}mm, ΔY={actual_move_y*1000:.2f}mm")
        
        refined_pos = np.array(actual_pos)
        
        # 重新检测
        camera_manager.update_robot_pose(robot)
        
                # 像素中心化过程中使用轻量级稳定检测
        desktop_data = camera_manager.get_stable_detection(
            num_samples=2,           # 只采样2次，速度优先
            skip_frames=2,           # 跳过2帧
            sample_interval=0.02,     # 间隔0.2秒
            angle_tolerance_deg=15.0, # 宽松的角度容差
            position_tolerance_mm=10.0 # 宽松的位置容差
        )
        
        if not desktop_data:
            print(f"       ✗ 迭代{iteration}: 未检测到积木")# 回退到单次检测
            
            desktop_data = camera_manager.get_detection(stabilize_time=1.0)
            if not desktop_data:
                break
        
        # ============ 关键修改：选择像素X最小的同类型积木 ============
        target_candidates = []
        for block_id, data in desktop_data.items():
            if (block_id.startswith(target_yolo_prefix) and 
                len(data) >= 3 and data[2] is not None):
                temp_px, temp_py = data[2]
                target_candidates.append((block_id, temp_px, temp_py))
        
        if not target_candidates:
            print(f"       ✗ 迭代{iteration}: 未找到类型 {target_yolo_prefix} 的积木")
            break
        
        # 选择像素X最小的积木（最左边的，对应真实世界Y最大的）
        target_candidates.sort(key=lambda x: x[1])  # 按像素X排序
        selected_block_id, px, py = target_candidates[0]
        
        print(f"       [目标跟踪] 发现 {len(target_candidates)} 个候选积木:")
        for i, (bid, tpx, tpy) in enumerate(target_candidates):
            marker = "★" if i == 0 else " "
            print(f"         {marker} {bid}: 像素({tpx:.1f}, {tpy:.1f})")
        
        new_offset = np.sqrt((px - img_center_x)**2 + (py - img_center_y)**2)
        
        # 计算像素变化
        actual_pixel_change_x = px - prev_px
        actual_pixel_change_y = py - prev_py
        
        print(f"       实际像素变化: ΔPx={actual_pixel_change_x:.1f}, ΔPy={actual_pixel_change_y:.1f}")
        print(f"       距离变化: {offset_distance:.1f}px → {new_offset:.1f}px")
        
        # 检查是否在朝正确方向收敛
        if iteration > 1 and new_offset > offset_distance * 1.2:
            print(f"       ⚠ 发散趋势，可能参数有误")
    
    final_offset = np.sqrt((px - img_center_x)**2 + (py - img_center_y)**2)
    if final_offset <= tolerance_pixels:
        print(f"\n  ✓ 精确对中成功！最终误差: {final_offset:.1f}px")
    else:
        print(f"\n  ⚠ 达最大迭代次数，最终误差: {final_offset:.1f}px")
    
    return refined_pos, (px, py)


# ================= 抓取偏移计算函数 =================
def calculate_grasp_offset(current_rz_deg, offset_distance_mm=20.0):
    """
    根据夹爪RZ角度计算抓取偏移量
    
    原理：
    - 夹爪的"前进方向"是沿着其本地坐标系的X轴
    - 需要将本地偏移投影到世界坐标系（机械臂基座坐标系）
    
    Args:
        current_rz_deg: 当前夹爪RZ角度（度）
        offset_distance_mm: 沿夹爪前进方向的偏移距离（毫米）
    
    Returns:
        (offset_x_m, offset_y_m): 机械臂基座坐标系下的XY偏移（米）
    """
    # 将角度转换为弧度
    rz_rad = np.radians(current_rz_deg)
    #todo 当前因为某些原因，不能用长轴的思路
    rz_rad=-np.pi/2
    # 夹爪的前进方向在世界坐标系中的投影
    # 当RZ=0°时，夹爪朝向+Y方向
    # 当RZ=-90°时，夹爪朝向+X方向
    # 当RZ=-180°时，夹爪朝向-Y方向
    
    # 旋转矩阵：从夹爪本地坐标系到世界坐标系
    # 夹爪本地X轴（前进方向）→ 世界坐标系
    offset_x_m = offset_distance_mm / 1000.0 * np.cos(rz_rad + np.pi/2)
    offset_y_m = offset_distance_mm / 1000.0 * np.sin(rz_rad + np.pi/2)
    
    print(f"  [抓取偏移] RZ={current_rz_deg:.1f}°, 偏移{offset_distance_mm}mm")
    print(f"             → 世界坐标: ΔX={offset_x_m*1000:.2f}mm, ΔY={offset_y_m*1000:.2f}mm")
    
    return offset_x_m, offset_y_m

def apply_grasp_offset(robot, offset_x_m, offset_y_m, speed=30):
    """
    执行抓取前的偏移调整
    
    Args:
        robot: 机械臂对象
        offset_x_m: X方向偏移（米）
        offset_y_m: Y方向偏移（米）
        speed: 移动速度
    """
    # 获取当前位置
    end_pose_msg = robot.GetArmEndPoseMsgs()
    current_pos = [
        end_pose_msg.end_pose.X_axis / 1000000.0,
        end_pose_msg.end_pose.Y_axis / 1000000.0,
        end_pose_msg.end_pose.Z_axis / 1000000.0
    ]
    
    current_euler = [
        end_pose_msg.end_pose.RX_axis,
        end_pose_msg.end_pose.RY_axis,
        end_pose_msg.end_pose.RZ_axis
    ]
    
    # 计算目标位置
    target_pos = [
        current_pos[0] + offset_x_m,
        current_pos[1] + offset_y_m,
        current_pos[2]  # Z保持不变
    ]
    
    print(f"  -> 应用抓取偏移...")
    print(f"     当前位置: [{current_pos[0]:.6f}, {current_pos[1]:.6f}, {current_pos[2]:.6f}]")
    print(f"     目标位置: [{target_pos[0]:.6f}, {target_pos[1]:.6f}, {target_pos[2]:.6f}]")
    
    # 执行移动
    piper_pos = [int(round(p * 1000000)) for p in target_pos]
    
    robot.MotionCtrl_2(0x01, 0x00, speed, 0x00)
    robot.EndPoseCtrl(piper_pos[0], piper_pos[1], piper_pos[2],
                     current_euler[0], current_euler[1], current_euler[2])
    time.sleep(1.5)
    
    # 验证
    end_pose_msg = robot.GetArmEndPoseMsgs()
    actual_pos = [
        end_pose_msg.end_pose.X_axis / 1000000.0,
        end_pose_msg.end_pose.Y_axis / 1000000.0,
        end_pose_msg.end_pose.Z_axis / 1000000.0
    ]
    
    print(f"     实际位置: [{actual_pos[0]:.6f}, {actual_pos[1]:.6f}, {actual_pos[2]:.6f}]")
    
    return np.array(actual_pos)

# ================= 修改候选积木选择函数 =================
def select_best_candidate(processed_yolo_data, yolo_prefix, robot, camera_manager,
                         detected_angle_rad, enable_refinement=True, 
                         enable_grasp_offset=True,
                         grasp_offset_mm=50.0,
                         img_center_x=320, img_center_y=240):
    """
    选择积木：全局观察选Y最大，局部观察选像素X最小
    """
    candidates = [k for k in processed_yolo_data.keys() if k.startswith(yolo_prefix)]
    
    if not candidates:
        return None, None, None
    
    best_candidate = None
    
    # ============ 修改选择策略：选择Y轴最大的积木 ============
    max_y = -float('inf')
    
    print(f"  -> 候选积木: {candidates}")
    for cand_id in candidates:
        data = processed_yolo_data[cand_id]
        pos_y = data[0][1]  # Y坐标
        
        print(f"    -> {cand_id}: 位置Y={pos_y:.3f}")
        
        if pos_y > max_y:
            max_y = pos_y
            best_candidate = cand_id
    
    if not best_candidate:
        return None, None, None
    
    selected_data = processed_yolo_data[best_candidate]
    print(f"  -> 选择积木: '{best_candidate}' (Y轴最大: {max_y:.3f})")
    
    # 步骤1: 获取当前夹爪RZ角度
    end_pose_msg = robot.GetArmEndPoseMsgs()
    current_rz_deg = end_pose_msg.end_pose.RZ_axis / 1000.0
    
    # 步骤2: 计算目标夹爪角度
    gripper_angle_rad, gripper_angle_deg = calculate_gripper_angle(
        detected_angle_rad, current_rz_deg
    )
    print(f"  -> 计算夹爪角度: {gripper_angle_deg:.1f}°")
    
    # 步骤3: 精确对中（像素中心化）- 使用新的跟踪策略
    if enable_refinement and len(selected_data) >= 3 and selected_data[2] is not None:
        end_pose_msg = robot.GetArmEndPoseMsgs()
        current_pos = np.array([
            end_pose_msg.end_pose.X_axis / 1000000.0,
            end_pose_msg.end_pose.Y_axis / 1000000.0,
            end_pose_msg.end_pose.Z_axis / 1000000.0
        ])
        
        refined_pos, final_pixel = refine_position_to_center_with_tracking(
            robot, camera_manager, current_pos, selected_data[2], yolo_prefix,
            img_center_x=img_center_x, img_center_y=img_center_y
        )
        
        print(f"  -> 精确对中后: [{refined_pos[0]:.6f}, {refined_pos[1]:.6f}]")
        
        # 步骤4: 应用抓取偏移
        if enable_grasp_offset:
            end_pose_msg = robot.GetArmEndPoseMsgs()
            final_rz_deg = end_pose_msg.end_pose.RZ_axis / 1000.0
            
            offset_x_m, offset_y_m = calculate_grasp_offset(final_rz_deg, grasp_offset_mm)
            final_pos = apply_grasp_offset(robot, offset_x_m, offset_y_m)
            
            selected_data[0][0] = final_pos[0]
            selected_data[0][1] = final_pos[1]
            
            print(f"  ✓ 抓取偏移完成: [{final_pos[0]:.6f}, {final_pos[1]:.6f}]")
        else:
            selected_data[0][0] = refined_pos[0]
            selected_data[0][1] = refined_pos[1]
            selected_data[2] = final_pixel
    
    return best_candidate, selected_data, gripper_angle_rad

# ================= 数据预处理 =================
def preprocess_yolo_angles(yolo_data, corrections, observation_rz_deg=-90.0):
    """
    预处理YOLO数据
    
    Args:
        yolo_data: YOLO检测结果
        corrections: 位置和角度修正
        observation_rz_deg: 观察点的RZ角度（用于角度补偿）
    """
    if not yolo_data:
        return {}

    import copy
    processed_data = copy.deepcopy(yolo_data)

    for block_id, data in processed_data.items():
        # 【关键修复】补偿观察姿态的RZ角度影响
        # test_yolo.py 在计算角度时会将机械臂姿态叠加进去
        # 需要减去观察点的RZ角度，恢复物体的真实图像角度
        angle_rad = data[1]
        angle_deg = np.degrees(angle_rad)
        
        # 减去观察姿态的影响
        compensated_angle_deg = angle_deg
        
        data[1] = np.radians(compensated_angle_deg)
        # 应用其他修正
        correction = corrections.get(block_id, [0.0, 0.0, 0.0, 0.0])
        if any(c != 0.0 for c in correction):
            data[0][0] += correction[0]
            data[0][1] += correction[1]
            data[0][2] += correction[2]
            data[1] += correction[3]

    return processed_data


# ================= 主函数 =================
def main():
    np.set_printoptions(precision=4, suppress=True, linewidth=120)
    
    YOLO_CORRECTIONS = {
        "code3_1": [0.0, 0.0, -0.02, 0.0],
        "code3_2": [0.0, 0.0, -0.02, 0.0],
        "code4": [0.0, 0.0, -0.05, 0.0],
    }
    
    TYPE_TO_YOLO_PREFIX = {
        "type1": "code1", "type2": "code2", "type3": "code3", "type4": "code4"
    }

    CAN_IFACE = "can0"
    robot = C_PiperInterface_V2(CAN_IFACE)
    robot.ConnectPort()
    while not robot.EnablePiper():
        print("等待机械臂使能...")
        time.sleep(0.5)

    camera_manager = None
    
    try:
        camera_manager = CameraManager(enable_visualization=True)
        
        executor = CommandExecutor(robot)
        scheduler = TaskScheduler(first_block_target_pos=[0.0200, -0.200, 0.11])
        
        build_order = (
            scheduler.architecture["layer_1"] + scheduler.architecture["layer_2"] +
            scheduler.architecture["layer_3"] + scheduler.architecture["layer_4"]
        )

        print("\n" + "="*60)
        print("--- 步骤1: 初始全局观察 ---")
        print("="*60)
        
        # 初始全局观察
        global_yolo_data = observe_from_global_view(robot, camera_manager)
        print(f"global_yolo_data: {global_yolo_data}")
        
        if not global_yolo_data:
            print("✗ 初始全局观察失败")
            return
        
        robot.GripperCtrl(gripper_angle=80000, gripper_effort=1000, gripper_code=0x01, set_zero=0x00)
        processed_global_data = preprocess_yolo_angles(
            global_yolo_data, 
            YOLO_CORRECTIONS,
            observation_rz_deg=GLOBAL_OBSERVATION_CONFIG['quat'][2]
        )
        
        # 初始化粗略位置表
        rough_positions = {}
        for block_id, data in processed_global_data.items():
            rough_positions[block_id] = data[0]
            print(f"  -> 初始粗略位置: {block_id} -> {data[0][:2]}")
        
        print("\n" + "="*60)
        print("--- 步骤2: 循环构建 ---")
        print("="*60)
        
        global_start_time = time.time()

        for task_idx, block_id in enumerate(build_order):
            print(f"\n>>> [{task_idx + 1}/{len(build_order)}] 处理积木: {block_id}")
            
            # ============ 关键修改：每次处理新积木前都重新全局观察 ============
            if task_idx > 0:  # 第一个积木已经有初始全局观察结果
                print(f"\n  === 重新全局观察（积木 {block_id}）===")
                fresh_global_data = observe_from_global_view(robot, camera_manager)
                
                if fresh_global_data:
                    fresh_processed_data = preprocess_yolo_angles(
                        fresh_global_data, 
                        YOLO_CORRECTIONS,
                        observation_rz_deg=GLOBAL_OBSERVATION_CONFIG['quat'][2]
                    )
                    
                    # 更新粗略位置表
                    rough_positions = {}
                    for bid, data in fresh_processed_data.items():
                        rough_positions[bid] = data[0]
                    
                    print(f"  -> 更新后粗略位置表（共{len(rough_positions)}个积木）：")
                    for bid, pos in rough_positions.items():
                        print(f"     {bid} -> [{pos[0]:.3f}, {pos[1]:.3f}]")
                else:
                    print(f"  ⚠ 重新全局观察失败，使用上一轮位置")
            
            # ============ 局部观察重试机制 ============
            max_retries = 3
            local_yolo_data = None
            processed_local_data = None
            is_local_observation = False
            
            for retry in range(max_retries):
                print(f"\n  === 局部观察尝试 {retry + 1}/{max_retries} ===")
                
                # ============ 修改：选择Y值最大的同类型积木作为局部观察目标 ============
                target_type = scheduler.instances[block_id]["type"]
                yolo_prefix = TYPE_TO_YOLO_PREFIX.get(target_type)
                
                # 从粗略位置表中找到所有同类型的积木
                same_type_positions = {}
                for bid, pos in rough_positions.items():
                    if bid.startswith(yolo_prefix):
                        same_type_positions[bid] = pos
                
                if not same_type_positions:
                    print(f"  ✗ 粗略位置表中未找到类型 {target_type} 的积木")
                    # 使用scheduler中的默认位置作为后备
                    rough_pos = scheduler.instances[block_id]["initial_pos"].copy()
                else:
                    # ============ 关键修改：选择Y值最大的积木位置（最远离机械臂的）============
                    max_y_bid = max(same_type_positions.items(), key=lambda x: x[1][1])
                    rough_pos = max_y_bid[1].copy()
                    print(f"  -> 选择同类型积木中Y值最大的: {max_y_bid[0]} (Y={max_y_bid[1][1]:.3f})")
                    print(f"     同类型候选: {list(same_type_positions.keys())}")
                    for bid, pos in same_type_positions.items():
                        marker = "★" if bid == max_y_bid[0] else " "
                        print(f"       {marker} {bid}: Y={pos[1]:.3f}")
                
                if isinstance(rough_pos, np.ndarray):
                    rough_pos = rough_pos.tolist()
                
                rough_pos[0] -= 0.05  # 局部观察点X偏移
                rough_pos[2] = 0.30   # 设置为观察高度
                
                # 尝试局部观察
                local_yolo_data, is_local = observe_from_local_view(robot, camera_manager, rough_pos)
                
                if local_yolo_data and is_local:
                    # 局部观察成功
                    processed_local_data = preprocess_yolo_angles(
                        local_yolo_data, 
                        YOLO_CORRECTIONS,
                    )
                    
                    # 检查是否有目标类型的积木
                    target_type = scheduler.instances[block_id]["type"]
                    yolo_prefix = TYPE_TO_YOLO_PREFIX.get(target_type)
                    
                    has_target = any(k.startswith(yolo_prefix) for k in processed_local_data.keys())
                    
                    if has_target:
                        print(f"  ✓ 局部观察成功，找到目标类型 {target_type}")
                        is_local_observation = True
                        break
                    else:
                        print(f"  ⚠ 局部观察到积木，但无目标类型 {target_type}")
                else:
                    print(f"  ✗ 局部观察失败")
                
                # 局部观察失败时的处理
                if retry < max_retries - 1:
                    print(f"  -> 局部观察失败，重新全局观察...")
                    fallback_global_data = observe_from_global_view(robot, camera_manager)
                    
                    if fallback_global_data:
                        fallback_processed_data = preprocess_yolo_angles(
                            fallback_global_data, 
                            YOLO_CORRECTIONS,
                            observation_rz_deg=GLOBAL_OBSERVATION_CONFIG['quat'][2]
                        )
                        
                        # 更新粗略位置表
                        rough_positions = {}
                        for bid, data in fallback_processed_data.items():
                            rough_positions[bid] = data[0]
                        
                        print(f"  -> 已更新粗略位置表（共{len(rough_positions)}个积木）")
                        
                        # 检查当前积木是否在新的全局观察中
                        target_type = scheduler.instances[block_id]["type"]
                        yolo_prefix = TYPE_TO_YOLO_PREFIX.get(target_type)
                        found_in_global = any(k.startswith(yolo_prefix) for k in fallback_processed_data.keys())
                        
                        if found_in_global:
                            print(f"  ✓ 全局观察中找到类型 {target_type}，准备下一次局部观察")
                        else:
                            print(f"  ⚠ 全局观察中未找到类型 {target_type} 的积木")
                    else:
                        print(f"  ✗ 后备全局观察也失败")
                else:
                    print(f"  ✗ 已达最大重试次数")
            
            # ============ 检查最终结果 ============
            if not processed_local_data or not is_local_observation:
                print(f"  ✗ 最终失败：无法获取积木 {block_id}，跳过")
                continue
            
            # ============ 选择最佳候选并执行任务 ============
            target_type = scheduler.instances[block_id]["type"]
            yolo_prefix = TYPE_TO_YOLO_PREFIX.get(target_type)
            
            # 获取检测角度
            detected_angle_rad = None
            for cand_id in processed_local_data.keys():
                if cand_id.startswith(yolo_prefix):
                    detected_angle_rad = processed_local_data[cand_id][1]
                    break
            
            if detected_angle_rad is None:
                print(f"  ✗ 未找到类型 {target_type} 的积木")
                continue
            
            selected_yolo_id, selected_data, gripper_angle_rad = select_best_candidate(
                processed_local_data, yolo_prefix, robot, camera_manager, 
                detected_angle_rad, 
                enable_refinement=True,
                enable_grasp_offset=True,
                grasp_offset_mm=100.0
            )
            
            if not selected_yolo_id:
                print(f"  ✗ 候选选择失败")
                continue
            
            print(f"  ✓ 绑定: '{block_id}' <- '{selected_yolo_id}'")
            print(f"  -> 最终夹爪角度: {np.degrees(gripper_angle_rad):.1f}°")
            
            # 更新Scheduler
            pos = selected_data[0]
            current_z = scheduler.instances[block_id]["initial_pos"][2]
            update_dict = {block_id: [[pos[0], pos[1], current_z], gripper_angle_rad]}
            scheduler.update_initial_states_from_dict(update_dict)

            # 执行任务
            current_task = scheduler.get_task_for_block(block_id, build_order, executor.last_projected_grasp_error)
            
            if current_task:
                print(f"\n  === 执行抓取放置任务 ===")
                executor.execute_task(current_task)
                scheduler.update_placement_error(block_id, executor.last_place_error_x)
                
                elapsed = time.time() - global_start_time
                print(f"  ✓ 完成 {block_id}，累计耗时: {elapsed:.2f}s")
                print(f"  -> 准备处理下一个积木...")
            else:
                print(f"  ✗ 无法生成任务")

        print("\n" + "="*60)
        print("--- 所有任务完成 ---")
        print("="*60)

    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"\n异常: {e}")
        traceback.print_exc()
    finally:
        if camera_manager:
            camera_manager.close()
        print("程序结束")

if __name__ == "__main__":
    main()
