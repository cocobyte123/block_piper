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
        """获取检测结果"""
        time.sleep(stabilize_time)
        return self.detector.run_single_detection()
    
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
    detected_angle_deg = detected_angle_deg % 180
    
    print(f"     [角度计算] 积木角度: {detected_angle_deg:.1f}°")
    print(f"     [当前状态] 夹爪RZ: {current_rz_deg:.1f}°")
    
    
    # 步骤3: 计算目标夹爪角度
    # 夹爪目标角度 = 当前RZ - 等效值
    target_rz_deg = (current_rz_deg - detected_angle_deg)%180
    
    print(f"********888计算: {current_rz_deg:.1f}° - ({detected_angle_deg:.1f}°) = {target_rz_deg:.1f}°")
    
    # 步骤4: 标准化到 [-180°, 180°]
    while target_rz_deg > 180.0:
        target_rz_deg -= 360.0
    while target_rz_deg < -180.0:
        target_rz_deg += 360.0
    
    print(f"       标准化后: {target_rz_deg:.1f}°")
    
    # 步骤5: 检查限位区并利用180°对称性
    if avoid_forbidden_zone and 60.0 <= target_rz_deg <= 80.0:
        # 尝试180°对称角度
        alternative_deg = target_rz_deg - 180.0
        if alternative_deg < -180.0:
            alternative_deg += 360.0
        
        if not (60.0 <= alternative_deg <= 80.0):
            target_rz_deg = alternative_deg
            print(f"       \033[93m[限位保护] 使用180°对称角度: {target_rz_deg:.1f}°\033[0m")
        else:
            # 两个都在限位区，选择边界
            if abs(target_rz_deg - 60) < abs(target_rz_deg - 80):
                target_rz_deg = 55.0
            else:
                target_rz_deg = 85.0
            print(f"       \033[93m[限位保护] 调整为边界: {target_rz_deg:.1f}°\033[0m")
    
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
    """从全局观察点进行检测（后备方案）"""
    print("\n  -> 【全局观察】移动到全局视角...")
    success = move_to_observation_point(
        robot, 
        GLOBAL_OBSERVATION_CONFIG['pos'], 
        GLOBAL_OBSERVATION_CONFIG['quat']
    )
    
    if not success:
        return None
    
    camera_manager.update_robot_pose(robot)
    return camera_manager.get_detection(stabilize_time=2.0)


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
    robot.MotionCtrl_2(0x01, 0x00, 30, 0x00)
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
def refine_position_to_center(robot, camera_manager, current_pos, pixel_offset, 
                              img_center_x=320, img_center_y=240,
                              max_iterations=8, tolerance_pixels=10):
    """
    通过PID控制，将积木精确对准画面中心
    【PID版本】简单、稳定、快速收敛
    """
    global GLOBAL_PIXEL_TO_MM_RATIO
    
    print(f"\n  -> 【精确对中-PID】开始迭代...")
    print(f"     图像中心: ({img_center_x}, {img_center_y})")
    
    # 获取并保存当前的夹爪姿态
    end_pose_msg = robot.GetArmEndPoseMsgs()
    current_gripper_euler = [
        end_pose_msg.end_pose.RX_axis,
        end_pose_msg.end_pose.RY_axis,
        end_pose_msg.end_pose.RZ_axis
    ]
    
    current_rz_deg = current_gripper_euler[2] / 1000.0
    print(f"     当前夹爪RZ: {current_rz_deg:.1f}°")
    print(f"     当前比例尺: X={GLOBAL_PIXEL_TO_MM_RATIO['x']:.4f}, Y={GLOBAL_PIXEL_TO_MM_RATIO['y']:.4f} mm/px")
    print(f"     当前方向: dir_x={GLOBAL_PIXEL_TO_MM_RATIO['direction_x']:+d}, dir_y={GLOBAL_PIXEL_TO_MM_RATIO['direction_y']:+d}")
    
    # 创建PID控制器
    pid = PixelPIDController(kp=0.80, ki=0.0, kd=0.02)
    
    refined_pos = current_pos.copy()
    px, py = pixel_offset
    
    # 用于学习方向和比例尺
    first_iteration_done = False
    
    for iteration in range(1, max_iterations + 1):
        offset_distance = np.sqrt((px - img_center_x)**2 + (py - img_center_y)**2)
        
        print(f"\n     [迭代 {iteration}/{max_iterations}]")
        print(f"       当前像素: ({px:.1f}, {py:.1f}), 距中心: {offset_distance:.1f}px")
        print(f"       当前位置: [{refined_pos[0]:.6f}, {refined_pos[1]:.6f}, {refined_pos[2]:.6f}]")
        
        if offset_distance <= tolerance_pixels:
            print(f"       ✓ 已达精度要求 ({offset_distance:.1f} ≤ {tolerance_pixels})")
            break
        
        # 计算像素误差（目标 - 当前）
        error_px = img_center_x - px
        error_py = img_center_y - py
        
        # PID计算移动量
        move_x_mm, move_y_mm = pid.compute(error_px, error_py)
        
        print(f"       像素误差: ΔPx={error_px:.1f}, ΔPy={error_py:.1f}")
        print(f"       PID输出: ΔX={move_x_mm:.2f}mm, ΔY={move_y_mm:.2f}mm")
        
        # 限制单次移动量（安全保护）
        max_move = 50.0  # mm
        if abs(move_x_mm) > max_move:
            move_x_mm = max_move if move_x_mm > 0 else -max_move
            print(f"       [限幅] X移动量限制到 ±{max_move}mm")
        if abs(move_y_mm) > max_move:
            move_y_mm = max_move if move_y_mm > 0 else -max_move
            print(f"       [限幅] Y移动量限制到 ±{max_move}mm")
        
        # 保存移动前状态
        prev_px, prev_py = px, py
        prev_pos = refined_pos.copy()
        prev_error_px = error_px
        prev_error_py = error_py
        
        # 执行移动
        refined_pos[0] += move_x_mm / 1000.0
        refined_pos[1] += move_y_mm / 1000.0
        
        piper_pos = [int(round(p * 1000000)) for p in refined_pos]
        piper_euler = current_gripper_euler
        
        robot.MotionCtrl_2(0x01, 0x00, 25, 0x00)
        robot.EndPoseCtrl(piper_pos[0], piper_pos[1], piper_pos[2], 
                         piper_euler[0], piper_euler[1], piper_euler[2])
        time.sleep(2.0)
        
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
        desktop_data = camera_manager.get_detection(stabilize_time=1.2)
        
        if not desktop_data:
            print(f"       ✗ 迭代{iteration}: 未检测到积木")
            break
        
        # 找到最近的积木
        min_dist = float('inf')
        best_pixel = None
        
        for block_id, data in desktop_data.items():
            if len(data) >= 3 and data[2] is not None:
                temp_px, temp_py = data[2]
                temp_dist = np.sqrt((temp_px - img_center_x)**2 + (temp_py - img_center_y)**2)
                if temp_dist < min_dist:
                    min_dist = temp_dist
                    best_pixel = (temp_px, temp_py)
        
        if not best_pixel:
            print(f"       ✗ 迭代{iteration}: 无有效像素坐标")
            break
        
        px, py = best_pixel
        new_offset = np.sqrt((px - img_center_x)**2 + (py - img_center_y)**2)
        
        # 计算像素变化
        actual_pixel_change_x = px - prev_px
        actual_pixel_change_y = py - prev_py
        
        print(f"       实际像素变化: ΔPx={actual_pixel_change_x:.1f}, ΔPy={actual_pixel_change_y:.1f}")
        print(f"       距离变化: {offset_distance:.1f}px → {new_offset:.1f}px")
        
        # ============ 第一次迭代：学习方向和比例尺 ============
        if iteration == 1 and not first_iteration_done:
            print(f"       [第一次迭代] 学习参数")
            
            # 学习方向
            if abs(prev_error_px) > 5 and abs(actual_pixel_change_x) > 2:
                if (prev_error_px * actual_pixel_change_x) < 0:
                    GLOBAL_PIXEL_TO_MM_RATIO['direction_x'] *= -1
                    print(f"       [学习] X方向反转: dir_x={GLOBAL_PIXEL_TO_MM_RATIO['direction_x']:+d}")
            
            if abs(prev_error_py) > 5 and abs(actual_pixel_change_y) > 2:
                if (prev_error_py * actual_pixel_change_y) < 0:
                    GLOBAL_PIXEL_TO_MM_RATIO['direction_y'] *= -1
                    print(f"       [学习] Y方向反转: dir_y={GLOBAL_PIXEL_TO_MM_RATIO['direction_y']:+d}")
            
            # 学习比例尺（粗略估计）
            if abs(actual_move_y) > 0.5 and abs(actual_pixel_change_x) > 5:
                measured_ratio_x = abs(actual_move_y * 1000) / abs(actual_pixel_change_x)
                if 0.04 < measured_ratio_x < 0.12:
                    GLOBAL_PIXEL_TO_MM_RATIO['x'] = measured_ratio_x
                    print(f"       [学习X比例] {abs(actual_move_y*1000):.2f}mm ÷ {abs(actual_pixel_change_x):.1f}px = {measured_ratio_x:.4f} mm/px")
            
            if abs(actual_move_x) > 0.5 and abs(actual_pixel_change_y) > 5:
                measured_ratio_y = abs(actual_move_x * 1000) / abs(actual_pixel_change_y)
                if 0.04 < measured_ratio_y < 0.12:
                    GLOBAL_PIXEL_TO_MM_RATIO['y'] = measured_ratio_y
                    print(f"       [学习Y比例] {abs(actual_move_x*1000):.2f}mm ÷ {abs(actual_pixel_change_y):.1f}px = {measured_ratio_y:.4f} mm/px")
            
            GLOBAL_PIXEL_TO_MM_RATIO['update_count'] += 1
            first_iteration_done = True
            
            # 重置PID（使用新学到的参数）
            pid.reset()
            print(f"       [学习完成] PID已重置，下次使用新参数")
    
    final_offset = np.sqrt((px - img_center_x)**2 + (py - img_center_y)**2)
    if final_offset <= tolerance_pixels:
        print(f"\n  ✓ 精确对中成功！最终误差: {final_offset:.1f}px")
    else:
        print(f"\n  ⚠ 达最大迭代次数，最终误差: {final_offset:.1f}px")
    
    print(f"  [最终参数] 比例尺: X={GLOBAL_PIXEL_TO_MM_RATIO['x']:.4f}, Y={GLOBAL_PIXEL_TO_MM_RATIO['y']:.4f} mm/px")
    
    return refined_pos, (px, py)


# ================= 候选积木选择（唯一正确版本）=================
def select_best_candidate(processed_yolo_data, yolo_prefix, robot, camera_manager,
                         detected_angle_rad, enable_refinement=True, 
                         img_center_x=320, img_center_y=240):
    """选择最接近画面中心的积木，先旋转夹爪再精确对中"""
    candidates = [k for k in processed_yolo_data.keys() if k.startswith(yolo_prefix)]
    
    if not candidates:
        return None, None, None
    
    best_candidate = None
    min_dist = float('inf')
    
    print(f"  -> 候选积木: {candidates}")
    for cand_id in candidates:
        data = processed_yolo_data[cand_id]
        
        if len(data) >= 3 and data[2] is not None:
            px, py = data[2]
            dist = np.sqrt((px - img_center_x)**2 + (py - img_center_y)**2)
            print(f"    -> {cand_id}: 像素({px:.1f}, {py:.1f}), 距中心{dist:.1f}px")
            
            if dist < min_dist:
                min_dist = dist
                best_candidate = cand_id
    
    if not best_candidate:
        return None, None, None
    
    selected_data = processed_yolo_data[best_candidate]
    print(f"  -> 初选: '{best_candidate}' (距中心 {min_dist:.2f}px)")
    
    # 【修复】获取当前夹爪RZ角度
    end_pose_msg = robot.GetArmEndPoseMsgs()
    current_rz_deg = end_pose_msg.end_pose.RZ_axis / 1000.0
    
    # 计算目标夹爪角度（考虑当前RZ）
    gripper_angle_rad, gripper_angle_deg = calculate_gripper_angle(
        detected_angle_rad, current_rz_deg
    )
    print(f"  -> 计算夹爪角度: {gripper_angle_deg:.1f}°")
    rotate_gripper_to_angle(robot, gripper_angle_rad)
    
    # 精确对中
    if enable_refinement and len(selected_data) >= 3 and selected_data[2] is not None:
        end_pose_msg = robot.GetArmEndPoseMsgs()
        current_pos = np.array([
            end_pose_msg.end_pose.X_axis / 1000000.0,
            end_pose_msg.end_pose.Y_axis / 1000000.0,
            end_pose_msg.end_pose.Z_axis / 1000000.0
        ])
        
        refined_pos, final_pixel = refine_position_to_center(
            robot, camera_manager, current_pos, selected_data[2],
            img_center_x=img_center_x, img_center_y=img_center_y
        )
        
        selected_data[0][0] = refined_pos[0]
        selected_data[0][1] = refined_pos[1]
        selected_data[2] = final_pixel
        
        print(f"  -> 精确对中后: [{refined_pos[0]:.6f}, {refined_pos[1]:.6f}]")
    
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
def observe_from_local_view(robot, camera_manager, rough_pos, observation_z=0.30):
    """
    从局部位置上方进行精细检测
    
    Returns:
        tuple: (detection_data, is_local_view)
               - detection_data: 检测结果字典
               - is_local_view: True=局部观察成功, False=已回退到全局观察
    """
    local_obs_pos = [rough_pos[0] - 0.05, rough_pos[1], observation_z]
    local_obs_quat = [-160, 0, -90]
    
    print(f"  -> 【局部观察】移动到 [{local_obs_pos[0]:.3f}, {local_obs_pos[1]:.3f}, {local_obs_pos[2]:.3f}]...")
    success = move_to_observation_point(robot, local_obs_pos, local_obs_quat)
    
    if not success:
        print("  ✗ 局部观察位置移动失败")
        return None, False
    
    camera_manager.update_robot_pose(robot)
    detection = camera_manager.get_detection()
    
    if not detection:
        print("  ✗ 局部未检测到积木")
        return None, False
    
    # 局部观察成功
    return detection, True


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
        scheduler = TaskScheduler(first_block_target_pos=[0.0200, -0.200, 0.15])
        
        build_order = (
            scheduler.architecture["layer_1"] + scheduler.architecture["layer_2"] +
            scheduler.architecture["layer_3"] + scheduler.architecture["layer_4"]
        )

        print("\n" + "="*60)
        print("--- 步骤1: 全局观察 ---")
        print("="*60)
        
        global_yolo_data = observe_from_global_view(robot, camera_manager)
        print(f"global_yolo_data: {global_yolo_data}")
        
        if not global_yolo_data:
            print("✗ 全局观察失败")
            return

        processed_global_data = preprocess_yolo_angles(
        global_yolo_data, 
        YOLO_CORRECTIONS,
        observation_rz_deg=GLOBAL_OBSERVATION_CONFIG['quat'][2]  # 传入观察点RZ
    )
        
        rough_positions = {}
        for block_id, data in processed_global_data.items():
            rough_positions[block_id] = data[0]
            print(f"  -> 粗略位置: {block_id} -> {data[0][:2]}")
        
        print("\n" + "="*60)
        print("--- 步骤2: 循环构建 ---")
        print("="*60)
        
        global_start_time = time.time()

        for task_idx, block_id in enumerate(build_order):
            print(f"\n>>> [{task_idx + 1}/{len(build_order)}] 处理积木: {block_id}")
            
            # ============ 修复：增加重试机制 ============
            max_retries = 3
            local_yolo_data = None
            processed_local_data = None
            is_local_observation = False
            
            for retry in range(max_retries):
                print(f"\n  === 尝试 {retry + 1}/{max_retries} ===")
                
                # 尝试局部观察
                rough_pos = rough_positions.get(block_id, scheduler.instances[block_id]["initial_pos"]).copy()
                if isinstance(rough_pos, np.ndarray):
                    rough_pos = rough_pos.tolist()
                
                rough_pos[0]-= 0.10  # 局部观察点X偏移
                rough_pos[2] = 0.32  # 设置为观察高度
                local_yolo_data, is_local = observe_from_local_view(robot, camera_manager, rough_pos)
                
                if local_yolo_data and is_local:
                    # 局部观察成功
                    processed_local_data = preprocess_yolo_angles(
                        local_yolo_data, 
                        YOLO_CORRECTIONS,
                        observation_rz_deg=-90.0  # 局部观察点也是RZ=-90°
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
                
                # 局部观察失败，进行全局重新观察
                print(f"  -> 执行全局重新观察（重试 {retry + 1}/{max_retries}）...")
                global_yolo_data = observe_from_global_view(robot, camera_manager)
                
                if not global_yolo_data:
                    print(f"  ✗ 全局观察失败")
                    continue
                
                processed_global_data = preprocess_yolo_angles(global_yolo_data, YOLO_CORRECTIONS)
                
                # 更新所有积木的粗略位置
                rough_positions = {}
                for bid, data in processed_global_data.items():
                    rough_positions[bid] = data[0]
                
                print(f"  -> 已更新粗略位置表（共{len(rough_positions)}个积木）")
                
                # 检查当前积木是否在新的全局观察中
                target_type = scheduler.instances[block_id]["type"]
                yolo_prefix = TYPE_TO_YOLO_PREFIX.get(target_type)
                found_in_global = any(k.startswith(yolo_prefix) for k in processed_global_data.keys())
                
                if not found_in_global:
                    print(f"  ⚠ 全局观察中未找到类型 {target_type} 的积木")
                    if retry == max_retries - 1:
                        print(f"  ✗ 已达最大重试次数，放弃该积木")
                        break
                else:
                    print(f"  ✓ 全局观察中找到类型 {target_type}，准备下一次局部观察")
            
            # 检查最终是否成功
            if not processed_local_data or not is_local_observation:
                print(f"  ✗ 最终失败：无法从局部观察获取积木 {block_id}，跳过")
                continue
            
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
                detected_angle_rad, enable_refinement=True
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
                executor.execute_task(current_task)
                scheduler.update_placement_error(block_id, executor.last_place_error_x)
                
                elapsed = time.time() - global_start_time
                print(f"  ✓ 完成 {block_id}，累计耗时: {elapsed:.2f}s")

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
