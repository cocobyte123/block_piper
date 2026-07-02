import numpy as np
import time
import traceback
import cv2
import threading
from scipy.spatial.transform import Rotation

from command_executor import CommandExecutor
from task_scheduler import TaskScheduler
from piper_sdk import C_PiperInterface_V2
from detection_system import DetectionSystem

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
    'pos': [0.105, 0.05, 0.32],
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
    """独立线程显示相机画面和YOLO检测结果"""
    def __init__(self, detector, detection_lock=None, detection_interval=0.5, main_detection_event=None):
        super().__init__(daemon=True)
        self.detector = detector
        self.detection_lock = detection_lock
        self.main_detection_event = main_detection_event
        self.detection_interval = detection_interval
        self.max_detection_age = 1.0
        self.last_detection_time = 0.0
        self.last_detection_result = None
        self.running = True
        self.show_detections = True
    
    def _draw_text_with_shadow(self, image, text, org, scale=0.55, color=(255, 255, 255), thickness=1):
        cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

    def draw_alignment_overlay(self, image, detection_count=0, detection_age=None):
        overlay = image.copy()
        h, w = image.shape[:2]
        center_x, center_y = w // 2, h // 2

        cv2.line(overlay, (center_x - 32, center_y), (center_x + 32, center_y), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(overlay, (center_x, center_y - 32), (center_x, center_y + 32), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(overlay, (center_x, center_y), 24, (80, 220, 255), 1, cv2.LINE_AA)
        cv2.circle(overlay, (center_x, center_y), 3, (0, 255, 255), -1, cv2.LINE_AA)
        image = cv2.addWeighted(overlay, 0.45, image, 0.55, 0)

        mode = "DETECT ON" if self.show_detections else "DETECT OFF"
        age_text = "--" if detection_age is None else f"{detection_age:.1f}s"
        self._draw_text_with_shadow(
            image,
            f"{mode}  blocks:{detection_count}  age:{age_text}   D toggle / Q quit",
            (12, h - 16),
            scale=0.55,
            color=(120, 255, 180) if self.show_detections else (190, 190, 190),
        )
        return image
        
    def draw_detection_results(self, image, detection_data):
        """
        在图像上绘制YOLO检测结果（参考 detection_system.py 的绘制风格）
        
        Args:
            image: OpenCV图像
            detection_data: YOLO检测结果字典
        
        Returns:
            绘制了检测框的图像
        """
        if not detection_data:
            return image
        
        # 定义不同积木类型的颜色（保持简洁）
        type_colors = {
            'code1': (0, 255, 0),      # 绿色
            'code2': (255, 0, 0),      # 蓝色  
            'code3': (0, 0, 255),      # 红色
            'code4': (0, 255, 255),    # 黄色
        }
        
        overlay = image.copy()
        image_h, image_w = image.shape[:2]
        image_center_x, image_center_y = image_w // 2, image_h // 2
        
        for block_id, data in detection_data.items():
            try:
                # 获取数据
                world_pos = data[0][:3]  # 世界坐标 [x, y, z]
                angle_rad = data[1]      # 角度（弧度）
                pixel_pos = data[2] if len(data) >= 3 and data[2] is not None else None
                
                # 确定颜色
                color = (128, 128, 128)  # 默认灰色
                for prefix, type_color in type_colors.items():
                    if block_id.startswith(prefix):
                        color = type_color
                        break
                
                if pixel_pos:
                    cx, cy = int(pixel_pos[0]), int(pixel_pos[1])
                    
                    # ============ 参考 detection_system.py 的绘制方式 ============
                    
                    # 1. 绘制OBB检测框（固定大小，类似 detection_system.py）
                    w, h = 80, 80  # 固定检测框大小
                    angle_degrees = np.degrees(angle_rad)
                    
                    # 绘制旋转矩形框
                    rect = ((cx, cy), (w, h), angle_degrees)
                    box = cv2.boxPoints(rect).astype(int)
                    cv2.drawContours(overlay, [box], 0, color, 2)
                    
                    # 2. 绘制中心点（红色圆点）
                    cv2.circle(overlay, (cx, cy), 5, (0, 0, 255), -1)
                    dx = cx - image_center_x
                    dy = cy - image_center_y
                    distance_px = float(np.hypot(dx, dy))
                    cv2.line(overlay, (image_center_x, image_center_y), (cx, cy), color, 1, cv2.LINE_AA)
                    
                    # 3. 绘制方向箭头（蓝色箭头）
                    arrow_length = min(w, h) / 2
                    end_x = int(cx + arrow_length * np.cos(angle_rad))
                    end_y = int(cy + arrow_length * np.sin(angle_rad))
                    cv2.arrowedLine(overlay, (cx, cy), (end_x, end_y), 
                                (255, 0, 0), 3, tipLength=0.3)
                    
                    label = f"{block_id}  d={distance_px:.0f}px"
                    cv2.putText(overlay, label, 
                            (cx - w//2, cy - h//2 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    offset_text = f"dx={dx:+d}px dy={dy:+d}px"
                    cv2.putText(overlay, offset_text,
                            (cx - w//2, cy + h//2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                    
            except Exception as e:
                print(f"绘制检测结果失败 {block_id}: {e}")
                continue
        
        return overlay
    
    def run(self):
        cv2.namedWindow('YOLO Detection Live View', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('YOLO Detection Live View', 1280, 720)
        
        print("📹 相机可视化启动!")
        print("   - 按 'D' 键切换检测框显示")
        print("   - 按 'Q' 键退出可视化")
        
        while self.running:
            lock_acquired = False
            color_image = None
            detection_to_draw = None
            detection_count = 0
            detection_age = None
            try:
                if self.main_detection_event and self.main_detection_event.is_set():
                    time.sleep(0.03)
                    continue

                if self.detection_lock:
                    lock_acquired = self.detection_lock.acquire(blocking=False)
                    if not lock_acquired:
                        time.sleep(0.02)
                        continue

                # 获取原始图像
                color_frame, _ = self.detector.get_frames()
                if not color_frame:
                    time.sleep(0.01)
                    continue
                
                color_image = np.asanyarray(color_frame.get_data())
                
                # 如果需要显示检测结果
                if self.show_detections:
                    try:
                        now = time.time()
                        if now - self.last_detection_time >= self.detection_interval:
                            self.last_detection_result = self.detector.run_single_detection()
                            self.last_detection_time = now
                        detection_age = now - self.last_detection_time if self.last_detection_time else None
                        is_fresh = detection_age is not None and detection_age <= self.max_detection_age
                        if self.last_detection_result and is_fresh:
                            detection_to_draw = self.last_detection_result
                            detection_count = len(self.last_detection_result)
                    except Exception as e:
                        # 检测失败时继续显示原图
                        cv2.putText(color_image, f"Detection Error: {str(e)[:50]}", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
            except Exception as e:
                print(f"可视化线程错误: {e}")
                time.sleep(0.1)
            finally:
                if lock_acquired:
                    self.detection_lock.release()

            if color_image is None:
                continue

            if detection_to_draw:
                color_image = self.draw_detection_results(color_image, detection_to_draw)

            if self.show_detections and self.last_detection_time:
                detection_age = time.time() - self.last_detection_time
                if detection_age > self.max_detection_age:
                    detection_count = 0

            color_image = self.draw_alignment_overlay(color_image, detection_count, detection_age)
            cv2.imshow('YOLO Detection Live View', color_image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("用户请求退出可视化")
                break
            elif key == ord('d') or key == ord('D'):
                self.show_detections = not self.show_detections
                status = "ON" if self.show_detections else "OFF"
                print(f"检测框显示已切换为: {status}")

            time.sleep(0.02)
        
        cv2.destroyAllWindows()
        print("📹 相机可视化已关闭")
    
    def stop(self):
        self.running = False


# ================= 修改相机管理类，增加稳定性检测 =================
class CameraManager:
    """相机管理类：保持相机持续开启，按需获取检测结果"""
    def __init__(self, enable_visualization=True):
        self.detector = DetectionSystem()
        self.detection_lock = threading.Lock()
        self.main_detection_event = threading.Event()
        self.enable_visualization = enable_visualization
        self.viz_thread = None
        
        if enable_visualization:
            self.viz_thread = CameraVisualizationThread(
                self.detector,
                self.detection_lock,
                main_detection_event=self.main_detection_event,
            )
            self.viz_thread.start()
            print("✓ 相机可视化线程已启动")
            print("   - 默认显示原始画面；按 'D' 后以低频率显示YOLO检测框和标签")
            print("   - 按 'D' 切换检测框显示，按 'Q' 关闭窗口")
        
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
        return self._run_single_detection()
    
    def _run_single_detection(self):
        self.main_detection_event.set()
        wait_start = time.perf_counter()
        try:
            with self.detection_lock:
                wait_time = time.perf_counter() - wait_start
                if wait_time > 0.2:
                    print(f"  ⚠ 等待相机/YOLO锁 {wait_time:.2f}s")
                detect_start = time.perf_counter()
                result = self.detector.run_single_detection()
                detect_time = time.perf_counter() - detect_start
                if detect_time > 0.8:
                    print(f"  ⚠ 单次YOLO检测耗时 {detect_time:.2f}s")
                return result
        finally:
            self.main_detection_event.clear()
    
    
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

            detection = self._run_single_detection()
            if detection:
                pass
            else:
                print(" 无检测结果")
            time.sleep(0.02)
        
        # 采集稳定样本
        samples = []
        
        for i in range(num_samples):
 
            
            detection = self._run_single_detection()
            
            if detection:
                samples.append(detection)
         
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
   
    
    print(f"*********************[角度计算] 积木角度: {detected_angle_deg:.1f}°")
    print(f"**********************[当前状态] 夹爪RZ: {current_rz_deg:.1f}°")
    
    
    # 步骤3: 计算目标夹爪角度
    # 夹爪目标角度 = 当前RZ - 等效值
    if detected_angle_deg > 90.0:
        target_rz_deg = -90.0 - detected_angle_deg+180
    else:
        target_rz_deg = -90.0 - detected_angle_deg
    
    print(f"计算: {current_rz_deg:.1f}° - ({detected_angle_deg:.1f}°) = {target_rz_deg:.1f}°")
    
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

def calculate_alignment_move_mm(error_px, error_py, current_rz_deg):
    """
    将图像中心误差转换为机械臂XY移动量，并根据当前RZ补偿相机视角旋转。

    error_px/error_py 使用“图像中心 - 当前像素”的方向；内部会转换成
    calculate_pixel_to_arm_transform 所需的“当前像素 - 图像中心”偏移。
    """
    pixel_offset = np.array([-error_px, -error_py])
    transform = calculate_pixel_to_arm_transform(current_rz_deg)
    arm_offset_px = transform @ pixel_offset

    move_x_mm = arm_offset_px[0] * GLOBAL_PIXEL_TO_MM_RATIO['x']
    move_y_mm = arm_offset_px[1] * GLOBAL_PIXEL_TO_MM_RATIO['y']

    base_rz = CAMERA_COORDINATE_CONFIG['base_rz_deg']
    rotation_dir = CAMERA_COORDINATE_CONFIG['rotation_direction']
    relative_angle_deg = (current_rz_deg - base_rz) * rotation_dir

    return move_x_mm, move_y_mm, relative_angle_deg


# ================= 移动与观察函数 =================
def move_to_observation_point(robot, pos, quat, speed=50):
    """移动到指定观察点"""
    piper_pos = [int(round(p * 1000000)) for p in pos]
    piper_euler = [int(round(e * 1000)) for e in quat]
    
    robot.MotionCtrl_2(0x01, 0x00, speed, 0x00)
    robot.EndPoseCtrl(piper_pos[0], piper_pos[1], piper_pos[2], 
                      piper_euler[0], piper_euler[1], piper_euler[2])
    time.sleep(1.5)    # 移动到观察点位后等待
    
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
    
    camera_manager.update_robot_pose(robot)
    if not success:
        print("  ⚠ 未到达全局观察点，使用当前视角继续检测")
    time.sleep(0.15 if success else 0.02)
    
    # 使用稳定检测：跳过2帧，采样4次
    return camera_manager.get_stable_detection(
        num_samples=2 if success else 1, 
        skip_frames=2 if success else 1, 
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
    
    camera_manager.update_robot_pose(robot)
    if not success:
        print("  ⚠ 未到达局部观察点，使用当前视角继续检测")
    time.sleep(0.12 if success else 0.02)
    
    # 使用稳定检测：跳过1帧，采样3次（局部观察要求稍低）
    detection = camera_manager.get_stable_detection(
        num_samples=2 if success else 1, 
        skip_frames=2 if success else 1, 
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
    time.sleep(1.0)   # 旋转夹爪后等待
    
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
        self.has_prev_error = False
    
    def compute_pixel_output(self, error_x, error_y, dt=1.0):
        """
        计算PID像素输出
        
        Args:
            error_x: X方向像素误差（目标 - 当前）
            error_y: Y方向像素误差
            dt: 时间步长（这里固定为1）
        
        Returns:
            (output_px_x, output_px_y): 本轮需要修正的像素量
        """
        # 积分项
        self.integral_x += error_x * dt
        self.integral_y += error_y * dt
        
        # 微分项：第一次没有历史误差，不引入额外冲击
        if self.has_prev_error:
            derivative_x = (error_x - self.prev_error_x) / dt
            derivative_y = (error_y - self.prev_error_y) / dt
        else:
            derivative_x = 0.0
            derivative_y = 0.0
        
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
        self.has_prev_error = True
        
        return output_px_x, output_px_y

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
        output_px_x, output_px_y = self.compute_pixel_output(error_x, error_y, dt=dt)
        
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
        self.has_prev_error = False

def pick_tracked_block_candidate(detection_data, target_yolo_prefix, tracking_mode,
                                 reference_world_pos, img_center_x=320, img_center_y=240):
    """按当前跟踪模式，从检测结果中选择一个同类型目标。"""
    if not detection_data:
        return None, []

    candidates = []
    image_center = np.array([img_center_x, img_center_y])
    reference_world_pos = np.array(reference_world_pos[:3])

    for block_id, data in detection_data.items():
        if not block_id.startswith(target_yolo_prefix):
            continue
        if len(data) < 3 or data[2] is None:
            continue

        world_pos = np.array(data[0][:3])
        pixel_pos = np.array(data[2])

        if tracking_mode == "center":
            metric = np.linalg.norm(pixel_pos - image_center)
        else:
            metric = np.linalg.norm(world_pos - reference_world_pos)

        candidates.append((block_id, world_pos, pixel_pos, metric))

    if not candidates:
        return None, []

    candidates.sort(key=lambda item: item[3])
    return candidates[0], candidates

def wait_for_stable_target_detection(robot, camera_manager, target_yolo_prefix, tracking_mode,
                                     reference_world_pos, img_center_x=320, img_center_y=240,
                                     stable_frames=3, stable_time=0.35,
                                     stable_pixel_tolerance=4.0, max_wait_time=1.0,
                                     sample_interval=0.03):
    """等待目标像素位置稳定，返回最近一次稳定检测结果。"""
    print(
        f"       [自适应稳定检测] 连续{stable_frames}帧 / {stable_time:.2f}s, "
        f"像素容差{stable_pixel_tolerance:.1f}px, 最多等待{max_wait_time:.2f}s"
    )

    start_time = time.perf_counter()
    last_pixel = None
    last_block_id = None
    stable_count = 0
    stable_since = None
    last_detection = None
    attempts = 0

    while True:
        attempts += 1
        camera_manager.update_robot_pose(robot)
        detection = camera_manager.get_detection(stabilize_time=0.0)
        now = time.perf_counter()

        selected, _ = pick_tracked_block_candidate(
            detection,
            target_yolo_prefix,
            tracking_mode,
            reference_world_pos,
            img_center_x=img_center_x,
            img_center_y=img_center_y
        )

        if selected is not None:
            block_id, _, pixel_pos, metric = selected
            last_detection = detection

            if last_pixel is not None:
                pixel_delta = np.linalg.norm(pixel_pos - last_pixel)
            else:
                pixel_delta = float("inf")

            same_block = last_block_id == block_id
            if last_pixel is not None and same_block and pixel_delta <= stable_pixel_tolerance:
                stable_count += 1
                if stable_since is None:
                    stable_since = now
            else:
                stable_count = 1
                stable_since = now

            stable_duration = now - stable_since
            print(
                f"       稳定帧{attempts}: {block_id}, "
                f"Δpixel={0.0 if pixel_delta == float('inf') else pixel_delta:.1f}px, "
                f"稳定{stable_count}帧/{stable_duration:.2f}s"
            )

            if stable_count >= stable_frames and stable_duration >= stable_time:
                print(f"       ✓ 目标检测已稳定，耗时{now - start_time:.2f}s")
                return detection

            last_pixel = pixel_pos
            last_block_id = block_id
        else:
            print(f"       稳定帧{attempts}: 未检测到目标类型 {target_yolo_prefix}")

        if now - start_time >= max_wait_time:
            print(f"       ⚠ 稳定等待超时，使用最近一次检测结果")
            return last_detection

        time.sleep(sample_interval)

# ================= 精确对中函数（PID版本）=================
# ============ 修改精确对中函数，返回最终跟踪积木的角度 ============
def refine_position_to_center_with_spatial_tracking(robot, camera_manager, current_pos, initial_pixel_offset, 
                                                   target_yolo_prefix, initial_world_pos,
                                                   img_center_x=320, img_center_y=240,
                                                   max_iterations=3, tolerance_pixels=20,
                                                   stage_name="精确对中",
                                                   tracking_mode="world",
                                                   use_pid=False,
                                                   pid_kp=0.35,
                                                   pid_ki=0.0,
                                                   pid_kd=0.05,
                                                   max_move_mm=50.0,
                                                   motion_speed=30,
                                                   move_settle_time=0.05,
                                                   detect_stabilize_time=0.0,
                                                   adaptive_stability=False,
                                                   stable_frames=3,
                                                   stable_time=0.35,
                                                   stable_pixel_tolerance=4.0,
                                                   stable_max_wait=1.0,
                                                   stable_sample_interval=0.03):
    """
    空间跟踪精确对中：基于空间位置和类型跟踪，避免ID混淆
    
    Args:
        target_yolo_prefix: 目标积木类型前缀（如"code1"）
        initial_world_pos: 初始选择积木的世界坐标 [x, y, z]
        
    Returns:
        (refined_pos, final_pixel): 精确对中后的位置、最终像素坐标
    """
    print(f"\n  -> 【空间跟踪{stage_name}】开始迭代...")
    print(f"     目标类型: {target_yolo_prefix}")
    print(f"     初始世界位置: [{initial_world_pos[0]:.3f}, {initial_world_pos[1]:.3f}, {initial_world_pos[2]:.3f}]")
    print(f"     图像中心: ({img_center_x}, {img_center_y})")
    print(f"     跟踪模式: {tracking_mode}")
    print(f"     控制模式: {'PID' if use_pid else '直接比例'}")
    print(f"     最大单步: {max_move_mm:.1f}mm, 速度: {motion_speed}")
    if adaptive_stability:
        print(f"     稳定策略: 移动后{move_settle_time:.2f}s + 自适应检测稳定")
    else:
        print(f"     稳定等待: 移动后{move_settle_time:.2f}s + 检测前{detect_stabilize_time:.2f}s")
    
    # 获取当前夹爪姿态
    end_pose_msg = robot.GetArmEndPoseMsgs()
    current_gripper_euler = [
        end_pose_msg.end_pose.RX_axis,
        end_pose_msg.end_pose.RY_axis,
        end_pose_msg.end_pose.RZ_axis
    ]
    
    current_rz_deg = current_gripper_euler[2] / 1000.0
    print(f"     当前夹爪RZ: {current_rz_deg:.1f}°")
    
    pid_controller = PixelPIDController(kp=pid_kp, ki=pid_ki, kd=pid_kd) if use_pid else None
    
    refined_pos = current_pos.copy()
    px, py = initial_pixel_offset
    
    # 记录初始检测状态
    initial_count = None
    
    for iteration in range(1, max_iterations + 1):
        offset_distance = np.sqrt((px - img_center_x)**2 + (py - img_center_y)**2)
        
        print(f"\n     [迭代 {iteration}/{max_iterations}]")
        print(f"       当前像素: ({px:.1f}, {py:.1f}), 距中心: {offset_distance:.1f}px")
        
        if offset_distance <= tolerance_pixels:
            print(f"       ✓ 已达精度要求")
            break
        
        # 计算移动量
        error_px = img_center_x - px
        error_py = img_center_y - py
        
        if use_pid:
            control_px_x, control_px_y = pid_controller.compute_pixel_output(error_px, error_py, dt=1.0)
            print(f"       PID参数: kp={pid_kp:.2f}, ki={pid_ki:.2f}, kd={pid_kd:.2f}")
        else:
            control_px_x, control_px_y = error_px, error_py
        
        raw_move_x_mm, raw_move_y_mm, relative_angle_deg = calculate_alignment_move_mm(
            control_px_x,
            control_px_y,
            current_rz_deg
        )
        print(
            f"       像素修正: 误差({error_px:.1f}, {error_py:.1f})px "
            f"→ 本步({control_px_x:.1f}, {control_px_y:.1f})px, "
            f"RZ方向补偿={relative_angle_deg:.1f}°"
        )
        
        # 限制移动量
        max_move = max_move_mm
        move_x_mm = max(-max_move, min(max_move, raw_move_x_mm))
        move_y_mm = max(-max_move, min(max_move, raw_move_y_mm))
        
        print(
            f"       移动指令: 原始ΔX={raw_move_x_mm:.2f}mm, 原始ΔY={raw_move_y_mm:.2f}mm "
            f"→ 限幅后ΔX={move_x_mm:.2f}mm, ΔY={move_y_mm:.2f}mm"
        )
        
        # 执行移动
        command_start_pos = refined_pos.copy()
        refined_pos[0] += move_x_mm / 1000.0
        refined_pos[1] += move_y_mm / 1000.0
        commanded_pos = refined_pos.copy()
        
        piper_pos = [int(round(p * 1000000)) for p in refined_pos]
        
        robot.MotionCtrl_2(0x01, 0x00, motion_speed, 0x00)
        robot.EndPoseCtrl(piper_pos[0], piper_pos[1], piper_pos[2], 
                         current_gripper_euler[0], current_gripper_euler[1], current_gripper_euler[2])
        time.sleep(move_settle_time)    # 对中微调等待（s）
        
        # 重新检测
        camera_manager.update_robot_pose(robot)
        detect_start = time.perf_counter()
        if adaptive_stability:
            desktop_data = wait_for_stable_target_detection(
                robot,
                camera_manager,
                target_yolo_prefix,
                tracking_mode,
                initial_world_pos,
                img_center_x=img_center_x,
                img_center_y=img_center_y,
                stable_frames=stable_frames,
                stable_time=stable_time,
                stable_pixel_tolerance=stable_pixel_tolerance,
                max_wait_time=stable_max_wait,
                sample_interval=stable_sample_interval
            )
        else:
            desktop_data = camera_manager.get_detection(stabilize_time=detect_stabilize_time)
        detect_elapsed = time.perf_counter() - detect_start
        print(f"       重新检测耗时: {detect_elapsed:.2f}s")

        # 在稳定检测之后再读取实际位置，避免下一轮使用运动未完成时的旧位姿。
        end_pose_msg = robot.GetArmEndPoseMsgs()
        actual_pos = np.array([
            end_pose_msg.end_pose.X_axis / 1000000.0,
            end_pose_msg.end_pose.Y_axis / 1000000.0,
            end_pose_msg.end_pose.Z_axis / 1000000.0
        ])
        actual_move_mm = np.linalg.norm((actual_pos[:2] - command_start_pos[:2]) * 1000.0)
        command_error_mm = np.linalg.norm((commanded_pos[:2] - actual_pos[:2]) * 1000.0)
        print(f"       实际XY移动: {actual_move_mm:.2f}mm, 距指令点: {command_error_mm:.2f}mm")
        refined_pos = actual_pos
        
        if not desktop_data:
            print(f"       ✗ 迭代{iteration}: 未检测到积木")
            break
        
        # ============ 关键修改：基于空间位置和类型数量跟踪 ============
        
        # 1. 统计同类型积木数量
        same_type_blocks = []
        for block_id, data in desktop_data.items():
            if (block_id.startswith(target_yolo_prefix) and 
                len(data) >= 3 and data[2] is not None):
                world_pos = data[0][:3]
                pixel_pos = data[2]
                same_type_blocks.append((block_id, world_pos, pixel_pos))
        
        current_count = len(same_type_blocks)
        
        # 记录初始数量
        if initial_count is None:
            initial_count = current_count
            print(f"       [初始状态] {target_yolo_prefix}类型积木数量: {initial_count}")
        
        print(f"       [检测状态] {target_yolo_prefix}类型积木: {current_count}个 (初始:{initial_count})")
        
        if current_count == 0:
            print(f"       ✗ 目标类型积木完全丢失")
            break
        
        # 2. 检查数量变化（异常情况）
        if current_count != initial_count:
            print(f"       ⚠ 积木数量变化！{initial_count} → {current_count}")
            if current_count > initial_count:
                print(f"         可能原因: 新积木进入视野，或检测分裂")
            else:
                print(f"         可能原因: 积木被遮挡，或检测合并")
        
        # 3. 目标跟踪：粗对中按世界坐标，旋转后精对中按画面中心最近。
        if current_count == 1:
            # 只有一个候选，直接使用
            selected_block_id, selected_world_pos, selected_pixel_pos = same_type_blocks[0]
            print(f"       ✓ 唯一候选: {selected_block_id}")
        elif tracking_mode == "center":
            distances = []
            image_center = np.array([img_center_x, img_center_y])
            for block_id, world_pos, pixel_pos in same_type_blocks:
                pixel_distance = np.linalg.norm(np.array(pixel_pos) - image_center)
                distances.append((block_id, world_pos, pixel_pos, pixel_distance))

            distances.sort(key=lambda x: x[3])
            selected_block_id, selected_world_pos, selected_pixel_pos, min_distance = distances[0]

            print(f"       [中心最近跟踪] 候选像素距离:")
            for i, (bid, wpos, ppos, dist) in enumerate(distances[:3]):
                marker = "★" if i == 0 else " "
                print(f"         {marker} {bid}: 距中心{dist:.1f}px, 像素({ppos[0]:.1f},{ppos[1]:.1f})")

            print(f"       ✓ 选择中心最近候选: {selected_block_id} (距中心{min_distance:.1f}px)")
        else:
            # 多个候选：选择空间距离最近的
            distances = []
            for block_id, world_pos, pixel_pos in same_type_blocks:
                # 计算与初始世界位置的距离
                distance_3d = np.linalg.norm(np.array(world_pos) - np.array(initial_world_pos))
                distances.append((block_id, world_pos, pixel_pos, distance_3d))
            
            # 按距离排序，选择最近的
            distances.sort(key=lambda x: x[3])
            selected_block_id, selected_world_pos, selected_pixel_pos, min_distance = distances[0]
            
            print(f"       [空间跟踪] 候选积木距离分析:")
            for i, (bid, wpos, ppos, dist) in enumerate(distances[:3]):  # 显示前3个
                marker = "★" if i == 0 else " "
                print(f"         {marker} {bid}: 距离{dist*1000:.1f}mm, 世界({wpos[0]:.3f},{wpos[1]:.3f})")
            
            print(f"       ✓ 选择最近候选: {selected_block_id} (距离{min_distance*1000:.1f}mm)")
        
        # 4. 更新像素位置
        px, py = selected_pixel_pos
        print(f"       → 跟踪像素位置: ({px:.1f}, {py:.1f})")
        
        # 5. 更新目标世界位置（用于下次跟踪）
        initial_world_pos = selected_world_pos
    
    final_offset = np.sqrt((px - img_center_x)**2 + (py - img_center_y)**2)
    if final_offset <= tolerance_pixels:
        print(f"\n  ✓ 空间跟踪{stage_name}成功！最终误差: {final_offset:.1f}px")
    else:
        print(f"\n  ⚠ {stage_name}达最大迭代次数，最终误差: {final_offset:.1f}px")
    
    return refined_pos, (px, py), np.array(initial_world_pos)
# ================= 抓取偏移计算函数 =================
def calculate_grasp_offset(current_rz_deg, offset_distance_mm=20.0):
    """
    根据当前夹爪RZ角度计算抓取偏移量。
    
    原理：
    - 像素中心对齐后，相机/夹爪和积木的相对位置近似固定。
    - 抓取点需要从当前夹爪位置沿夹爪“前方”偏移一段距离。
    - 这里沿用原来的方向约定：
      RZ=0°   -> 世界坐标 +Y
      RZ=-90° -> 世界坐标 +X
      RZ=180° -> 世界坐标 -Y
    
    Args:
        current_rz_deg: 当前夹爪RZ角度（度）
        offset_distance_mm: 沿夹爪前进方向的偏移距离（毫米）
    
    Returns:
        (offset_x_m, offset_y_m): 机械臂基座坐标系下的XY偏移（米）
    """
    rz_rad = np.radians(current_rz_deg)
    forward_rad = rz_rad + np.pi / 2
    offset_distance_m = offset_distance_mm / 1000.0

    offset_x_m = offset_distance_m * np.cos(forward_rad)
    offset_y_m = offset_distance_m * np.sin(forward_rad)
    
    print(f"  [抓取偏移] RZ={current_rz_deg:.1f}°, 偏移{offset_distance_mm}mm")
    print(f"             → 夹爪前方角度: {np.degrees(forward_rad):.1f}°")
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
    time.sleep(1.0)    # 抓取偏移移动后等待
    
    # 验证
    end_pose_msg = robot.GetArmEndPoseMsgs()
    actual_pos = [
        end_pose_msg.end_pose.X_axis / 1000000.0,
        end_pose_msg.end_pose.Y_axis / 1000000.0,
        end_pose_msg.end_pose.Z_axis / 1000000.0
    ]
    
    print(f"     实际位置: [{actual_pos[0]:.6f}, {actual_pos[1]:.6f}, {actual_pos[2]:.6f}]")
    
    return np.array(actual_pos)

def find_nearest_tracked_candidate(detection_data, yolo_prefix, reference_world_pos):
    """从检测结果中找离参考世界坐标最近的同类型积木。"""
    if not detection_data:
        return None, None

    candidates = []
    reference_world_pos = np.array(reference_world_pos[:3])

    for block_id, data in detection_data.items():
        if not block_id.startswith(yolo_prefix) or len(data) < 3 or data[2] is None:
            continue
        world_pos = np.array(data[0][:3])
        distance = np.linalg.norm(world_pos - reference_world_pos)
        candidates.append((distance, block_id, data))

    if not candidates:
        return None, None

    candidates.sort(key=lambda item: item[0])
    distance, block_id, data = candidates[0]
    print(f"  -> [旋转后重锁定] {block_id}, 距参考位置 {distance * 1000:.1f}mm")
    return block_id, data

def find_center_tracked_candidate(detection_data, yolo_prefix, img_center_x=320, img_center_y=240):
    """从检测结果中找离图像中心最近的同类型积木。"""
    if not detection_data:
        return None, None

    candidates = []
    image_center = np.array([img_center_x, img_center_y])

    for block_id, data in detection_data.items():
        if not block_id.startswith(yolo_prefix):
            continue
        pixel_pos = np.array(data[2])
        pixel_distance = np.linalg.norm(pixel_pos - image_center)
        candidates.append((pixel_distance, block_id, data))

    if not candidates:
        return None, None

    candidates.sort(key=lambda item: item[0])
    pixel_distance, block_id, data = candidates[0]
    print(f"  -> [旋转后重锁定] {block_id}, 距画面中心 {pixel_distance:.1f}px")
    return block_id, data

# ================= 修改候选积木选择函数 =================
def select_best_candidate(processed_yolo_data, yolo_prefix, robot, camera_manager,
                         target_gripper_angle_rad,
                         enable_refinement=True, 
                         enable_grasp_offset=True,
                         grasp_offset_mm=50.0,
                         img_center_x=320, img_center_y=240):
    """
    选择积木：基于真实世界坐标（先左后右，从上到下 = Y最大，相近时X最大）
    """
    candidates = [k for k in processed_yolo_data.keys() if k.startswith(yolo_prefix)]
    
    if not candidates:
        return None, None, None
    
    # ============ 修正：选择策略 = 先左后右，从上到下 ============
    world_candidates = []
    for cand_id in candidates:
        data = processed_yolo_data[cand_id]
        pos_x, pos_y, pos_z = data[0][:3]
        pixel_info = data[2] if len(data) >= 3 else None
        world_candidates.append((cand_id, pos_x, pos_y, pos_z, pixel_info))
    
    if not world_candidates:
        return None, None, None
    
    # 显示候选分析
    print(f"  -> 候选积木分析（真实世界坐标）:")
    for cand_id, pos_x, pos_y, pos_z, pixel_info in world_candidates:
        print(f"     {cand_id}: 世界坐标({pos_x:.3f}, {pos_y:.3f}, {pos_z:.3f})")
    
    # 第1步：按Y坐标排序（先左后右：Y最大优先）
    world_candidates.sort(key=lambda x: x[2], reverse=True)  # 按Y坐标降序排序
    
    # 第2步：在Y相近的候选中，选择X最大的（从上到下：X最大优先）
    max_y = world_candidates[0][2]  # 最大的Y坐标
    y_tolerance = 0.010  # Y坐标容差：10mm
    
    # 找出所有Y坐标在容差范围内的候选
    y_similar_candidates = []
    for cand_id, pos_x, pos_y, pos_z, pixel_info in world_candidates:
        if abs(pos_y - max_y) <= y_tolerance:
            y_similar_candidates.append((cand_id, pos_x, pos_y, pos_z, pixel_info))
    
    # 在Y相近的候选中，选择X最大的
    y_similar_candidates.sort(key=lambda x: x[1], reverse=True)  # 按X坐标降序排序
    selected_candidate = y_similar_candidates[0]
    best_candidate = selected_candidate[0]
    
    # 显示选择过程
    print(f"  -> [选择策略] 先左后右，从上到下:")
    print(f"               Y最大={max_y:.3f} (最左边), 容差±{y_tolerance*1000:.0f}mm")
    print(f"               Y相近候选: {len(y_similar_candidates)}个")
    
    for i, (cand_id, pos_x, pos_y, pos_z, pixel_info) in enumerate(world_candidates):
        if cand_id == best_candidate:
            marker = "★"
            extra_info = f" ← 选中 (最左且最上)"
        elif abs(pos_y - max_y) <= y_tolerance:
            marker = "◦"
            extra_info = f" (Y相近)"
        else:
            marker = " "
            extra_info = ""
        
        print(f"     {marker} {cand_id}: 世界坐标({pos_x:.3f}, {pos_y:.3f}, {pos_z:.3f}){extra_info}")
    
    print(f"  -> [选择结果] {best_candidate} (Y={selected_candidate[2]:.3f}, X={selected_candidate[1]:.3f})")
    
    selected_data = processed_yolo_data[best_candidate]
    
    print(f"  -> [角度] 使用预先计算的角度: {np.degrees(target_gripper_angle_rad):.1f}°")
    gripper_angle_rad = target_gripper_angle_rad
    
    # ============ 关键修改：使用空间跟踪精确对中 ============
    if enable_refinement and len(selected_data) >= 3 and selected_data[2] is not None:
        end_pose_msg = robot.GetArmEndPoseMsgs()
        current_pos = np.array([
            end_pose_msg.end_pose.X_axis / 1000000.0,
            end_pose_msg.end_pose.Y_axis / 1000000.0,
            end_pose_msg.end_pose.Z_axis / 1000000.0
        ])
        
        # 记录初始选择的积木世界位置
        initial_world_pos = selected_data[0][:3]
        
        print(f"  -> 开始空间跟踪精确对中")
        print(f"     选择积木: {best_candidate}")
        print(f"     初始像素: ({selected_data[2][0]:.1f}, {selected_data[2][1]:.1f})")
        print(f"     初始世界位置: [{initial_world_pos[0]:.3f}, {initial_world_pos[1]:.3f}, {initial_world_pos[2]:.3f}]")
        
        # 先做粗对中：只需要大致靠近中心，避免一上来旋转导致目标跑出视野。
        rough_pos, rough_pixel, tracked_world_pos = refine_position_to_center_with_spatial_tracking(
            robot, camera_manager, current_pos, selected_data[2], 
            yolo_prefix, initial_world_pos,
            img_center_x=img_center_x, img_center_y=img_center_y,
            max_iterations=3,
            tolerance_pixels=55,
            stage_name="粗对中",
            tracking_mode="world",
            use_pid=True,
            pid_kp=0.65,
            pid_ki=0.0,
            pid_kd=0.04,
            max_move_mm=25.0,
            motion_speed=22,
            move_settle_time=0.12,
            detect_stabilize_time=0.05
        )

        print(f"  -> 粗对中后: [{rough_pos[0]:.6f}, {rough_pos[1]:.6f}], 像素({rough_pixel[0]:.1f}, {rough_pixel[1]:.1f})")

        # 粗对中后先旋转夹爪，后续精对中基于最终抓取姿态进行。
        rotate_gripper_to_angle(robot, target_gripper_angle_rad)
        camera_manager.update_robot_pose(robot)

        rotated_detection = camera_manager.get_detection(stabilize_time=0.20)
        tracked_id, tracked_data = find_center_tracked_candidate(
            rotated_detection,
            yolo_prefix,
            img_center_x=img_center_x,
            img_center_y=img_center_y
        )

        if tracked_data is not None:
            best_candidate = tracked_id
            selected_data = tracked_data
            initial_world_pos = selected_data[0][:3]
            initial_pixel = selected_data[2]
        else:
            print("  ⚠ 旋转后未重新锁定目标，使用粗对中结果继续精对中")
            initial_world_pos = tracked_world_pos
            initial_pixel = rough_pixel

        end_pose_msg = robot.GetArmEndPoseMsgs()
        current_pos = np.array([
            end_pose_msg.end_pose.X_axis / 1000000.0,
            end_pose_msg.end_pose.Y_axis / 1000000.0,
            end_pose_msg.end_pose.Z_axis / 1000000.0
        ])

        # 旋转后再做精对中：此时相机/夹爪和积木的相对关系已经是最终抓取姿态。
        refined_pos, final_pixel, tracked_world_pos = refine_position_to_center_with_spatial_tracking(
            robot, camera_manager, current_pos, initial_pixel,
            yolo_prefix, initial_world_pos,
            img_center_x=img_center_x, img_center_y=img_center_y,
            max_iterations=5,
            tolerance_pixels=5,
            stage_name="旋转后精对中",
            tracking_mode="center",
            use_pid=True,
            pid_kp=0.60,
            pid_ki=0.0,
            pid_kd=0.04,
            max_move_mm=25.0,
            motion_speed=18,
            move_settle_time=0.10,
            detect_stabilize_time=0.0,
            adaptive_stability=True,
            stable_frames=3,
            stable_time=0.35,
            stable_pixel_tolerance=4.0,
            stable_max_wait=1.0,
            stable_sample_interval=0.03
        )
        
        print(f"  -> 精确对中后: [{refined_pos[0]:.6f}, {refined_pos[1]:.6f}]")
        print(f"  -> 最终像素: ({final_pixel[0]:.1f}, {final_pixel[1]:.1f})")
        
        # 应用抓取偏移（仅计算位置，不物理移动；偏移会合并到 execute_task 的 pre_grasp_pos 中）
        if enable_grasp_offset:
            end_pose_msg = robot.GetArmEndPoseMsgs()
            final_rz_deg = end_pose_msg.end_pose.RZ_axis / 1000.0

            # 获取当前位置（对中后的位置）
            current_pos = [
                end_pose_msg.end_pose.X_axis / 1000000.0,
                end_pose_msg.end_pose.Y_axis / 1000000.0,
                end_pose_msg.end_pose.Z_axis / 1000000.0
            ]

            offset_x_m, offset_y_m = calculate_grasp_offset(final_rz_deg, grasp_offset_mm)
            final_x = current_pos[0] + offset_x_m
            final_y = current_pos[1] + offset_y_m

            selected_data[0][0] = final_x
            selected_data[0][1] = final_y

            print(f"  ✓ 抓取偏移计入: ΔX={offset_x_m*1000:.1f}mm, ΔY={offset_y_m*1000:.1f}mm → 目标 [{final_x:.6f}, {final_y:.6f}]")
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
        # detection_system.py 在计算角度时会将机械臂姿态叠加进去
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


def main():
    np.set_printoptions(precision=4, suppress=True, linewidth=120)
    
    # 配置参数
    YOLO_CORRECTIONS = {
        "code3_1": [0.0, 0.0, -0.02, 0.0],
        "code3_2": [0.0, 0.0, -0.02, 0.0],
        "code4": [0.0, 0.0, -0.05, 0.0],
    }
    
    TYPE_TO_YOLO_PREFIX = {
        "type1": "code1", "type2": "code2", "type3": "code3", "type4": "code4"
    }

    # 初始化系统
    CAN_IFACE = "can0"
    robot = C_PiperInterface_V2(CAN_IFACE)
    robot.ConnectPort()
    while not robot.EnablePiper():
        print("等待机械臂使能...")
        time.sleep(0.5)

    camera_manager = None
    
    try:
        # 启动相机和可视化
        camera_manager = CameraManager(enable_visualization=True)
        
        # 初始化执行器和调度器
        executor = CommandExecutor(robot)
        scheduler = TaskScheduler(first_block_target_pos=[0.0800, -0.200, 0.135])
        
        # 获取构建顺序
        build_order = (
            scheduler.architecture["layer_1"] + scheduler.architecture["layer_2"] +
            scheduler.architecture["layer_3"] + scheduler.architecture["layer_4"]
        )

        print("\n" + "="*60)
        print("--- 步骤1: 初始全局观察 ---")
        print("="*60)
        
        # 初始全局观察
        global_yolo_data = observe_from_global_view(robot, camera_manager)
        
        if not global_yolo_data:
            print("✗ 初始全局观察失败")
            return
        
        # 打开夹爪
        robot.GripperCtrl(gripper_angle=80000, gripper_effort=1000, gripper_code=0x01, set_zero=0x00)
        
        # 处理全局数据
        processed_global_data = preprocess_yolo_angles(
            global_yolo_data, 
            YOLO_CORRECTIONS,
            observation_rz_deg=GLOBAL_OBSERVATION_CONFIG['quat'][2]
        )
        
        # 建立粗略位置表
        rough_positions = {}
        for block_id, data in processed_global_data.items():
            rough_positions[block_id] = data[0]
            print(f"  -> 初始粗略位置: {block_id} -> [{data[0][0]:.3f}, {data[0][1]:.3f}]")
        
        print("\n" + "="*60)
        print("--- 步骤2: 循环构建 ---")
        print("="*60)
        
        global_start_time = time.time()

        # 主循环：逐个处理积木
        for task_idx, block_id in enumerate(build_order):
            print(f"\n>>> [{task_idx + 1}/{len(build_order)}] 处理积木: {block_id}")
            
            # ============ 步骤2.1: 重新全局观察（除第一个积木外）============
            if task_idx > 0:
                print(f"\n  === 重新全局观察 ===")
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
                    
                    print(f"  ✓ 更新粗略位置表（共{len(rough_positions)}个积木）")
                else:
                    print(f"  ⚠ 重新全局观察失败，使用上轮位置")
            
            # ============ 步骤2.2: 确定局部观察位置 ============
            target_type = scheduler.instances[block_id]["type"]
            yolo_prefix = TYPE_TO_YOLO_PREFIX.get(target_type)
            
            # ============ 修正：从粗略位置表中找同类型积木，按"先左后右，从上到下"选择 ============
            same_type_positions = {}
            for bid, pos in rough_positions.items():
                if bid.startswith(yolo_prefix):
                    same_type_positions[bid] = pos
            
            if same_type_positions:
                print(f"  -> 同类型积木候选分析:")
                
                # 显示所有候选
                candidates = list(same_type_positions.items())
                for bid, pos in candidates:
                    print(f"     {bid}: 真实世界({pos[0]:.3f}, {pos[1]:.3f})")
                
                # ============ 关键修正：先左后右，从上到下 = Y最大，相近时X最大 ============
                # 第1步：按Y坐标排序（左→右：Y递减，所以要选Y最大的，即最左边的）
                candidates.sort(key=lambda x: x[1][1], reverse=True)  # 按Y坐标降序排序
                
                # 第2步：在Y相近的候选中，选择X最大的（上→下：X递减，所以要选X最大的，即最上面的）
                max_y = candidates[0][1][1]  # 最大的Y坐标（最左边）
                y_tolerance = 0.015  # Y坐标容差：15mm
                
                # 找出所有Y坐标在容差范围内的候选（同一列的积木）
                y_similar_candidates = []
                for bid, pos in candidates:
                    if abs(pos[1] - max_y) <= y_tolerance:
                        y_similar_candidates.append((bid, pos))
                
                # 在同一列中，选择X最大的（最上面的）
                y_similar_candidates.sort(key=lambda x: x[1][0], reverse=True)  # 按X坐标降序排序
                selected_bid, selected_pos = y_similar_candidates[0]
                
                rough_pos = selected_pos.copy()
                
                # 显示选择过程
                print(f"  -> [选择策略] 先左后右，从上到下:")
                print(f"               Y最大={max_y:.3f} (最左边), 容差±{y_tolerance*1000:.0f}mm")
                print(f"               Y相近候选: {len(y_similar_candidates)}个")
                
                for i, (bid, pos) in enumerate(candidates):
                    if bid == selected_bid:
                        marker = "★"
                        extra_info = f" ← 选中 (最左且最上)"
                    elif abs(pos[1] - max_y) <= y_tolerance:
                        marker = "◦"
                        extra_info = f" (Y相近)"
                    else:
                        marker = " "
                        extra_info = ""
                    
                    print(f"     {marker} {bid}: 世界({pos[0]:.3f}, {pos[1]:.3f}){extra_info}")
                
                print(f"  -> 选择局部观察目标: {selected_bid} (Y={selected_pos[1]:.3f}, X={selected_pos[0]:.3f})")
                
            else:
                # 后备：使用scheduler默认位置
                rough_pos = scheduler.instances[block_id]["initial_pos"].copy()
                print(f"  -> 使用默认位置作为局部观察目标")
            
            # ============ 步骤2.3: 局部观察（带重试）============
            max_retries = 1
            local_success = False
            processed_local_data = None
            
            for retry in range(max_retries):
                print(f"\n  === 局部观察尝试 {retry + 1}/{max_retries} ===")
                
                # 尝试局部观察
                local_yolo_data, is_local = observe_from_local_view(robot, camera_manager, rough_pos)
                
                if local_yolo_data and is_local:
                    processed_local_data = preprocess_yolo_angles(local_yolo_data, YOLO_CORRECTIONS)
                    
                    # 检查是否有目标类型积木
                    has_target = any(k.startswith(yolo_prefix) for k in processed_local_data.keys())
                    
                    if has_target:
                        print(f"  ✓ 局部观察成功，找到目标类型 {target_type}")
                        local_success = True
                        break
                    else:
                        print(f"  ⚠ 局部观察到积木，但无目标类型 {target_type}")
                else:
                    print(f"  ✗ 局部观察失败")
                
                # 重试：先全局观察更新位置
                if retry < max_retries - 1:
                    print(f"  -> 重试前先全局观察更新位置...")
                    fallback_data = observe_from_global_view(robot, camera_manager)
                    
                    if fallback_data:
                        fallback_processed = preprocess_yolo_angles(
                            fallback_data, YOLO_CORRECTIONS,
                            observation_rz_deg=GLOBAL_OBSERVATION_CONFIG['quat'][2]
                        )
                        
                        # 更新粗略位置
                        rough_positions = {}
                        for bid, data in fallback_processed.items():
                            rough_positions[bid] = data[0]
                        
                        # 重新选择局部观察位置
                        same_type_positions = {}
                        for bid, pos in rough_positions.items():
                            if bid.startswith(yolo_prefix):
                                same_type_positions[bid] = pos
                        
                        if same_type_positions:
                            max_y_bid = max(same_type_positions.items(), key=lambda x: x[1][1])
                            rough_pos = max_y_bid[1].copy()
                        
                        print(f"  -> 已更新位置，准备重试")
            
            # ============ 步骤2.4: 检查局部观察结果 ============
            if not local_success or not processed_local_data:
                print(f"  ✗ 局部观察最终失败，跳过积木 {block_id}")
                continue
            
            # ============ 步骤2.5: 计算目标角度（全局锁定）============
            # 从局部观察数据中找到目标积木并计算角度
            target_detected_angle_rad = None
            target_yolo_id = None
            
            # ============ 修正：按优先级选择目标积木（先左后右，从上到下）============
            world_candidates = []
            for cand_id, data in processed_local_data.items():
                if cand_id.startswith(yolo_prefix):
                    pos_x, pos_y, pos_z = data[0][:3]
                    angle_rad = data[1]
                    world_candidates.append((cand_id, pos_x, pos_y, pos_z, angle_rad))
            
            if world_candidates:
                print(f"  -> [角度锁定] 局部观察候选分析:")
                
                # 显示所有候选
                for cand_id, pos_x, pos_y, pos_z, angle_rad in world_candidates:
                    print(f"     {cand_id}: 世界({pos_x:.3f}, {pos_y:.3f}), 角度{np.degrees(angle_rad):.1f}°")
                
                # ============ 使用相同的选择策略：先左后右，从上到下 ============
                # 第1步：按Y坐标排序（左→右：Y递减，选Y最大）
                world_candidates.sort(key=lambda x: x[2], reverse=True)  # 按Y坐标降序排序
                
                # 第2步：在Y相近的候选中，选择X最大的（上→下：X递减，选X最大）
                max_y = world_candidates[0][2]  # 最大的Y坐标
                y_tolerance = 0.010  # 10mm容差
                
                y_similar_candidates = []
                for cand_id, pos_x, pos_y, pos_z, angle_rad in world_candidates:
                    if abs(pos_y - max_y) <= y_tolerance:
                        y_similar_candidates.append((cand_id, pos_x, pos_y, pos_z, angle_rad))
                
                # 在Y相近的候选中，选择X最大的
                y_similar_candidates.sort(key=lambda x: x[1], reverse=True)  # 按X坐标降序排序
                target_yolo_id, _, _, _, target_detected_angle_rad = y_similar_candidates[0]
                
                # 显示选择过程
                print(f"  -> [选择策略] 先左后右，从上到下:")
                print(f"               Y最大={max_y:.3f} (最左边), 容差±{y_tolerance*1000:.0f}mm")
                print(f"               Y相近候选: {len(y_similar_candidates)}个")
                
                for i, (cand_id, pos_x, pos_y, pos_z, angle_rad) in enumerate(world_candidates):
                    if cand_id == target_yolo_id:
                        marker = "★"
                        extra_info = f" ← 选中 (最左且最上)"
                    elif abs(pos_y - max_y) <= y_tolerance:
                        marker = "◦"
                        extra_info = f" (Y相近)"
                    else:
                        marker = " "
                        extra_info = ""
                    
                    print(f"     {marker} {cand_id}: 世界({pos_x:.3f}, {pos_y:.3f}), 角度{np.degrees(angle_rad):.1f}°{extra_info}")
                
                print(f"  -> [全局角度锁定] 目标积木: {target_yolo_id}")
                print(f"                   检测角度: {np.degrees(target_detected_angle_rad):.1f}°")
            else:
                print(f"  ✗ 未找到类型 {target_type} 的积木")
                continue
                
            # 计算夹爪角度（全局锁定，不再改变）
            end_pose_msg = robot.GetArmEndPoseMsgs()
            current_rz_deg = end_pose_msg.end_pose.RZ_axis / 1000.0
            target_gripper_angle_rad, target_gripper_angle_deg = calculate_gripper_angle(
                target_detected_angle_rad, current_rz_deg
            )
            
            print(f"  -> [全局角度计算] 夹爪目标角度: {target_gripper_angle_deg:.1f}°")
            
            # ============ 步骤2.6: 选择最佳候选并执行 ============
            selected_yolo_id, selected_data, gripper_angle_rad = select_best_candidate(
                processed_local_data, yolo_prefix, robot, camera_manager,
                target_gripper_angle_rad,  # 传入预先计算好的角度
                enable_refinement=True,
                enable_grasp_offset=True,
                grasp_offset_mm=70.0
            )
            
            if not selected_yolo_id:
                print(f"  ✗ 候选选择失败")
                continue
            
            print(f"  ✓ 绑定: '{block_id}' <- '{selected_yolo_id}'")
            print(f"  -> 最终夹爪角度: {np.degrees(gripper_angle_rad):.1f}°")
            
            # ============ 步骤2.7: 更新调度器并执行任务 ============
            pos = selected_data[0]
            current_z = scheduler.instances[block_id]["initial_pos"][2]
            update_dict = {block_id: [[pos[0], pos[1], current_z], gripper_angle_rad]}
            scheduler.update_initial_states_from_dict(update_dict)

            # 生成并执行任务
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

        # ============ 完成所有任务 ============
        print("\n" + "="*60)
        print("--- 所有任务完成 ---")
        print("="*60)
        
        total_time = time.time() - global_start_time
        print(f"总耗时: {total_time:.2f}s")
        print(f"平均每个积木: {total_time/len(build_order):.2f}s")

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
