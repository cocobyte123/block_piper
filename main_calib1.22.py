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
# 手眼标定矩阵路径
HAND_EYE_MATRIX_PATH = "hand_eye_transform.npy"

# 全局观察点位置
GLOBAL_OBSERVATION_CONFIG = {
    'pos': [0.105, 0.05, 0.32],
    'quat': [-160.0, 0.0, -90.0]
}

# 像素中心容差（宽裕范围，单位：像素）
PIXEL_CENTER_TOLERANCE = 20  # 从中心±80像素内都可接受

# 预抓取高度偏移（单位：米）
PRE_GRASP_Z_OFFSET = 0.100  # 在积木上方100mm


# ================= 相机可视化线程 =================
class CameraVisualizationThread(threading.Thread):
    """独立线程显示相机画面和YOLO检测结果"""
    def __init__(self, detector):
        super().__init__(daemon=True)
        self.detector = detector
        self.running = True
        self.show_detections = True
        
    def draw_detection_results(self, image, detection_data):
        """在图像上绘制YOLO检测结果"""
        if not detection_data:
            return image
        
        type_colors = {
            'code1': (0, 255, 0),
            'code2': (255, 0, 0),
            'code3': (0, 0, 255),
            'code4': (0, 255, 255),
        }
        
        overlay = image.copy()
        
        for block_id, data in detection_data.items():
            try:
                world_pos = data[0][:3]
                angle_rad = data[1]
                pixel_pos = data[2] if len(data) >= 3 and data[2] is not None else None
                
                color = (128, 128, 128)
                for prefix, type_color in type_colors.items():
                    if block_id.startswith(prefix):
                        color = type_color
                        break
                
                if pixel_pos:
                    cx, cy = int(pixel_pos[0]), int(pixel_pos[1])
                    w, h = 80, 80
                    angle_degrees = np.degrees(angle_rad)
                    
                    rect = ((cx, cy), (w, h), angle_degrees)
                    box = cv2.boxPoints(rect).astype(int)
                    cv2.drawContours(overlay, [box], 0, color, 2)
                    
                    cv2.circle(overlay, (cx, cy), 5, (0, 0, 255), -1)
                    
                    arrow_length = min(w, h) / 2
                    end_x = int(cx + arrow_length * np.cos(angle_rad))
                    end_y = int(cy + arrow_length * np.sin(angle_rad))
                    cv2.arrowedLine(overlay, (cx, cy), (end_x, end_y), 
                                (255, 0, 0), 3, tipLength=0.3)
                    
                    label = f"{block_id}"
                    cv2.putText(overlay, label, (cx - w//2, cy - h//2 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    coord_text = f"World: ({world_pos[0]:.3f}, {world_pos[1]:.3f})"
                    cv2.putText(overlay, coord_text, (cx - w//2, cy + h//2 + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
            except Exception as e:
                print(f"绘制检测结果失败 {block_id}: {e}")
                continue
        
        h, w = image.shape[:2]
        center_x, center_y = w // 2, h // 2
        cv2.circle(overlay, (center_x, center_y), PIXEL_CENTER_TOLERANCE, (0, 255, 255), 2)
        cv2.line(overlay, (center_x-20, center_y), (center_x+20, center_y), (255, 255, 255), 2)
        cv2.line(overlay, (center_x, center_y-20), (center_x, center_y+20), (255, 255, 255), 2)
        
        status_text = f"Detections: {len(detection_data)} | Tolerance: {PIXEL_CENTER_TOLERANCE}px"
        cv2.putText(overlay, status_text, (10, h - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return overlay
    
    def run(self):
        cv2.namedWindow('Hand-Eye Calibration View', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Hand-Eye Calibration View', 1280, 720)
        
        print("📹 相机可视化启动!")
        
        while self.running:
            try:
                color_frame, _ = self.detector.get_frames()
                if not color_frame:
                    time.sleep(0.01)
                    continue
                
                color_image = np.asanyarray(color_frame.get_data())
                
                if self.show_detections:
                    try:
                        detection_result = self.detector.run_single_detection()
                        if detection_result:
                            color_image = self.draw_detection_results(color_image, detection_result)
                    except Exception as e:
                        cv2.putText(color_image, f"Detection Error: {str(e)[:50]}", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                cv2.imshow('Hand-Eye Calibration View', color_image)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    break
                    
            except Exception as e:
                print(f"可视化线程错误: {e}")
                time.sleep(0.1)
        
        cv2.destroyAllWindows()
    
    def stop(self):
        self.running = False


# ================= 相机管理类 =================
class CameraManager:
    """相机管理类"""
    def __init__(self, enable_visualization=True):
        self.detector = DetectionSystem()
        self.enable_visualization = enable_visualization
        self.viz_thread = None
        
        if enable_visualization:
            self.viz_thread = CameraVisualizationThread(self.detector)
            self.viz_thread.start()
        
        print("✓ 相机系统已初始化")
    
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
    
    def get_stable_detection(self, num_samples=2, skip_frames=2, sample_interval=0.02):
        """获取稳定的检测结果"""
        print(f"  [稳定性检测] 跳过{skip_frames}帧，采样{num_samples}次...")
        
        for i in range(skip_frames):
            self.detector.run_single_detection()
            time.sleep(0.02)
        
        samples = []
        for i in range(num_samples):
            detection = self.detector.run_single_detection()
            if detection:
                samples.append(detection)
            if i < num_samples - 1:
                time.sleep(sample_interval)
        
        if len(samples) < 2:
            return samples[0] if samples else None
        
        # 简单平均处理
        return samples[-1]  # 返回最后一次检测结果
    
    def close(self):
        """关闭相机"""
        if self.viz_thread:
            self.viz_thread.stop()
            self.viz_thread.join(timeout=2)
        if self.detector:
            self.detector.close()


# ================= 核心功能函数 =================
def load_hand_eye_matrix():
    """加载手眼标定矩阵"""
    try:
        T_cam2gripper = np.load(HAND_EYE_MATRIX_PATH)
        print("✓ 成功加载手眼标定矩阵")
        print(f"  矩阵:\n{T_cam2gripper}")
        return T_cam2gripper
    except FileNotFoundError:
        print(f"✗ 错误: 未找到手眼标定矩阵文件 {HAND_EYE_MATRIX_PATH}")
        return None


def calculate_gripper_angle(detected_angle_rad, current_rz_deg):
    """计算夹爪旋转角度"""
    detected_angle_deg = np.degrees(detected_angle_rad)
    
    if detected_angle_deg > 90.0:
        target_rz_deg = -90.0 - detected_angle_deg + 180
    else:
        target_rz_deg = -90.0 - detected_angle_deg
    
    while target_rz_deg > 180:
        target_rz_deg -= 360.0
    while target_rz_deg < -180.0:
        target_rz_deg += 360.0
    
    if 60.0 <= target_rz_deg <= 80.0:
        target_rz_deg = target_rz_deg - 180.0
    
    gripper_angle_rad = np.radians(target_rz_deg)
    
    return gripper_angle_rad, target_rz_deg


def move_to_observation_point(robot, pos, quat, speed=50):
    """移动到观察点"""
    piper_pos = [int(round(p * 1000000)) for p in pos]
    piper_euler = [int(round(e * 1000)) for e in quat]
    
    robot.MotionCtrl_2(0x01, 0x00, speed, 0x00)
    robot.EndPoseCtrl(piper_pos[0], piper_pos[1], piper_pos[2], 
                      piper_euler[0], piper_euler[1], piper_euler[2])
    time.sleep(1.2)
    
    return True


def observe_from_global_view(robot, camera_manager):
    """全局观察"""
    print("\n  -> 【全局观察】移动到全局视角...")
    success = move_to_observation_point(
        robot, 
        GLOBAL_OBSERVATION_CONFIG['pos'], 
        GLOBAL_OBSERVATION_CONFIG['quat']
    )
    
    if not success:
        return None
    
    camera_manager.update_robot_pose(robot)
    time.sleep(0.2)
    
    return camera_manager.get_stable_detection(num_samples=1)


def check_pixel_in_center(pixel_pos, img_center_x=320, img_center_y=240):
    """检查像素是否在中心容差范围内"""
    if pixel_pos is None:
        return False
    
    px, py = pixel_pos
    distance = np.sqrt((px - img_center_x)**2 + (py - img_center_y)**2)
    
    return distance <= PIXEL_CENTER_TOLERANCE


# ================= PID控制器类 =================
class PixelPIDController:
    """像素对中的PID控制器"""
    def __init__(self, kp=0.08, ki=0.0, kd=0.01):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error_x = 0.0
        self.prev_error_y = 0.0
        self.integral_x = 0.0
        self.integral_y = 0.0
    
    def compute(self, error_x, error_y, dt=1.0):
        """计算PID输出"""
        self.integral_x += error_x * dt
        self.integral_y += error_y * dt
        
        derivative_x = (error_x - self.prev_error_x) / dt
        derivative_y = (error_y - self.prev_error_y) / dt
        
        output_px_x = (self.kp * error_x + self.ki * self.integral_x + self.kd * derivative_x)
        output_px_y = (self.kp * error_y + self.ki * self.integral_y + self.kd * derivative_y)
        
        self.prev_error_x = error_x
        self.prev_error_y = error_y
        
        # 使用固定比例尺转换
        ratio_mm_per_px = 0.67
        move_x_mm = output_px_y * ratio_mm_per_px
        move_y_mm = output_px_x * ratio_mm_per_px
        
        return move_x_mm, move_y_mm
    
    def reset(self):
        self.prev_error_x = 0.0
        self.prev_error_y = 0.0
        self.integral_x = 0.0
        self.integral_y = 0.0


def refine_position_to_center(robot, camera_manager, current_pos, initial_pixel_pos,
                             target_yolo_prefix, img_center_x=320, img_center_y=240,
                             max_iterations=5, tolerance_pixels=PIXEL_CENTER_TOLERANCE):
    """使用直接比例控制将积木快速移动到相机中心容差范围内（参考main.py）"""
    print(f"\n  -> 【快速对中】将积木移动到相机视野中心...")
    print(f"     初始像素: ({initial_pixel_pos[0]:.1f}, {initial_pixel_pos[1]:.1f})")
    print(f"     目标容差: ±{tolerance_pixels}px")
    
    end_pose_msg = robot.GetArmEndPoseMsgs()
    current_euler = [
        end_pose_msg.end_pose.RX_axis,
        end_pose_msg.end_pose.RY_axis,
        end_pose_msg.end_pose.RZ_axis
    ]
    
    # 固定映射参数（参考main.py）
    direction_x = +1
    direction_y = +1
    ratio_mm_per_px = 0.67
    
    refined_pos = np.array(current_pos)
    px, py = initial_pixel_pos
    
    for iteration in range(1, max_iterations + 1):
        offset_distance = np.sqrt((px - img_center_x)**2 + (py - img_center_y)**2)
        
        print(f"\n     [迭代 {iteration}/{max_iterations}] 像素: ({px:.1f}, {py:.1f}), 距中心: {offset_distance:.1f}px")
        
        if offset_distance <= tolerance_pixels:
            print(f"       ✓ 已进入容差范围")
            break
        
        # 计算移动量（直接比例控制，更快速）
        error_px = img_center_x - px
        error_py = img_center_y - py
        
        move_x_mm = direction_y * error_py * ratio_mm_per_px
        move_y_mm = direction_x * error_px * ratio_mm_per_px
        
        # 限制移动量
        max_move = 100.0
        move_x_mm = max(-max_move, min(max_move, move_x_mm))
        move_y_mm = max(-max_move, min(max_move, move_y_mm))
        
        print(f"       移动: ΔX={move_x_mm:.2f}mm, ΔY={move_y_mm:.2f}mm")
        
        # 执行移动
        refined_pos[0] += move_x_mm / 1000.0
        refined_pos[1] += move_y_mm / 1000.0
        
        piper_pos = [int(round(p * 1000000)) for p in refined_pos]
        robot.MotionCtrl_2(0x01, 0x00, 30, 0x00)
        robot.EndPoseCtrl(piper_pos[0], piper_pos[1], piper_pos[2],
                         current_euler[0], current_euler[1], current_euler[2])
        time.sleep(0.1)
        
        # 重新检测
        camera_manager.update_robot_pose(robot)
        time.sleep(0.1)
        detection = camera_manager.get_stable_detection(num_samples=1)
        
        if not detection:
            print(f"       ✗ 检测失败")
            break
        
        # 使用一致的策略（先左后右，从上到下）重新锁定目标
        # 即使在对中过程中，也必须坚持使用同样的排序逻辑，防止锁定到错误的积木（例如下方的积木）
        found = False
        target_id, target_data = select_best_candidate_simple(detection, target_yolo_prefix)
        
        if target_id and target_data:
             found = True
             px, py = target_data[2]
             print(f"       >>> 重新锁定目标: {target_id} 像素({px:.1f}, {py:.1f})")
        else:
             print(f"       ✗ 丢失目标积木 (筛选后无结果)")
             break
    
    final_offset = np.sqrt((px - img_center_x)**2 + (py - img_center_y)**2)
    print(f"\n  ✓ 快速对中完成，最终距离中心: {final_offset:.1f}px")
    
    # 如果循环中找到了目标，latest_match_data 将包含最后一次检测的数据
    # 如果一开始就在容差内（没进循环或第一次check就break且没检测），则为None
    return refined_pos, (px, py), target_data if 'target_data' in locals() else None


def select_best_candidate_simple(yolo_data, yolo_prefix):
    """选择积木：基于像素坐标，先左后右，从上到下"""
    print("\n  >>> DEBUG: 进入选择逻辑 (select_best_candidate_simple) <<<")
    candidates = []
    
    # Debug: 打印原始输入
    # print(f"  [DEBUG] 原始YOLO数据 keys: {list(yolo_data.keys())}")

    for block_id, data in yolo_data.items():
        if block_id.startswith(yolo_prefix):
            # data结构预期: [world_pos, angle, pixel_pos]
            world_pos = data[0][:3]
            pixel_pos = data[2] if len(data) >= 3 and data[2] is not None else None
            
            if pixel_pos is not None:
                candidates.append((block_id, pixel_pos[0], pixel_pos[1], world_pos))
            else:
                print(f"  [警告] {block_id} 缺少像素坐标，跳过")
    
    if not candidates:
        print("  [警告] 没有找到符合前缀的积木")
        return None, None
    
    # 显示所有候选（未排序）
    print(f"  -> 找到 {len(candidates)} 个候选 {yolo_prefix}:")
    for cand_id, px, py, world_pos in candidates:
        print(f"     [Original] {cand_id}: Pixel(X={px:.1f}, Y={py:.1f}), World(X={world_pos[0]:.3f})")
    
    # ---------------------------------------------------------
    # 第1步：按像素X坐标排序（先左后右：X最小优先）
    # ---------------------------------------------------------
    # 图像坐标系：X=0在左边，X变大向右
    candidates.sort(key=lambda x: x[1])  # X升序
    
    print(f"  -> 按X排序后（左->右）:")
    for cand_id, px, py, world_pos in candidates:
        print(f"     {cand_id}: X={px:.1f}")

    # ---------------------------------------------------------
    # 第2步：筛选最左边的一列（X容差范围内）
    # ---------------------------------------------------------
    if not candidates: return None, None
        
    min_x = candidates[0][1]  # 最小的像素X坐标（最左边）
    x_tolerance = 50  # 像素X容差：50像素
    
    x_similar_candidates = []
    for cand_id, px, py, world_pos in candidates:
        if abs(px - min_x) <= x_tolerance:
            x_similar_candidates.append((cand_id, px, py, world_pos))
            
    print(f"  -> 筛选最左列 (MinX={min_x:.1f} ±{x_tolerance}px): 选中 {len(x_similar_candidates)} 个")

    # ---------------------------------------------------------
    # 第3步：按像素Y坐标排序（从上到下：Y最小优先）
    # ---------------------------------------------------------
    # 图像坐标系：Y=0在上边，Y变大向下
    # 所以要选“上”面的，就是选Y小的。升序排列。
    x_similar_candidates.sort(key=lambda x: x[2])  # Y升序
    
    selected_candidate = x_similar_candidates[0]
    best_candidate = selected_candidate[0]
    
    # 显示最终排序结果
    print(f"  -> [最终排序] 左列中按Y升序（上->下）:")
    for cand_id, px, py, world_pos in x_similar_candidates:
        mark = "★ SELECT" if cand_id == best_candidate else "         "
        print(f"     {mark} {cand_id}: Y={py:.1f} (X={px:.1f})")
    
    print(f"  >>> 最终决定抓取: {best_candidate} <<<\n")
    
    return best_candidate, yolo_data[best_candidate]


def main():
    """基于手眼标定的主流程"""
    np.set_printoptions(precision=6, suppress=True)
    
    # 配置
    TYPE_TO_YOLO_PREFIX = {
        "type1": "code1", "type2": "code2", "type3": "code3", "type4": "code4"
    }
    
    # 加载手眼标定矩阵
    T_cam2gripper = load_hand_eye_matrix()
    if T_cam2gripper is None:
        print("✗ 无法加载手眼标定矩阵，程序退出")
        return
    
    # 初始化硬件
    CAN_IFACE = "can0"
    robot = C_PiperInterface_V2(CAN_IFACE)
    robot.ConnectPort()
    while not robot.EnablePiper():
        time.sleep(0.5)
    
    camera_manager = None
    
    try:
        camera_manager = CameraManager(enable_visualization=True)
        
        executor = CommandExecutor(robot)
        scheduler = TaskScheduler(first_block_target_pos=[0.1200, -0.200, 0.125])
        
        build_order = (
            scheduler.architecture["layer_1"] + scheduler.architecture["layer_2"] +
            scheduler.architecture["layer_3"] + scheduler.architecture["layer_4"]
        )
        
        print("\n" + "="*60)
        print("=== 基于手眼标定的积木抓取系统 (V1.23_DEBUG_SORTING) ===")
        print("="*60)
        
        # 打开夹爪
        robot.GripperCtrl(gripper_angle=80000, gripper_effort=1000, gripper_code=0x01, set_zero=0x00)
        
        global_start_time = time.time()
        
        for task_idx, block_id in enumerate(build_order):
            print(f"\n{'='*60}")
            print(f">>> [{task_idx + 1}/{len(build_order)}] 处理积木: {block_id}")
            print(f"{'='*60}")
            
            # 全局观察
            yolo_data = observe_from_global_view(robot, camera_manager)
            
            if not yolo_data:
                print("✗ 全局观察失败")
                continue
            
            # 选择目标积木
            target_type = scheduler.instances[block_id]["type"]
            yolo_prefix = TYPE_TO_YOLO_PREFIX.get(target_type)
            
            selected_id, selected_data = select_best_candidate_simple(yolo_data, yolo_prefix)
            
            if not selected_id:
                print(f"✗ 未找到类型 {target_type} 的积木")
                continue
            
            print(f"  ✓ 选择积木: {selected_id}")
            
            # 提取积木位置和角度
            world_pos = selected_data[0][:3]
            angle_rad = selected_data[1]
            pixel_pos = selected_data[2] if len(selected_data) >= 3 else None
            
            print(f"\n  -> 【检测结果】")
            print(f"     积木位置: [{world_pos[0]:.6f}, {world_pos[1]:.6f}, {world_pos[2]:.6f}]")
            print(f"     检测角度: {np.degrees(angle_rad):.1f}°")
            
            # 检查像素位置
            need_refinement = False
            if pixel_pos:
                pixel_distance = np.sqrt((pixel_pos[0] - 320)**2 + (pixel_pos[1] - 240)**2)
                print(f"     像素位置: ({pixel_pos[0]:.1f}, {pixel_pos[1]:.1f})")
                print(f"     距中心: {pixel_distance:.1f}px (容差: {PIXEL_CENTER_TOLERANCE}px)")
                
                is_centered = check_pixel_in_center(pixel_pos)
                if is_centered:
                    print(f"     ✓ 在容差范围内，直接使用手眼标定")
                else:
                    print(f"     ✗ 超出容差范围，需要PID对中")
                    need_refinement = True
            
            # 如果需要对中，先执行PID对中
            final_world_pos = world_pos.copy()
            if need_refinement:
                end_pose_msg = robot.GetArmEndPoseMsgs()
                current_pos = [
                    end_pose_msg.end_pose.X_axis / 1000000.0,
                    end_pose_msg.end_pose.Y_axis / 1000000.0,
                    end_pose_msg.end_pose.Z_axis / 1000000.0
                ]
                
                # PID对中
                refined_pos, final_pixel, last_detection_data = refine_position_to_center(
                    robot, camera_manager, current_pos, pixel_pos,
                    yolo_prefix, max_iterations=5, tolerance_pixels=PIXEL_CENTER_TOLERANCE
                )
                
                # 【精度修正】PID由于是在移动中检测，坐标可能存在微小偏差。
                # 为了保证抓取精度，我们在对中完成后，做一个快速但稳定的最终确认。
                # 优化耗时：仅等待0.15秒让机械臂停稳，然后采样1次。
                
                print(f"  -> [执行最终确认] 等待停稳(0.15s)...")
                time.sleep(0.15) 
                camera_manager.update_robot_pose(robot)
                
                # 获取一次稳定检测
                final_detection = camera_manager.get_stable_detection(num_samples=1)
                
                if final_detection:
                    # 重新锁定目标 (使用相同的逻辑)
                    confirm_id, confirm_data = select_best_candidate_simple(final_detection, yolo_prefix)
                    
                    if confirm_id:
                        final_world_pos = confirm_data[0][:3]
                        angle_rad = confirm_data[1]
                        print(f"     ✓ 确认目标: {confirm_id}")
                        print(f"     最终位置: [{final_world_pos[0]:.6f}, {final_world_pos[1]:.6f}, {final_world_pos[2]:.6f}]")
                        print(f"     最终角度: {np.degrees(angle_rad):.1f}°")
                    elif last_detection_data:
                        # 如果最终确认没找到（比如被遮挡），回退使用PID过程中的数据
                        print(f"     ! 最终确认未找到目标，回退使用PID数据")
                        final_world_pos = last_detection_data[0][:3]
                        angle_rad = last_detection_data[1]
                elif last_detection_data:
                     print(f"     ! 最终检测失败，回退使用PID数据")
                     final_world_pos = last_detection_data[0][:3]
                     angle_rad = last_detection_data[1]

            # 计算夹爪角度
            end_pose_msg = robot.GetArmEndPoseMsgs()
            current_rz_deg = end_pose_msg.end_pose.RZ_axis / 1000.0
            gripper_angle_rad, gripper_angle_deg = calculate_gripper_angle(angle_rad, current_rz_deg)
            print(f"     计算的夹爪角度: {gripper_angle_deg:.1f}°")
            
            # 使用手眼标定得到的精确位姿更新调度器
            update_dict = {
                block_id: [[final_world_pos[0]-0.025, final_world_pos[1], final_world_pos[2]], gripper_angle_rad]
            }
            scheduler.update_initial_states_from_dict(update_dict)
            
            # 执行抓取放置
            current_task = scheduler.get_task_for_block(
                block_id, build_order, executor.last_projected_grasp_error
            )
            
            if current_task:
                print(f"\n  === 执行抓取放置 ===")
                executor.execute_task(current_task)
                scheduler.update_placement_error(block_id, executor.last_place_error_x)
                
                elapsed = time.time() - global_start_time
                print(f"  ✓ 完成 {block_id}，累计耗时: {elapsed:.2f}s")
            else:
                print(f"  ✗ 无法生成任务")
        
        print("\n" + "="*60)
        print("=== 所有任务完成 ===")
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