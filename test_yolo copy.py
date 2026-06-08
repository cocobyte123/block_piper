import pyrealsense2 as rs
import cv2
import numpy as np
from ultralytics import YOLO
import time
import yaml
import math
import json
import os 
# from airbot_py.arm import AIRBOTPlay  # <--- 移除机械臂SDK导入
from scipy.spatial.transform import Rotation
from piper_sdk import C_PiperInterface_V2


def get_z_axis_rotation_angle(quaternion, degrees=True):
    """
    从四元数中提取绕基坐标系Z轴的旋转角度。
    
    Args:
        quaternion: [qx, qy, qz, qw] 列表或数组
        degrees: True 返回度数，False 返回弧度
    
    Returns:
        float: 绕Z轴的旋转角度
    """
    # 创建Rotation对象
    rot = Rotation.from_quat(quaternion)  # 输入格式 [x, y, z, w]
    
    # 提取欧拉角（顺序 'xyz'），yaw 是第三个元素
    euler_angles = rot.as_euler('xyz', degrees=degrees)
    yaw_angle = euler_angles[2]  # yaw（绕Z轴）
    
    return yaw_angle

class DetectionSystem:
    def __init__(self, config_path="config/camera.yaml"):
        # 加载配置
        self.config_path = config_path
        self.load_config()
        
        # 初始化模型
        self.model = YOLO(self.model_path)

        # 初始化相机
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        self.setup_camera()
        self.start_camera()
        
        # 坐标变换参数
        self.extrinsics_matrix = np.array(self.camera_extrinsics['matrix'])
        self.q_eef_cam = self.rotation_matrix_to_quaternion(self.extrinsics_matrix[:3, :3])  # 相机到末端的姿态四元数
        
        # 机器人位姿参数
        self.robot_position = None
        self.robot_quaternion = None
        
        print("检测系统初始化完成")
    
    def close(self):
        """关闭相机管道"""
        print("正在关闭相机...")
        self.pipeline.stop()
        print("相机已关闭。")

    def run_single_detection(self):
        """
        执行单次检测并返回结果，不显示UI。
        """
        print("正在执行单次检测...")
        color_frame, depth_frame = self.get_frames()
        if not color_frame or not depth_frame:
            print("获取帧失败,无法执行检测。")
            return None
        
        color_image = np.asanyarray(color_frame.get_data())
        
        if self.robot_position is None or self.robot_quaternion is None:
            print("错误：机器人位姿未设置，无法计算基座坐标。")
            return None
            
        _, detections = self.process_frame(
            color_image.copy(), 
            depth_frame, 
            self.robot_position, 
            self.robot_quaternion
        )
        
        # 【关键修复】转换为desktop_data格式，并添加像素坐标
        desktop_data = {}
        desktop_z = 0.014
        
        if detections:
            sorted_detections = sorted(detections, key=lambda d: (d['class_id'], d['base_coords'][0]))
            category_counters = {1: 1, 2: 1, 3: 1, 4: 1}
            
            for det in sorted_detections:
                cat_id = det['class_id'] + 1
                if cat_id not in category_counters:
                    continue
                
                instance_num = category_counters[cat_id]
                block_name = f"code{cat_id}_{instance_num}"
                if cat_id == 4:
                    block_name = "code4"
                
                coords = det['base_coords'].tolist()
                quat = det['quaternion']
                angle_rad = Rotation.from_quat(quat).as_euler('xyz', degrees=False)[2]
                
                # 【核心修复】直接从 det 中获取像素坐标
                pixel_center = det['pixel_coords']  # (cx, cy) 元组
                
                # 格式：[[x, y, z], angle_rad, pixel_center]
                desktop_data[block_name] = [[coords[0], coords[1], desktop_z], angle_rad, pixel_center]
                
                category_counters[cat_id] += 1
        
        print(f"检测完成，发现 {len(desktop_data)} 个物体。")
        return desktop_data

    def _get_block_name_from_detection(self, det, category_counters):
        """辅助函数：从detection生成block_name"""
        cat_id = det['class_id'] + 1
        if cat_id == 4:
            return "code4"
        # 这里需要维护一个计数逻辑，或者直接在主循环中处理
        # 简化版本：返回None，让主循环处理
        return None


    def load_config(self):
        """加载配置文件"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        # 基本参数
        self.conf_threshold = config_data['detection']['confidence_threshold']
        self.model_path = config_data['detection']['model_path']
        self.window_name = config_data['display']['window_name']
        
        # 相机参数
        self.camera_extrinsics = config_data['camera_extrinsics']
        self.color_stream = config_data['camera_streams']['color']
        self.depth_stream = config_data['camera_streams']['depth']
        
        # 深度处理参数
        self.depth_kernel_size = config_data['depth_processing']['kernel_size']
        
        print(f"加载配置文件: {self.config_path}")
        print(f"置信度阈值: {self.conf_threshold}")
        print(f"模型路径: {self.model_path}")
    
    def setup_camera(self):
        """配置相机流"""
        # 配置彩色流
        self.config.enable_stream(
            rs.stream.color,
            self.color_stream['width'],
            self.color_stream['height'],
            getattr(rs.format, self.color_stream['format']),
            self.color_stream['fps']
        )
        
        # 配置深度流
        self.config.enable_stream(
            rs.stream.depth,
            self.depth_stream['width'],
            self.depth_stream['height'],
            getattr(rs.format, self.depth_stream['format']),
            self.depth_stream['fps']
        )
    
    def start_camera(self):
        """启动相机"""
        self.profile = self.pipeline.start(self.config)
        
        # 获取深度传感器参数
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        
        # 获取相机内参
        color_profile = rs.video_stream_profile(self.profile.get_stream(rs.stream.color))
        self.color_intrinsics = color_profile.get_intrinsics()

        print("信息：使用默认相机内参:")
        print(f"  fx: {self.color_intrinsics.fx}, fy: {self.color_intrinsics.fy}")
        print(f"  ppx: {self.color_intrinsics.ppx}, ppy: {self.color_intrinsics.ppy}")
        print(123)
        # self.color_intrinsics.fx = 611.70375776
        # self.color_intrinsics.fy = 611.72122191
        # self.color_intrinsics.ppx = 331.62218223
        # self.color_intrinsics.ppy = 261.36506807
        
        # print("信息：已替换为手动设置的相机内参:")
        # print(f"  fx: {self.color_intrinsics.fx}, fy: {self.color_intrinsics.fy}")
        # print(f"  ppx: {self.color_intrinsics.ppx}, ppy: {self.color_intrinsics.ppy}")
        
        # 配置对齐器
        self.align = rs.align(rs.stream.color)
        
        print("相机启动成功")
    
    def get_frames(self):
        """获取对齐后的帧"""
        try:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            return aligned_frames.get_color_frame(), aligned_frames.get_depth_frame()
        except Exception as e:
            print(f"获取帧失败: {e}")
            return None, None
    
    def set_robot_pose(self, position, quaternion):
        """设置机器人当前位置和姿态（四元数）"""
        self.robot_position = position  # [x, y, z]
        self.robot_quaternion = quaternion  # [qx, qy, qz, qw]
        print(f"机器人位置设置: {position}")
        print(f"机器人姿态设置: {quaternion}")
    
    def obb_angle_to_standard(self, angle_degrees, w, h):
        """将OBB角度转换为标准角度"""
        if w > h:
            standard_angle = angle_degrees + 90
        else:
            standard_angle = angle_degrees + 180
        return standard_angle % 180
    
    def get_average_depth(self, depth_frame, cx, cy, w, h):
        """获取检测框区域的平均深度值"""
        # 将深度帧转换为numpy数组
        depth_image = np.asanyarray(depth_frame.get_data())
        
        # 计算检测框区域
        x_start = int(max(0, cx - w/2))
        x_end = int(min(depth_image.shape[1], cx + w/2))
        y_start = int(max(0, cy - h/2))
        y_end = int(min(depth_image.shape[0], cy + h/2))
        
        # 获取区域内的深度值
        depth_region = depth_image[y_start:y_end, x_start:x_end]
        valid_depths = depth_region[depth_region > 0]
        
        if len(valid_depths) > 0:
            # 计算平均深度并转换为米
            avg_depth_raw = np.mean(valid_depths)
            return avg_depth_raw * self.depth_scale
        else:
            return 0.0
    
    def pixel2cam(self, pixel_x, pixel_y, depth):
        """像素坐标转换为相机坐标系"""
        # 归一化像素坐标
        x_norm = (pixel_x - self.color_intrinsics.ppx) / self.color_intrinsics.fx
        y_norm = (pixel_y - self.color_intrinsics.ppy) / self.color_intrinsics.fy
        
        # 计算相机坐标系下的坐标
        X = depth * x_norm
        Y = depth * y_norm
        Z = depth
        
        return np.array([X, Y, Z])
    
    def rotation_matrix_to_quaternion(self, R):
        """从旋转矩阵计算四元数 [qx, qy, qz, qw]"""
        qw = 0.5 * np.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2])
        qx = (R[2, 1] - R[1, 2]) / (4 * qw)
        qy = (R[0, 2] - R[2, 0]) / (4 * qw)
        qz = (R[1, 0] - R[0, 1]) / (4 * qw)
        return [qx, qy, qz, qw]

    def multiply_quaternions(self, q1, q2):
        """
        执行四元数乘法 q_final = q1 * q2.
        格式: [qx, qy, qz, qw]
        """
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        
        w_new = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x_new = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y_new = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z_new = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        
        return [x_new, y_new, z_new, w_new]
    
    def quaternion_to_rotation_matrix(self, q):
        """四元数转换为旋转矩阵"""
        qx, qy, qz, qw = q
        
        # 计算旋转矩阵
        R = np.array([
            [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy]
        ])
        
        return R
    
    def cam2base(self, cam_x, cam_y, cam_z, robot_position, robot_quaternion):
        """相机坐标系到基座坐标系的转换"""
        if robot_position is None or robot_quaternion is None:
            raise ValueError("机器人位置和姿态未设置")
        
        # 相机坐标系下的点
        cam_point = np.array([cam_x, cam_y, cam_z])
        
        # 使用外部标定矩阵转换到机器人基座坐标系
        cam_point_homogeneous = np.append(cam_point, 1.0)
        base_point_homogeneous = self.extrinsics_matrix @ cam_point_homogeneous
        base_point = base_point_homogeneous[:3]
        
        # 获取机器人旋转矩阵
        R_robot = self.quaternion_to_rotation_matrix(robot_quaternion)
        
        # 考虑机器人姿态的变换
        transformed_point = R_robot @ base_point + np.array(robot_position)
        
        return transformed_point
    
    def draw_detection(self, image, detection):
        """在图像上绘制检测结果"""
        cx, cy = detection['pixel_coords']
        w, h = detection['size']
        angle_degrees = detection['obb_angle']
        standard_angle = detection['image_angle']
        conf = detection['confidence']
        class_name = detection['class_name']
        depth = detection['depth']
        
        # 绘制OBB框
        rect = ((cx, cy), (w, h), angle_degrees)
        box = cv2.boxPoints(rect).astype(int)
        cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
        
        # 绘制中心点
        cv2.circle(image, (int(cx), int(cy)), 5, (0, 0, 255), -1)
        
        # 绘制方向箭头
        arrow_length = min(w, h) / 2
        end_x = int(cx + arrow_length * np.cos(np.radians(standard_angle)))
        end_y = int(cy + arrow_length * np.sin(np.radians(standard_angle)))
        cv2.arrowedLine(image, (int(cx), int(cy)), (end_x, end_y),
                       (255, 0, 0), 3, tipLength=0.3)
        
        # 添加标签
        label = f"{class_name}: {conf:.2f}"
        cv2.putText(image, label, (int(cx-w/2), int(cy-h/2-10)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        angle_text = f"Angle: {standard_angle:.1f}"
        cv2.putText(image, angle_text, (int(cx-w/2), int(cy+h/2+20)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        depth_text = f"Depth: {depth*100:.0f}cm"
        cv2.putText(image, depth_text, (int(cx-w/2), int(cy+h/2+50)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    def process_frame(self, color_image, depth_frame, robot_position, robot_quaternion):
        """
        处理单帧图像，检测物体并计算3D位置
        
        Args:
            color_image: 彩色图像 (numpy array)
            depth_frame: 深度帧 (pyrealsense2 depth frame)
            robot_position: 机器人位置 [x, y, z]
            robot_quaternion: 机器人姿态四元数 [qx, qy, qz, qw]
            
        Returns:
            processed_image: 处理后的图像
            detections: 检测结果列表
        """
        # 运行YOLO检测
        results = self.model(color_image, verbose=False)[0]
        
        detections = []
        
        if hasattr(results, 'obb') and results.obb is not None:
            for i in range(len(results.obb.data)):
                cx, cy, w, h, angle_rad, conf, cls = results.obb.data[i].tolist()[:7]
                
                if conf > self.conf_threshold:
                    # 处理角度
                    angle_degrees = np.degrees(angle_rad)
                    standard_angle = self.obb_angle_to_standard(angle_degrees, w, h)
                    
                    # 获取深度
                    depth_value = self.get_average_depth(depth_frame, cx, cy, w, h)
                    
                    if depth_value > 0:
                        try:
                            # 1. 像素坐标 → 相机坐标
                            camera_coords = self.pixel2cam(cx, cy, depth_value)
                            
                            # 2. 相机坐标 → 基坐标
                            base_coords = self.cam2base(
                                camera_coords[0], camera_coords[1], camera_coords[2],
                                robot_position, robot_quaternion
                            )

                            # 步骤 3: 计算姿态
                            # 3.1. 物体相对于相机的姿态 (q_cam_obj)
                            angle_rad = np.radians(standard_angle) # 使用标准角度
                            half_angle = angle_rad / 2.0
                            q_cam_obj = [0.0, 0.0, math.sin(half_angle), math.cos(half_angle)]

                            # 3.2. 执行姿态变换链: q_base_obj = q_base_eef * q_eef_cam * q_cam_obj
                            q_base_eef = robot_quaternion
                            
                            # q_eef_obj = q_eef_cam * q_cam_obj
                            q_eef_obj = self.multiply_quaternions(self.q_eef_cam, q_cam_obj)
                            
                            # q_base_obj = q_base_eef * q_eef_obj
                            base_quaternion = self.multiply_quaternions(q_base_eef, q_eef_obj)

                            # # 计算物体的四元数（假设只绕 Z 轴旋转）
                            # half_angle = angle_rad / 2.0
                            # quaternion = [0.0, 0.0, math.sin(half_angle), math.cos(half_angle)]
                            
                            # 保存检测结果
                            detection = {
                                'class_id': int(cls),
                                'class_name': results.names[int(cls)],
                                'pixel_coords': (cx, cy),
                                'camera_coords': camera_coords,
                                'base_coords': base_coords,
                                'image_angle': standard_angle,
                                'depth': depth_value,
                                'confidence': conf,
                                'size': (w, h),
                                'obb_angle': angle_degrees,
                                'z_angle_rad': np.radians(standard_angle),
                                'quaternion': base_quaternion  # 修正：现在是物体在基坐标系下的四元数
                            }
                            detections.append(detection)
                            
                            # 绘制到图像上
                            self.draw_detection(color_image, detection)
                            
                        except Exception as e:
                            print(f"处理检测时出错: {e}")
        
        return color_image, detections

    def save_detections_as_json(self, detections, output_path="yolo_output.json"):
        """
        将检测结果转换为指定格式并保存为JSON文件。
        """
        if not detections:
            print("没有检测到物体，无法保存。")
            return

        # 用于为每个类别生成序号 (e.g., code1_1, code1_2)
        category_counters = {1: 1, 2: 1, 3: 1, 4: 1}
        yolo_detected_data = {}

        # 按类别和x坐标排序，确保命名一致性
        sorted_detections = sorted(detections, key=lambda d: (d['class_id'], d['base_coords'][0]))

        for det in sorted_detections:

            cat_id = det['class_id'] + 1

            if cat_id not in category_counters:
                print(f"警告：跳过未知的 category_id: {cat_id}")
                continue

            # 1. 生成积木名称
            instance_num = category_counters[cat_id]
            block_name = f"code{cat_id}_{instance_num}"
            if cat_id == 4: # code4是唯一的
                block_name = "code4"

            # 2. 提取基座坐标
            coords = det['base_coords'].tolist()

            # 3. 提取四元数（替换旋转角度）
            quat = det['quaternion']
            z_angle_deg = Rotation.from_quat(quat).as_euler('xyz', degrees=False)[2]

            # 4. 存入字典
            yolo_detected_data[block_name] = [coords, z_angle_deg]

            # 5. 更新计数器
            category_counters[cat_id] += 1
        
        # 保存为JSON文件
        with open(output_path, 'w') as f:
            json.dump(yolo_detected_data, f, indent=4)
        
        print(f"\n检测结果已成功保存到: {output_path}")
        # 打印到控制台方便预览
        import pprint
        pprint.pprint(yolo_detected_data)
        print("-" * 50)
        return yolo_detected_data
    
    def run_detection_loop(self):
        """运行检测循环"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        print("\n" + "="*50)
        print("按 's' 保存当前帧的检测结果到 yolo_output.json")
        print("按 'q' 退出检测循环")
        print("="*50 + "\n")

        
        try:
            while True:
                # 获取帧
                color_frame, depth_frame = self.get_frames()
                if not color_frame or not depth_frame:
                    continue
                
                # 转换为numpy数组
                color_image = np.asanyarray(color_frame.get_data())
                
                # 处理帧
                processed_image, detections = self.process_frame(
                    color_image, depth_frame, 
                    self.robot_position, self.robot_quaternion
                )
                
                # 显示结果信息
                if detections:
                    print(f"\n检测到 {len(detections)} 个物体:")
                    for i, det in enumerate(detections):
                        print(f"  目标 #{i+1}: {det['class_name']}")
                        print(f"    像素坐标: ({det['pixel_coords'][0]:.1f}, {det['pixel_coords'][1]:.1f})")
                        print(f"    相机坐标: ({det['camera_coords'][0]:.3f}, {det['camera_coords'][1]:.3f}, {det['camera_coords'][2]:.3f}) m")
                        print(f"    基座坐标: ({det['base_coords'][0]:.3f}, {det['base_coords'][1]:.3f}, {det['base_coords'][2]:.3f}) m")
                        print(f"    角度: {det['image_angle']:.1f}°, 深度: {det['depth']*100:.0f}cm")
                        print("  " + "-" * 40)
                
                # 显示 YOLO 检测结果图像（带有检测框）
                cv2.imshow("Detection", processed_image)
                
                # # 显示深度图
                # depth_image = np.asanyarray(depth_frame.get_data())
                # depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                # depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
                # cv2.imshow("Depth", depth_colormap)
                
                # 键盘控制
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'): # 新增保存功能
                    print("\n[操作] 正在保存当前帧的检测结果...")
                    
                    # 创建保存目录（如果不存在）
                    if not os.path.exists('saved_frames'):
                        os.makedirs('saved_frames')
                    
                    # 生成时间戳
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    
                    # 保存 RGB 图像
                    cv2.imwrite(f'saved_frames/rgb_{timestamp}.png', color_image)
                    
                    # 保存 YOLO 检测结果图像（带有检测框）
                    cv2.imwrite(f'saved_frames/detection_{timestamp}.png', processed_image)
                    
                    # 保存深度图（使用 colormap）
                    # cv2.imwrite(f'saved_frames/depth_{timestamp}.png', depth_colormap)
                    
                    # 保存 JSON 文件（使用时间戳避免覆盖）
                    self.save_detections_as_json(detections, f'saved_frames/yolo_output_{timestamp}.json')
        
        except KeyboardInterrupt:
            print("\n检测被用户中断")
        except Exception as e:
            print(f"检测运行出错: {e}")
        finally:
            # 清理资源
            self.pipeline.stop()
            cv2.destroyAllWindows()
            print("检测已结束")

# 主函数
if __name__ == "__main__":
    detector = DetectionSystem()

    print("\n[模式] 独立检测模式 (连接机械臂获取位姿)")
    
    # 初始化机械臂
    CAN_IFACE = "can0"
    robot = C_PiperInterface_V2(CAN_IFACE)
    robot.ConnectPort()
    while not robot.EnablePiper():
        time.sleep(0.2)
    
    # 获取机械臂末端位姿并转换为YOLO需要的格式
    end_pose_msg = robot.GetArmEndPoseMsgs()
    default_position = [
        end_pose_msg.end_pose.X_axis / 1000000.0,
        end_pose_msg.end_pose.Y_axis / 1000000.0,
        end_pose_msg.end_pose.Z_axis / 1000000.0
    ]
    default_quaternion = Rotation.from_euler('xyz', [
        end_pose_msg.end_pose.RX_axis / 1000.0,
        end_pose_msg.end_pose.RY_axis / 1000.0,
        end_pose_msg.end_pose.RZ_axis / 1000.0
    ], degrees=True).as_quat().tolist()
    print(f"使用默认虚拟位姿: Pos={default_position}, Quat={default_quaternion}")
    detector.set_robot_pose(default_position, default_quaternion)
    
    detector.run_detection_loop()