import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import threading
import time

class CameraViewerNode(Node):
    """
    一个专门用于订阅和实时显示彩色与深度图像的ROS2节点。
    """
    def __init__(self):
        # 【核心修复】在创建节点时，指定与发布者相同的命名空间 'airbot_play'
        super().__init__('camera_viewer_node', namespace='airbot_play')
        self.bridge = CvBridge()
        
        # --- 状态变量 ---
        self.latest_color_image = None
        self.latest_depth_image = None
        self.image_lock = threading.Lock()
        self.shutdown_flag = threading.Event()
        
        # --- 订阅者 ---
        # 【核心修复】使用您提供的正确相对话题名称
        self.color_subscription = self.create_subscription(
            Image,
            'side_camera/color/image_raw',  # 订阅正确的彩色图像话题
            self._color_callback,
            10)
        
        self.depth_subscription = self.create_subscription(
            Image,
            'side_camera/aligned_depth_to_color/image_raw',  # 订阅正确的深度图像话题
            self._depth_callback,
            10)
            
        self.get_logger().info('摄像机画面查看器已启动，正在订阅 /airbot_play/arm_camera/... 话题...')

    def _color_callback(self, msg):
        """处理接收到的彩色图像消息。"""
        try:
            # 将ROS图像消息转换为OpenCV格式 (BGR8是cv2.imshow的标准格式)
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            with self.image_lock:
                self.latest_color_image = cv_image
        except Exception as e:
            self.get_logger().error(f'转换彩色图像时出错: {e}')

    def _depth_callback(self, msg):
        """处理接收到的深度图像消息。"""
        try:
            # 'passthrough'保留原始数据类型 (通常是16UC1)
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
            with self.image_lock:
                self.latest_depth_image = cv_image
        except Exception as e:
            self.get_logger().error(f'转换深度图像时出错: {e}')

    def _display_loop(self):
        """在循环中显示图像的函数，将在独立线程中运行。"""
        while not self.shutdown_flag.is_set():
            color_display, depth_display = None, None
            
            with self.image_lock:
                if self.latest_color_image is not None:
                    color_display = self.latest_color_image.copy()
                if self.latest_depth_image is not None:
                    # 为了可视化，将深度图归一化到0-255的8位灰度图
                    depth_copy = self.latest_depth_image.copy()
                    # 安全地处理全黑的深度图
                    if np.max(depth_copy) > 0:
                        depth_normalized = cv2.normalize(depth_copy, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                        depth_display = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET) # 使用伪彩色更易观察
                    else:
                        depth_display = np.zeros((depth_copy.shape[0], depth_copy.shape[1], 3), dtype=np.uint8)

            # 显示图像
            if color_display is not None:
                cv2.imshow('Color Camera Feed', color_display)
            if depth_display is not None:
                cv2.imshow('Depth Camera Feed', depth_display)

            # 按 'q' 键退出循环
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.shutdown_flag.set()
                break
            
            time.sleep(0.03) # 约30 FPS

        # 清理
        cv2.destroyAllWindows()
        self.get_logger().info('图像显示窗口已关闭。')

    def start_display_thread(self):
        """启动显示线程。"""
        display_thread = threading.Thread(target=self._display_loop)
        display_thread.daemon = True
        display_thread.start()
        return display_thread

def run_camera_viewer():
    """
    启动摄像机查看器节点的入口函数。
    """
    print("[Camera Process] 正在初始化ROS2节点...")
    # rclpy.init() # 注意：初始化应由主程序完成
    camera_node = CameraViewerNode()
    
    # 启动显示线程
    display_thread = camera_node.start_display_thread()
    
    # 使用spin来处理回调，直到节点被关闭
    try:
        rclpy.spin(camera_node)
    except KeyboardInterrupt:
        pass
    finally:
        # 清理
        camera_node.shutdown_flag.set()
        display_thread.join(timeout=1)
        camera_node.destroy_node()
        # rclpy.shutdown() # 注意：关闭也应由主程序完成
        print("[Camera Process] 摄像机节点已关闭。")

# 如果直接运行此文件，则独立启动
if __name__ == '__main__':
    rclpy.init()
    run_camera_viewer()
    rclpy.shutdown()