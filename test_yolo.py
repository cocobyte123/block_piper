#!/usr/bin/env python3
"""单独测试 YOLO 检测（不连接机械臂），用于排查检测/环境问题。"""

import sys
import time
import traceback

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO


def main():
    # ==================== 加载配置 ====================
    import yaml
    CONFIG_PATH = "config/camera.yaml"
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    MODEL_PATH = cfg['detection']['model_path']
    CONFIDENCE_THRESHOLD = cfg['detection']['confidence_threshold']
    color_cfg = cfg['camera_streams']['color']
    depth_cfg = cfg['camera_streams']['depth']

    print("=" * 60)
    print("YOLO 检测测试脚本")
    print(f"  配置文件: {CONFIG_PATH}")
    print(f"  模型路径: {MODEL_PATH}")
    print(f"  置信度阈值: {CONFIDENCE_THRESHOLD}")
    print(f"  相机分辨率: {color_cfg['width']}x{color_cfg['height']} @ {color_cfg['fps']}fps")
    print("=" * 60)

    # ==================== 加载 YOLO 模型 ====================
    print("\n[1] 加载 YOLO 模型...")
    try:
        model = YOLO(MODEL_PATH)
        print("  ✓ 模型加载成功")
    except Exception as e:
        print(f"  ✗ 模型加载失败: {e}")
        traceback.print_exc()
        return

    # ==================== 启动 RealSense 相机 ====================
    print("\n[2] 启动 RealSense 相机...")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color,
                         color_cfg['width'], color_cfg['height'],
                         getattr(rs.format, color_cfg['format']),
                         color_cfg['fps'])
    config.enable_stream(rs.stream.depth,
                         depth_cfg['width'], depth_cfg['height'],
                         getattr(rs.format, depth_cfg['format']),
                         depth_cfg['fps'])

    try:
        profile = pipeline.start(config)
        print("  ✓ 相机启动成功")
    except Exception as e:
        print(f"  ✗ 相机启动失败: {e}")
        traceback.print_exc()
        return

    print(f"  已启用: {color_cfg['width']}x{color_cfg['height']} @ {color_cfg['fps']}fps")

    # ==================== 检测循环 ====================
    print("\n[3] 开始检测循环...")
    print("  按 'q' 或 Esc 退出")
    print("  按 's' 保存当前帧的检测结果图像")
    print("-" * 60)

    cv2.namedWindow("YOLO Detection Test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLO Detection Test", 1280, 720)

    frame_count = 0
    start_time = time.time()
    last_fps_time = start_time
    measured_fps = 0.0

    # 对齐到彩色帧
    align = rs.align(rs.stream.color)

    try:
        while True:
            # 获取对齐帧
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()

            if not color_frame:
                time.sleep(0.01)
                continue

            color_image = np.asanyarray(color_frame.get_data())
            frame_count += 1

            # FPS 计算
            now = time.time()
            if now - last_fps_time >= 1.0:
                measured_fps = frame_count / (now - last_fps_time)
                frame_count = 0
                last_fps_time = now

            # ============ YOLO OBB 检测 ============
            try:
                results = model(color_image, verbose=False)[0]
            except Exception as e:
                print(f"  ✗ YOLO 推理失败: {e}")
                traceback.print_exc()
                continue

            # 在图像上绘制检测结果
            detection_count = 0
            if hasattr(results, 'obb') and results.obb is not None:
                for i in range(len(results.obb.data)):
                    cx, cy, w, h, angle_rad, conf, cls = results.obb.data[i].tolist()[:7]

                    if conf < CONFIDENCE_THRESHOLD:
                        continue

                    detection_count += 1
                    class_name = results.names[int(cls)]
                    angle_deg = np.degrees(angle_rad)

                    # 绘制旋转矩形框 (绿色)
                    rect = ((cx, cy), (w, h), angle_deg)
                    box = cv2.boxPoints(rect).astype(int)
                    cv2.drawContours(color_image, [box], 0, (0, 255, 0), 2)

                    # 中心点 (红色)
                    cv2.circle(color_image, (int(cx), int(cy)), 5, (0, 0, 255), -1)

                    # 方向箭头 (蓝色)
                    arrow_len = min(w, h) / 2
                    end_x = int(cx + arrow_len * np.cos(angle_rad))
                    end_y = int(cy + arrow_len * np.sin(angle_rad))
                    cv2.arrowedLine(color_image, (int(cx), int(cy)),
                                    (end_x, end_y), (255, 0, 0), 2, tipLength=0.3)

                    # 标签
                    label = f"{class_name} {conf:.2f}"
                    cv2.putText(color_image, label,
                                (int(cx - w/2), int(cy - h/2 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # 叠加状态信息
            elapsed = now - start_time
            info_lines = [
                f"FPS: {measured_fps:.1f}",
                f"Detections: {detection_count}",
                f"Elapsed: {elapsed:.0f}s",
                "Q/Esc: quit | S: save",
            ]
            for i, text in enumerate(info_lines):
                cv2.putText(color_image, text, (10, 25 + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow("YOLO Detection Test", color_image)

            # 按键处理
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):  # q 或 Esc
                print("\n用户请求退出")
                break
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"test_yolo_{timestamp}.png"
                cv2.imwrite(filename, color_image)
                print(f"  ✓ 已保存检测图像: {filename}")

    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"\n运行异常: {e}")
        traceback.print_exc()
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        elapsed = time.time() - start_time
        print(f"\n检测结束，总运行时间: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
