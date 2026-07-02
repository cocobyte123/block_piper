#!/usr/bin/env python3
"""Warm up Piper and show the RealSense color stream.

This script is intended as a quick hardware sanity check before running the
full pick-and-place demo.
"""

import argparse
import time


def go_zero(can_iface: str, speed: int, enable_timeout: float, settle_time: float) -> None:
    from piper_sdk import C_PiperInterface_V2

    print(f"[arm] Connecting Piper on {can_iface}...")
    piper = C_PiperInterface_V2(can_iface)
    piper.ConnectPort()

    print("[arm] Enabling Piper...")
    start_time = time.time()
    while not piper.EnablePiper():
        if time.time() - start_time > enable_timeout:
            raise TimeoutError(f"Piper enable timed out after {enable_timeout:.1f}s")
        time.sleep(0.01)

    factor = 57295.7795  # 1000 * 180 / pi
    position = [0, 0, 5, 0, 0, 0, 0]
    joints = [round(value * factor) for value in position]

    print(f"[arm] Sending zero pose joints: {joints}")
    piper.ModeCtrl(0x01, 0x01, speed, 0x00)
    piper.JointCtrl(joints[0], joints[1], joints[2], joints[3], joints[4], joints[5])
    piper.GripperCtrl(abs(joints[6]), 1000, 0x01, 0)

    if settle_time > 0:
        print(f"[arm] Waiting {settle_time:.1f}s for the arm to settle...")
        time.sleep(settle_time)

    print("[arm] Zero command sent.")


def show_realsense(width: int, height: int, fps: int, seconds: float) -> None:
    import cv2
    import numpy as np
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    print(f"[camera] Starting RealSense color stream: {width}x{height}@{fps}")
    pipeline.start(config)

    window_name = "block_piper RealSense Warmup"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    start_time = time.time()
    frame_count = 0
    last_fps_time = start_time
    measured_fps = 0.0

    try:
        for _ in range(10):
            pipeline.wait_for_frames()

        print("[camera] Showing camera image. Press q or Esc to exit.")
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            frame_count += 1

            now = time.time()
            if now - last_fps_time >= 1.0:
                measured_fps = frame_count / (now - last_fps_time)
                frame_count = 0
                last_fps_time = now

            cv2.putText(
                color_image,
                f"RealSense OK | FPS {measured_fps:.1f} | q/Esc to exit",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.imshow(window_name, color_image)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

            if seconds > 0 and now - start_time >= seconds:
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("[camera] RealSense stream stopped.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Warm up Piper by sending zero pose, then show RealSense color image."
    )
    parser.add_argument("--can", default="can0", help="CAN interface name. Default: can0")
    parser.add_argument("--speed", type=int, default=30, help="Piper zero motion speed. Default: 30")
    parser.add_argument("--enable-timeout", type=float, default=10.0, help="Piper enable timeout in seconds")
    parser.add_argument("--settle-time", type=float, default=2.0, help="Wait time after zero command")
    parser.add_argument("--skip-arm", action="store_true", help="Only show camera, do not move Piper")
    parser.add_argument("--skip-camera", action="store_true", help="Only zero Piper, do not open camera")
    parser.add_argument("--width", type=int, default=640, help="RealSense color width")
    parser.add_argument("--height", type=int, default=480, help="RealSense color height")
    parser.add_argument("--fps", type=int, default=30, help="RealSense color FPS")
    parser.add_argument(
        "--seconds",
        type=float,
        default=0.0,
        help="Camera display duration. Default 0 means until q/Esc.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.skip_arm:
        go_zero(args.can, args.speed, args.enable_timeout, args.settle_time)

    if not args.skip_camera:
        show_realsense(args.width, args.height, args.fps, args.seconds)

    print("[warmup] Done.")


if __name__ == "__main__":
    main()
