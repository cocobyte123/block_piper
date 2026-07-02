#!/usr/bin/env python3
"""Test the simplified offset-based grasp flow.

This script does not run the full build/place pipeline. It is meant for
calibrating three grasp offsets:

1. ALIGN_OFFSET_X_MM: base-frame X offset after pixel centering.
2. ALIGN_OFFSET_Y_MM: base-frame Y offset after pixel centering.
3. GRASP_LATERAL_OFFSET_MM: gripper-right offset applied after gripper rotation.

The arm iteratively centers the target, moves to the final grasp pose, pauses
for inspection, and then returns to the global observation point.
"""

import argparse
import time

import numpy as np

from main import (
    CameraManager,
    GLOBAL_OBSERVATION_CONFIG,
    calculate_gripper_angle,
    observe_from_global_view,
    observe_from_local_view,
    preprocess_yolo_angles,
    refine_position_to_center_with_spatial_tracking,
    rotate_gripper_to_angle,
)
from piper_sdk import C_PiperInterface_V2


CAN_IFACE = "can0"
TARGET_PREFIX = "code1"

# Offset 1/2: base-frame fixed offset after pixel centering.
ALIGN_OFFSET_X_MM = 0.0
ALIGN_OFFSET_Y_MM = 0.0

# Offset 3: gripper-right offset after the gripper has rotated.
GRASP_LATERAL_OFFSET_MM = 5.0

PICK_Z_M = 0.120
PRE_GRASP_HEIGHT_M = 0.080
PAUSE_AT_PICK_SECONDS = 5.0
CLOSE_GRIPPER_AFTER_PAUSE = False

ALIGN_MAX_ITERATIONS = 5
ALIGN_TOLERANCE_PIXELS = 15

YOLO_CORRECTIONS = {
    "code3_1": [0.0, 0.0, -0.02, 0.0],
    "code3_2": [0.0, 0.0, -0.02, 0.0],
    "code4": [0.0, 0.0, -0.05, 0.0],
}


def get_current_pose(robot):
    msg = robot.GetArmEndPoseMsgs()
    pos = np.array([
        msg.end_pose.X_axis / 1000000.0,
        msg.end_pose.Y_axis / 1000000.0,
        msg.end_pose.Z_axis / 1000000.0,
    ])
    euler_deg = np.array([
        msg.end_pose.RX_axis / 1000.0,
        msg.end_pose.RY_axis / 1000.0,
        msg.end_pose.RZ_axis / 1000.0,
    ])
    return pos, euler_deg


def move_to_pose(robot, pos, euler_deg, speed=30, wait_time=0.8):
    piper_pos = [int(round(v * 1000000)) for v in pos]
    piper_euler = [int(round(v * 1000)) for v in euler_deg]

    print(
        f"  -> Move: pos=[{pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f}], "
        f"euler=[{euler_deg[0]:.1f}, {euler_deg[1]:.1f}, {euler_deg[2]:.1f}], speed={speed}"
    )
    robot.MotionCtrl_2(0x01, 0x00, speed, 0x00)
    robot.EndPoseCtrl(
        piper_pos[0], piper_pos[1], piper_pos[2],
        piper_euler[0], piper_euler[1], piper_euler[2],
    )
    time.sleep(wait_time)


def gripper_open(robot):
    print("  -> Open gripper")
    robot.GripperCtrl(gripper_angle=80000, gripper_effort=1000, gripper_code=0x01, set_zero=0x00)
    time.sleep(0.5)


def gripper_close(robot):
    print("  -> Close gripper")
    robot.GripperCtrl(gripper_angle=0, gripper_effort=1000, gripper_code=0x01, set_zero=0x00)
    time.sleep(0.8)


def choose_candidate(processed_yolo_data, target_prefix):
    candidates = []
    for block_id, data in processed_yolo_data.items():
        if not block_id.startswith(target_prefix):
            continue
        if len(data) < 3 or data[2] is None:
            continue

        pos_x, pos_y, pos_z = data[0][:3]
        candidates.append((block_id, pos_x, pos_y, pos_z, data))

    if not candidates:
        return None, None

    candidates.sort(key=lambda item: (item[2], item[1]), reverse=True)

    print("  -> Candidate list:")
    for i, (block_id, pos_x, pos_y, pos_z, data) in enumerate(candidates):
        marker = "*" if i == 0 else " "
        px, py = data[2]
        print(
            f"     {marker} {block_id}: world=({pos_x:.3f}, {pos_y:.3f}, {pos_z:.3f}), "
            f"pixel=({px:.1f}, {py:.1f})"
        )

    return candidates[0][0], candidates[0][4]


def gripper_right_offset_xy(current_rz_deg, lateral_offset_mm):
    rz_rad = np.radians(current_rz_deg)
    offset_m = lateral_offset_mm / 1000.0
    return np.array([
        offset_m * np.cos(rz_rad),
        offset_m * np.sin(rz_rad),
    ])


def parse_args():
    parser = argparse.ArgumentParser(description="Offset calibration grasp test.")
    parser.add_argument("--target-prefix", default=TARGET_PREFIX, help="YOLO id prefix, e.g. code1/code2/code3/code4")
    parser.add_argument("--align-x-mm", type=float, default=ALIGN_OFFSET_X_MM, help="Base-frame X offset after centering")
    parser.add_argument("--align-y-mm", type=float, default=ALIGN_OFFSET_Y_MM, help="Base-frame Y offset after centering")
    parser.add_argument("--lateral-mm", type=float, default=GRASP_LATERAL_OFFSET_MM, help="Gripper-right offset after rotation")
    parser.add_argument("--pick-z", type=float, default=PICK_Z_M, help="Final grasp Z in meters")
    parser.add_argument("--pre-height", type=float, default=PRE_GRASP_HEIGHT_M, help="Height above pick pose before descent")
    parser.add_argument("--pause", type=float, default=PAUSE_AT_PICK_SECONDS, help="Pause seconds at final grasp pose")
    parser.add_argument("--align-iters", type=int, default=ALIGN_MAX_ITERATIONS, help="Max pixel-centering iterations")
    parser.add_argument("--align-tol", type=float, default=ALIGN_TOLERANCE_PIXELS, help="Pixel-centering tolerance")
    parser.add_argument("--close", action="store_true", default=CLOSE_GRIPPER_AFTER_PAUSE, help="Close gripper after pause")
    parser.add_argument("--no-viz", action="store_true", help="Disable camera visualization window")
    return parser.parse_args()


def main():
    args = parse_args()
    np.set_printoptions(precision=4, suppress=True, linewidth=120)

    print("\n=== test_offset: simplified offset grasp test ===")
    print(f"target_prefix={args.target_prefix}")
    print(f"align_offset=({args.align_x_mm:.1f}, {args.align_y_mm:.1f})mm in base XY")
    print(f"grasp_lateral_offset={args.lateral_mm:.1f}mm along gripper right")
    print(f"pick_z={args.pick_z:.3f}m, pre_height={args.pre_height:.3f}m, pause={args.pause:.1f}s")
    print(f"iterative_alignment=max {args.align_iters} iterations, tolerance={args.align_tol:.1f}px")
    print(f"close_after_pause={args.close}")

    robot = C_PiperInterface_V2(CAN_IFACE)
    robot.ConnectPort()
    while not robot.EnablePiper():
        print("Waiting for Piper enable...")
        time.sleep(0.5)

    camera_manager = None
    try:
        camera_manager = CameraManager(enable_visualization=not args.no_viz)
        gripper_open(robot)

        print("\n--- 1. Global observation ---")
        global_data = observe_from_global_view(robot, camera_manager)
        if not global_data:
            print("No global detection.")
            return

        processed_global = preprocess_yolo_angles(
            global_data,
            YOLO_CORRECTIONS,
            observation_rz_deg=GLOBAL_OBSERVATION_CONFIG["quat"][2],
        )
        global_id, global_candidate = choose_candidate(processed_global, args.target_prefix)
        if global_candidate is None:
            print(f"No candidate found for prefix {args.target_prefix}.")
            return

        rough_pos = global_candidate[0][:3]

        print("\n--- 2. Local observation ---")
        local_data, _ = observe_from_local_view(robot, camera_manager, rough_pos)
        if not local_data:
            print("Local observation failed; falling back to global data.")
            local_data = global_data

        processed_local = preprocess_yolo_angles(
            local_data,
            YOLO_CORRECTIONS,
            observation_rz_deg=GLOBAL_OBSERVATION_CONFIG["quat"][2],
        )
        target_id, target_data = choose_candidate(processed_local, args.target_prefix)
        if target_data is None:
            print(f"No local candidate found for prefix {args.target_prefix}.")
            return

        print(f"\n--- 3. Iterative pixel centering target: {target_id} ---")
        current_pos, _ = get_current_pose(robot)
        initial_world_pos = target_data[0][:3]
        initial_pixel = target_data[2]

        aligned_pos, final_pixel, tracked_world_pos = refine_position_to_center_with_spatial_tracking(
            robot,
            camera_manager,
            current_pos,
            initial_pixel,
            args.target_prefix,
            initial_world_pos,
            max_iterations=args.align_iters,
            tolerance_pixels=args.align_tol,
            stage_name="test_offset迭代对齐",
            tracking_mode="world",
            use_pid=True,
            pid_kp=0.65,
            pid_ki=0.0,
            pid_kd=0.04,
            max_move_mm=25.0,
            motion_speed=22,
            move_settle_time=0.12,
            detect_stabilize_time=0.05,
        )

        print(f"  -> aligned_pos=[{aligned_pos[0]:.6f}, {aligned_pos[1]:.6f}, {aligned_pos[2]:.6f}]")
        print(f"  -> final_pixel=({final_pixel[0]:.1f}, {final_pixel[1]:.1f})")

        print("\n--- 4. Rotate gripper ---")
        _, euler_deg = get_current_pose(robot)
        current_rz_deg = euler_deg[2]
        target_angle_rad, target_angle_deg = calculate_gripper_angle(target_data[1], current_rz_deg)
        print(f"  -> detected_angle={np.degrees(target_data[1]):.1f}deg, target_rz={target_angle_deg:.1f}deg")
        rotate_gripper_to_angle(robot, target_angle_rad)

        _, final_euler_deg = get_current_pose(robot)
        final_rz_deg = final_euler_deg[2]

        print("\n--- 5. Apply three offsets ---")
        base_xy = aligned_pos[:2] + np.array([args.align_x_mm, args.align_y_mm]) / 1000.0
        lateral_xy = gripper_right_offset_xy(final_rz_deg, args.lateral_mm)
        grasp_xy = base_xy + lateral_xy

        print(f"  -> aligned_xy:     [{aligned_pos[0]:.6f}, {aligned_pos[1]:.6f}]")
        print(f"  -> base XY offset: [{args.align_x_mm:.1f}, {args.align_y_mm:.1f}]mm")
        print(f"  -> lateral offset: [{lateral_xy[0]*1000:.1f}, {lateral_xy[1]*1000:.1f}]mm (RZ={final_rz_deg:.1f}deg)")
        print(f"  -> final grasp XY: [{grasp_xy[0]:.6f}, {grasp_xy[1]:.6f}]")

        pre_grasp_pos = np.array([grasp_xy[0], grasp_xy[1], args.pick_z + args.pre_height])
        pick_pos = np.array([grasp_xy[0], grasp_xy[1], args.pick_z])

        print("\n--- 6. Move to inspection grasp pose ---")
        move_to_pose(robot, pre_grasp_pos, final_euler_deg, speed=30, wait_time=0.8)
        move_to_pose(robot, pick_pos, final_euler_deg, speed=12, wait_time=0.8)

        print(f"\n  -> Holding final grasp pose for {args.pause:.1f}s. Observe the offset now.")
        time.sleep(args.pause)

        if args.close:
            gripper_close(robot)
        else:
            print("  -> Skipping gripper close. Use --close when you want a real grasp test.")

        print("  -> Returning to global observation point; no placement step is executed.")
        observation_pos = np.array(GLOBAL_OBSERVATION_CONFIG["pos"])
        observation_euler = np.array(GLOBAL_OBSERVATION_CONFIG["quat"])
        move_to_pose(robot, observation_pos, observation_euler, speed=45, wait_time=1.2)

        print("\n=== test_offset done ===")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        if camera_manager is not None:
            camera_manager.close()


if __name__ == "__main__":
    main()
