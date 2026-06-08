import numpy as np
import time
import traceback
import json # 新增

from command_executor import CommandExecutor
from task_scheduler import TaskScheduler
from airbot_py.arm import AIRBOTPlay
from scipy.spatial.transform import Rotation

# 从其他脚本导入必要的类
# 确保这些文件与 main.py 在同一个Python路径下
from test_yolo import DetectionSystem
from compute_2d_transform import convert_json_to_desktop_format

def get_yolo_data(robot):
    """
    移动到观察点，运行YOLO检测，并返回积木数据。
    """
    # 1. 定义观察点位姿
    observation_pos = [0.12, 0.02, 0.30] 
    observation_quat = [
        0.5367762354319134,
        0.4601111298137444,
        -0.5279815453771249,
        0.4705364056459881,
    ]
    
    print("\n--- 步骤 1.1: 移动到YOLO观察点 ---")
    robot.move_to_cart_pose([observation_pos, observation_quat])
    print("已到达观察点，准备执行检测。")
    time.sleep(1)  # 等待机械臂稳定
    
    # 2. 初始化并运行YOLO检测
    detector = None
    try:
        print("初始化检测系统...")
        detector = DetectionSystem()
        
        # 等待相机稳定
        print("等待相机稳定...")
        time.sleep(2)
        
        # 获取当前机器人位姿并设置给检测器
        current_pose = robot.get_end_pose()
        detector.set_robot_pose(current_pose[0], current_pose[1])
        
        # 执行单次检测（现在直接返回desktop_data）
        desktop_data = detector.run_single_detection()
    finally:
        if detector:
            detector.close()
    
    if not desktop_data:
        print("\n错误：YOLO未检测到任何物体，无法继续任务。")
        return None
    
    print("检测和转换完成。")
    return desktop_data


def main():
    """
    项目主函数，负责初始化和执行积木搭建任务。
    """
    np.set_printoptions(precision=4, suppress=True, linewidth=120)
    
    # 定义机械臂的初始/归位姿态
    HOME_POSE = {
        "pos": [0.12, 0.02, 0.30] ,
        "quat": [
        0.5367762354319134,
        0.4601111298137444,
        -0.5279815453771249,
        0.4705364056459881,
    ]# x, y, z, w
    }

    # 使用 with 语句确保机器人连接安全关闭
    with AIRBOTPlay(url="localhost", port=50000) as robot:
        try:
            executor = CommandExecutor(robot)
            
            # 初始化任务规划器
            first_block_target_pos = [0.1365, 0.300, 0.014]
            scheduler = TaskScheduler(first_block_target_pos=first_block_target_pos)

            # =================================================================
            # 【修改】从YOLO实时获取数据，替换硬编码
            yolo_detected_data = get_yolo_data(robot)
            
            if yolo_detected_data is None:
                return # 如果检测失败，则退出

            scheduler.set_active_blocks_from_detection(yolo_detected_data)
            # =================================================================

            # 生成任务顺序
            build_order = (
                scheduler.architecture["layer_1"] 
                + scheduler.architecture["layer_2"]
                + scheduler.architecture["layer_3"] 
                + scheduler.architecture["layer_4"]
            )

            # 步骤 1: 移动到初始位置
            print("\n--- 步骤 1: 移动到初始归位姿态 ---")
            executor.robot.move_to_cart_pose([HOME_POSE["pos"], HOME_POSE["quat"]])
            print("机器人已就绪！")
            
            # 步骤 2: 执行搭建任务
            print("\n--- 步骤 2: 开始执行搭建任务 ---")
            executor.run_mission(scheduler, build_order)

            # 步骤 3: 任务完成，执行新的分步撤离逻辑
            print("\n" + "="*80)
            print("--- 步骤 3: 所有任务已完成，开始执行分步撤离 ---")
            
            # ... (后续撤离逻辑保持不变) ...
            current_pose = executor.robot.get_end_pose()
            current_pos = np.array(current_pose[0])
            
            safe_retreat_point = np.array([0.15, 0.1, 0.3])
            
            vertical_down_quat = Rotation.from_euler('y', 90, degrees=True).as_quat().tolist()

            print(f"  -> 当前位置: [{current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}]")
            print(f"  -> 目标安全点: [{safe_retreat_point[0]:.3f}, {safe_retreat_point[1]:.3f}, {safe_retreat_point[2]:.3f}]")

            step1_pos = [safe_retreat_point[0], current_pos[1], current_pos[2]]
            print(f"  -> 3.2. 沿X轴移动...")
            executor.robot.move_to_cart_pose([step1_pos, vertical_down_quat])

            step2_pos = [safe_retreat_point[0], current_pos[1], safe_retreat_point[2]]
            print(f"  -> 3.3. 沿Z轴移动 (抬升)...")
            executor.robot.move_to_cart_pose([step2_pos, vertical_down_quat])

            step3_pos = [safe_retreat_point[0], safe_retreat_point[1], safe_retreat_point[2]]
            print(f"  -> 3.4. 沿Y轴移动...")
            executor.robot.move_to_cart_pose([step3_pos, vertical_down_quat])
            
            print("  -> 已到达中间安全点。")

            print("\n--- 步骤 4: 移动到最终归位姿态 ---")
            executor.robot.move_to_cart_pose([HOME_POSE["pos"], HOME_POSE["quat"]])
            
            print("\n已移动到最终位置，任务结束。")
            print("="*80)

        except KeyboardInterrupt:
            print("\n用户通过 Ctrl+C 退出程序。")
        except Exception as e:
            print(f"\n程序运行出错: {e}")
            traceback.print_exc()
        finally:
            print("程序退出。")

if __name__ == "__main__":
    main()