import numpy as np
import time
import traceback
from scipy.spatial.transform import Rotation

from command_executor import CommandExecutor
from task_scheduler import TaskScheduler
from piper_sdk import C_PiperInterface_V2
from test_yolo import DetectionSystem

# ================= 辅助函数定义 =================

def select_best_candidate(processed_yolo_data, yolo_prefix, img_center_x=640, img_center_y=360):
    """
    根据类型前缀筛选候选积木，选择最接近画面中心的积木。
    """
    # 1. 筛选出所有符合类型前缀（如 "code1"）的积木
    candidates = [k for k in processed_yolo_data.keys() if k.startswith(yolo_prefix)]
    
    if not candidates:
        return None, None
    
    best_candidate = None
    min_dist = float('inf')
    
    print(f"  -> 候选积木列表: {candidates}")
    for cand_id in candidates:
        data = processed_yolo_data[cand_id]
        # data 结构: [pos, angle] 或 [pos, angle, pixel_center]
        
        dist = float('inf')
        # 优先使用像素距离判断（最准）
        if len(data) >= 3 and data[2] is not None:
            px, py = data[2]
            dist = np.sqrt((px - img_center_x)**2 + (py - img_center_y)**2)
            print(f"    -> {cand_id}: 像素位置 ({px:.1f}, {py:.1f}), 距离中心 {dist:.1f}")
        else:
            # 回退：如果没有像素坐标，这里简单设为0，或者你可以传入当前末端坐标计算世界距离
            dist = 0 
            print(f"    -> {cand_id}: 无像素坐标，默认优先级")
        
        if dist < min_dist:
            min_dist = dist
            best_candidate = cand_id
    
    if best_candidate is None:
        return None, None
    
    selected_data = processed_yolo_data[best_candidate]
    print(f"  -> 最终选中: '{best_candidate}' (距离中心 {min_dist:.2f})")
    return best_candidate, selected_data

def get_yolo_data(robot, detector, block_pos, observation_quat=[-179, 0, -90], observation_z=0.25):
    """
    移动到观察点，运行YOLO检测，并返回积木数据。
    传入 detector，避免重复创建。
    """
    # 1. 计算观察点位姿：保持 Z 高度不变，调整 X 和 Y 到积木上方
    observation_pos = [block_pos[0], block_pos[1], observation_z]
    
    # Piper 移动：位置米 -> 0.001 mm，四元数 -> 0.001 度欧拉角
    piper_pos = [int(round(p * 1000000)) for p in observation_pos]  # 米 -> 0.001 mm
    print(f"  -> 转换为 Piper 单位位置: {piper_pos} (0.001 mm)")
    
    piper_euler = [int(round(e * 1000)) for e in observation_quat]  # 度 -> 0.001 度
    robot.MotionCtrl_2(0x01, 0x00, 50, 0x00)
    robot.EndPoseCtrl(piper_pos[0], piper_pos[1], piper_pos[2], piper_euler[0], piper_euler[1], piper_euler[2])
    

    time.sleep(3)
    end_pose_msg = robot.GetArmEndPoseMsgs()
    current_pos = [
        end_pose_msg.end_pose.X_axis / 1000000.0,
        
        end_pose_msg.end_pose.Y_axis / 1000000.0,
        end_pose_msg.end_pose.Z_axis / 1000000.0
    ]
    print(f"  -> 实际到达位置: [{current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}]")
    if not np.allclose(current_pos, observation_pos, atol=0.01):
        print("警告：机械臂未到达目标观察点，可能超出工作空间。")
        return None

    print("已到达观察点，准备执行检测。")

    try:
        print("等待相机稳定...")
        time.sleep(2)
        
        # 获取当前机器人位姿：Piper 返回 0.001 mm 和 0.001 度 -> 米 和 四元数
        end_pose_msg = robot.GetArmEndPoseMsgs()
        current_pose = [
            [end_pose_msg.end_pose.X_axis / 1000000.0, end_pose_msg.end_pose.Y_axis / 1000000.0, end_pose_msg.end_pose.Z_axis / 1000000.0],
            Rotation.from_euler('xyz', [end_pose_msg.end_pose.RX_axis / 1000.0, end_pose_msg.end_pose.RY_axis / 1000.0, end_pose_msg.end_pose.RZ_axis / 1000.0], degrees=True).as_quat().tolist()
        ]
        print(f"当前机械臂末端位姿: 位置 {current_pose[0]}, 四元数 {current_pose[1]}")
        
        detector.set_robot_pose(current_pose[0], current_pose[1])  # 传递机器人位姿给检测器，用于坐标变换
        
        desktop_data = detector.run_single_detection()
    except Exception as e:
        print(f"检测过程出错: {e}")
        return None
    
    if not desktop_data:
        print("\n错误：YOLO未检测到任何物体，无法继续任务。")
        return None
    
    print("检测和转换完成。",desktop_data)
    return desktop_data 

def preprocess_yolo_angles(yolo_data, corrections, angle_overrides):
    """
    预处理YOLO检测到的积木数据：单位转换 + 修正 + 角度归一化
    """
    if not yolo_data:
        return {}

    import copy
    processed_data = copy.deepcopy(yolo_data)

    # 内部函数：单位归一化
    def _to_meters(pos):
        if pos is None: return pos
        try:
            x = abs(float(pos[0]))
            # 启发式规则：大于5 -> mm, 1~5 -> cm, 其他 -> m
            if x > 5.0: return [float(p) / 1000.0 for p in pos]
            elif x > 1.0: return [float(p) / 100.0 for p in pos]
            else: return [float(p) for p in pos]
        except: return pos

    # 1. 单位转换
    for k, v in processed_data.items():
        if isinstance(v, (list, tuple)) and len(v) >= 1:
            v[0] = _to_meters(v[0])

    # 2. 应用修正和角度处理
    for block_id, data in processed_data.items():
        # data: [pos, angle, (optional)pixel]
        
        # 应用位置/角度修正
        correction = corrections.get(block_id, [0.0, 0.0, 0.0, 0.0])
        if any(c != 0.0 for c in correction):
            data[0][0] += correction[0]
            data[0][1] += correction[1]
            data[0][2] += correction[2]
            data[1] += correction[3]
        
        # 角度归一化 (对称物体)
        original_angle = data[1]
        if 'code3' not in block_id: # code3 可能不对称
            normalized_angle = (original_angle + np.pi / 2) % np.pi - np.pi / 2
            data[1] = normalized_angle
        
        # 强制覆盖角度
        if angle_overrides and block_id in angle_overrides:
            data[1] = angle_overrides[block_id]

    return processed_data

# ================= 主函数 =================

def main():
    
    np.set_printoptions(precision=4, suppress=True, linewidth=120)
    
    # 配置参数
    YOLO_CORRECTIONS = {
        "code3_1": [0.0, 0.0, -0.02, 0.0],
        "code3_2": [0.0, 0.0, -0.02, 0.0], 
        "code4":   [0.0, 0.0, -0.05, 0.0],
    }
    CODE3_ANGLE_OVERRIDES = {"code3_1": 0, "code3_2": 0}
    
    # 初始化机械臂
    CAN_IFACE = "can0"
    robot = C_PiperInterface_V2(CAN_IFACE)
    robot.ConnectPort()
    while not robot.EnablePiper():
        print("等待机械臂使能...")        
        time.sleep(0.5)

    try:
        # 初始化检测系统
        detector = DetectionSystem()
        
        # ---------------------------------------------------------
        # 步骤: 单次检测并依次移动到积木上方
        # ---------------------------------------------------------
        print("\n" + "="*60)
        print("--- 步骤: 单次检测并依次移动到积木上方 ---")
        print("="*60)
        
        # 固定观察点（可以调整）
        observation_pos = [0.105, 0.00, 0.32]  # 高处观察点
        observation_quat = [-160.0, 0.0, -90.0]
        
        # 运行YOLO检测
        yolo_data = get_yolo_data(robot, detector, observation_pos, observation_quat, observation_z=0.32)
        
        if not yolo_data:
            print("错误：未检测到任何积木，程序终止。")
            
            return
        
        # 预处理检测数据
        processed_data = preprocess_yolo_angles(yolo_data, YOLO_CORRECTIONS, CODE3_ANGLE_OVERRIDES)
        
        # 按照检测到的顺序（键排序）依次移动到积木上方
        sorted_block_ids = sorted(processed_data.keys())
        print(f"检测到积木: {sorted_block_ids}")
        
        for idx, block_id in enumerate(sorted_block_ids):
            print(f"\n>>> 处理第 {idx + 1}/{len(sorted_block_ids)} 个积木: {block_id}")
            
            data = processed_data[block_id]
            pos = data[0]  # [x, y, z]
            
            # 移动到积木上方（Z高度设为0.25m）
            target_pos = [pos[0], pos[1], 0.25]
            print(f"  -> 移动到积木上方: {target_pos}")
            
            piper_pos = [int(round(p * 1000000)) for p in target_pos]
            piper_euler = [int(round(e * 1000)) for e in observation_quat]
            robot.MotionCtrl_2(0x01, 0x00, 50, 0x00)
            robot.EndPoseCtrl(piper_pos[0], piper_pos[1], piper_pos[2], piper_euler[0], piper_euler[1], piper_euler[2])
            
            # 等待移动稳定
            time.sleep(2.5)
            
            # 验证位置
            end_pose_msg = robot.GetArmEndPoseMsgs()
            current_pos = [
                end_pose_msg.end_pose.X_axis / 1000000.0,
                end_pose_msg.end_pose.Y_axis / 1000000.0,
                end_pose_msg.end_pose.Z_axis / 1000000.0
            ]
            print(f"  -> 实际到达: [{current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}]")
            
            # 可选：在这里添加暂停或输入确认
            input("按Enter继续下一个积木...")

    except KeyboardInterrupt:
        print("\n用户中断程序。")
    except Exception as e:
        print(f"\n程序异常: {e}")
        traceback.print_exc()
    finally:
        try:
            detector.close()
        except:
            pass
        print("断开连接...")
        pass

if __name__ == "__main__":
    main()