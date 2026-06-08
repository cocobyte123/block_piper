import yaml
import pyrealsense2 as rs
import cv2
import numpy as np
import time
from ultralytics import YOLO
from scipy.spatial.transform import Rotation
from airbot_py.arm import AIRBOTPlay, RobotMode, SpeedProfile


# 初始化机器人
class RobotInitializer:

    def __init__(self, config_path, robot):

        # self.config = self.load_config(config_path)
        self.robot = robot

        self.translation_1 = [0.1096244807434647, -0.0005740185108923657, 0.29688057655132444]
        observation_quat = [0.0, 1.0, 0.0, 0.0] 

        
        # 获取参数
        # self.robot_pos = self.config(['cart_waypoints'])

        self.robot.switch_mode(RobotMode.PLANNING_POS)
        self.robot.set_speed_profile(SpeedProfile.DEFAULT)
        self.robot.move_to_cart_pose(
            [
                self.translation_1,
                self.orientation_1,
            ]
        )
        
    def load_config(self, config_path):
        """加载YAML配置文件"""
        with open(config_path) as f:
            return yaml.safe_load(f)
        
    def move_to_target(self):
        try:
            
            translation= [0.30, 0.02, 0.05]    
            translation= [0.20, 0.05, 0.05]  
            translation= [0.40, 0.12, 0.05]   
            translation= [0.15, 0.17, 0.05] 


            translation= [0.2114, 0.072, 0.05]  
            # translation= [0.1749, 0.1686, 0.05]   
            # translation= [0.216, -0.0285, 0.05] 
         


            orientation = [
                    0.5367762354319134,
                    0.4601111298137444,
                    -0.5279815453771249,
                    0.4705364056459881,
                ]
            print(f"第一步：目标位置: {translation}, 目标姿态: {orientation}")
            
            self.robot.move_to_cart_pose([translation, orientation])
            self.robot.move_eef_pos([0.00])
            
            time.sleep(0.5)
    
            
        except Exception as e:
                print(f"移动失败: {e}")

    def run(self):
        self.move_to_target()
        
def main():

    config_path = "config/camera.yaml"
    with AIRBOTPlay(url="localhost", port=50000) as robot:
        try:
            # 创建初始化实例
            detector = RobotInitializer(config_path, robot)
            
            detector.run()
        except FileNotFoundError:
            print(f"配置文件未找到: {config_path}")
        except KeyboardInterrupt:
            print("程序被用户中断")
        except Exception as e:
            print(f"程序运行出错: {e}")
            import traceback
            traceback.print_exc()
            
if __name__ == "__main__":  
    main()


                
