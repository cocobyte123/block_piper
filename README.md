
## Piper 官方 SDK

https://github.com/agilexrobotics/piper_sdk

### 拉取

```
cd ../
git clone git@github.com:agilexrobotics/piper_sdk.git
```

## PiperX + RealSense 初始化

### 激活环境

在联想小本儿上面运行  `conda activate piper`

### RealSense

运行指令

```
realsense-viewer
```

观察是否连接上对应的realsense摄像头，在这个界面能看到就ok

### Piper

#### 自用

```
sudo ip link set can0 down
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up
```

#### 官方

- PC has only one USB-to-CAN module connected:

```shell
bash can_activate.sh can0 1000000
```

Here, `can0` can be replaced with any name, and `1000000` is the baud rate, which cannot be changed.

- PC has multiple USB-to-CAN modules connected, but only one module is activated at a time

Note: This case applies when using both the robot arm and the chassis.

(1) Find the USB hardware address of the CAN module. Unplug all CAN modules and plug in only the one connected to the robot arm, then execute:

```shell
bash find_all_can_port.sh
```

Record the USB port value, for example, 3-1.4:1.0.

(2) Activate the CAN device. Assuming the USB port value is 3-1.4:1.0, run:

```shell
bash can_activate.sh can_piper 1000000 "3-1.4:1.0"
```

#### 运行前检查清单

1. CAN 适配器是否被识别

```
lsusb | grep "CAN adapter"
```
预期输出包含：`OpenMoko, Inc. Geschwister Schneider CAN adapter`

2. can0 接口状态

```
ip link show can0
```
预期看到 <UP,NOARP> 且 state UP。如果显示 state DOWN，需要先执行：

```
cd /home/coco/python_xm/piper/block_model_12.29_v0
sudo bash enable_can.sh
```
3. 相机是否连接

```
lsusb | grep RealSense
```
预期输出包含：`Intel(R) RealSense(TM) Depth Camera 455f`


一键检查脚本

```
echo "=== 1.CAN适配器 ===" && lsusb | grep "CAN adapter" && \
echo "=== 2.can0状态 ===" && ip link show can0 | head -1 && \
echo "=== 3.相机 ===" && lsusb | grep RealSense && \
echo "=== 4.Python依赖 ===" && python3 -c "from piper_sdk import C_PiperInterface_V2; import numpy; import cv2; print('OK')" && \
echo "=== 5.YOLO权重 ===" && ls -lh /home/coco/python_xm/piper/block_model_12.29_v0/weights/best_1210_2.pt && \
echo "=== 全部检查通过 ==="
```

全部通过后，运行：
```
python /home/coco/python_xm/piper/block_model_12.29_v0/main.py
```

### 问题

### 检测不到can0

将机械臂USB接口直接连接到上位机，不要通过拓展坞。再次检测


## 运行

### 在 block_model 目录下

```
sudo bash enable_can.sh can0 1000000
```

### 在 piper_sdk/demo/v2 目录下

重置
```shell
python piper_ctrl_reset.py
```

使能
```
python piper_ctrl_enable.py
```

回零
```
python piper_ctrl_go_zero.py
```

### 回到 block_model 目录

```
python main.py
```


## 记录
