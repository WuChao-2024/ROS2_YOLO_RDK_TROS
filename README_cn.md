# 如何在RDK的机器人操作系统(ROS2)中高效使用YOLO算法?

## Abstract

![](imgs/ROS2_YOLO_RDK_TROS_169.jpeg)

- Ultralytics YOLO是一个开源的算法框架, 可以非常轻松高效的训练自己的算法模型. 
- RDK是地瓜机器人的机器人开发套件, 可以在RDK上运行基于Ubuntu 22.04的RDK OS. 同时可以运行基于ROS2 Humble的TROS, 高效利用RDK的异构计算能力, 也能方便的集成到其他由ROS2构建的机器人系统中.
- RDK Model Zoo维护了大部分的Ultralytics YOLO模型的导出ONNX, 转化到TROS的ROS2节点来进行高效算法推理模型的文档, 也提供了Python API和C++ API来使用YOLO算法模型. 


## Demo Videos

### RDK S100 Plus / RDK S100

#### S100: YOLO11x, 640x640, 80 Classes, e2e 110FPS

![](imgs/S100_YOLO11x.png)

- [BiliBili](https://www.bilibili.com/video/BV1EUE4zgE2V)
- [Youtube](https://www.youtube.com/watch?v=GeHY4D59PnU)

#### S100: YOLO12n, 1280x1280, 80 Classes, e2e 30FPS

![](imgs/S100_1280YOLO12n.png)

- [BiliBili](https://www.bilibili.com/video/BV1JyjGzfEEW)
- [Youtube](https://www.youtube.com/watch?v=PAP8rTfv4og)

#### S100: 8 Channel YOLO12n, 640x640, 80 Classes, e2e 8 x 30 FPS

![](imgs/S100_8YOLO12n.png)

- [BiliBili](https://www.bilibili.com/video/BV1RXjGzpEBU)
- [Youtube](https://www.youtube.com/watch?v=vVFwuFYN3KA)

#### S100: YOLOE 11 l Seg, Prompt Free, 640x640, 4585 Classes, e2e 30FPS

![](imgs/S100_YOLOE_11Seg.png)

- [BiliBili](https://www.bilibili.com/video/BV1EyjGzfEy6)
- [Youtube](https://www.youtube.com/watch?v=8qkSbUNlfNw)

### RDK X5

#### X5: YOLO11n, 640x640, 80 Classes, e2e 200FPS

![](imgs/X5_YOLO11n.png)

- [BiliBili](https://www.bilibili.com/video/BV1NN91Y1EBP)
- [Youtube](https://www.youtube.com/watch?v=uWW0bd0FZ-4)

#### X5: YOLOv8n, 640x640, 80 Classes, e2e 220FPS

![](imgs/X5_YOLOv8n.png)

- [BiliBili](https://www.bilibili.com/video/BV12h41eYEer)
- [Youtube](https://www.youtube.com/watch?v=H11J3lWZZsY)

#### X5: YOLOv8n Seg, 640x640, 80 Classes, e2e 166FPS

![](imgs/X5_YOLOv8Seg.png)

- [BiliBili](https://www.bilibili.com/video/BV1xE4UeuEJh)
- [Youtube](https://www.youtube.com/watch?v=4YyhO8oDpZE)

## Introduction to ROS2

![](imgs/ros2_history.png)

机器人操作系统（ROS）是一组用于构建机器人应用程序的软件库和工具. 从驱动程序和最先进的算法到强大的开发人员工具, ROS拥有您下一个机器人项目所需的开源工具. 

自从2007年ROS开始以来, 机器人和ROS社区发生了很多变化. ROS 2项目的目标是适应这些变化, 利用ROS 1的优点并改进缺点. 

![](imgs/Nodes-TopicandService.gif)

ROS中的每个节点应负责单个模块目的 (例如, 一个用于控制车轮电机的节点, 一个用于控制激光测距仪的节点等). 每个节点可以通过话题、服务、动作或参数向其他节点发送和接收数据, 一个完整的机器人系统由许多协同工作的节点组成. 

**这些节点的开发者可以是硬件厂商, 软件厂商, 算法工程师, 机器人工程师等. 只要我们开发的节点符合ROS2的规范, 就可以轻松的集成进一个庞大的机器人系统.**

- [ROS](https://www.ros.org/)
- [ROS2 英文手册](https://docs.ros.org/en/rolling/index.html)
- [ROS2 中文手册 by FishROS社区](http://dev.ros2.fishros.com/doc/)  
- [古月居 ROS 社区](https://www.guyuehome.com/)
- [BiliBili ROS2 入门21讲](https://www.bilibili.com/video/BV16B4y1Q7jQ/)
- [图文教程 ROS2 入门21讲](https://book.guyuehome.com/)
- [Gitee ROS2 入门21讲](https://gitee.com/guyuehome/ros2_21_tutorials/tree/master) 

## Introduction to Ultralytics YOLO

![](imgs/ultralytics_YOLO.jpg)

最新版本的广受好评的实时目标检测和图像分割模型Ultralytics YOLO11. YOLO11是建立在深度学习和计算机视觉的前沿进展, 在速度和准确性方面提供无与伦比的性能. 它的流线型设计使其适用于各种应用程序, 并易于适应不同的硬件平台, 从边缘设备到云API. 

![](imgs/ultralytics_YOLO_tasks.jpg)

Ultralytics YOLO11是一个通用的AI框架, 支持多种计算机视觉任务. 该框架可用于执行检测、分割、OBB、分类和姿态估计. 这些任务中的每一个都有不同的目标和用例, 允许您使用单个框架解决各种计算机视觉挑战. 

**我们可以使用Ultralytics YOLO来训练自己的算法模型, 得到自己的算法模型后, 参考RDK Model Zoo将其转化为TROS可运行的BPU模型.**

- [Ultralytics](https://www.ultralytics.com/)
- [Ultralytics YOLO Document](https://docs.ultralytics.com/)
- [Ultralytics YOLO GitHub](https://github.com/ultralytics/ultralytics)


## Introduction to D-Robotics RDK

### D-Robotics

![](imgs/d-robotics.png)

地瓜机器人是专注机器人智能进化的底层基础设施提供商, 通过提供高性能计算芯片和软硬协同、端云一体的全链路开发平台, 全方位助力机器人开发与规模化落地, 为人们带来更智能、更美好的体验. 

- [地瓜机器人中文官网](https://d-robotics.cc/)
- [地瓜机器人英文官网](https://en.d-robotics.cc/)
- [地瓜开发者社区](https://developer.d-robotics.cc/)


### RDK S100 Plus / RDK S100

- CPU: 6 x A78AE @ 2.0GHz / 6 x A78AE @ 1.5GHz
- BPU: 128TOPs @ int8 / 80TOPs @ int8
- DDR: LPDDR5@6400, 96bit / LPDDR5@6400, 96bit
- OS: RDK OS Based on Ubuntu 22.04

![](imgs/rdk_s100.png)

RDK S100 是地瓜机器人面向具身智能和机器人场景推出的高性能、大算力模组, 独特的异构设计可以同时兼顾感知推理和实时运动控制的需求, 减少控制系统的体积和复杂度.

- [RDK S100 介绍(中文)](https://developer.d-robotics.cc/rdks100)

### RDK X5

- CPU: 8 x A55 @ 1.5 GHz
- BPU: 10 TOPs @ int8
- DDR: LPDDR4@4266, 32bit
- OS: RDK OS Based on Ubuntu 22.04

![](imgs/rdk_x5.jpg)

D-Robotics RDK X5搭载Sunrise 5智能计算芯片, 可提供高达10 Tops的算力, 是一款面向智能计算与机器人应用的全能开发套件, 接口丰富, 极致易用, 支持Transformer、RWKV、Occupancy、Stereo Perception等多种复杂模型和最新算法, 加速智能化应用快速落地. 

- [RDK X5 介绍(中文)](https://developer.d-robotics.cc/rdkx5)
- [RDK X5 介绍(英文)](https://developer.d-robotics.cc/en/rdkx5)


## Introduction to TogetheROS.Bot (TROS)

![](imgs/Introduction_to_TROS.png)

TogetheROS.Bot是D-Robotics面向机器人厂商和生态开发者推出的机器人操作系统, 旨在释放机器人场景的智能潜能, 助力生态开发者和商业客户能够高效、便捷的进行机器人开发, 打造具有竞争力的智能机器人产品. 

TogetheROS.Bot支持在RDK平台上运行, 同时提供模拟器版本在X86平台上运行. RDK平台涵盖下图所示的全部功能, X86平台支持以图片回灌方式体验部分功能, 提高用户算法开发和验证效率, 并能够快速迁移到RDK平台. 

**我们维护了超30个TROS的Package, 这些都是标准的ROS2 Humble 节点. 源码托管在GitHub组织D-Robotics, 仓库名称一般由hobot_开头.**

- [GitHub](https://github.com/D-Robotics/)

### DataFlows

通过标准的ROS2 Humble 节点的的组合, TROS 可以对 USB、MIPI 和 IPC摄像头的数据进行智能分析. 同时也支持搭建多路并行式的视频流智能分析应用. 

#### RDK TROS USB Camera DataFlow

![](imgs/USB_Camera.png)

#### RDK TROS MIPI Camera DataFlow

![](imgs/USB_Camera.png)

#### RDK TROS IPC Camera DataFlow

![](imgs/IPC_Camera.png)

### TROS Packages and Nodes

#### IPC摄像头节点

这个节点负责将来自IPC网络摄像头的H264/H265的RTSP码流转化为ROS2 Topic.

- [TROS手册(中文社区)](https://developer.d-robotics.cc/rdk_doc/Robot_development/apps/video_boxs)
- [TROS手册(GitHub)](https://d-robotics.github.io/rdk_doc/en/Robot_development/apps/video_boxs/)
- [GitHub](https://github.com/D-Robotics/hobot_rtsp_client)

#### MIPI摄像头节点 (重点)

这个节点负责将来自MIPI摄像头的YUV420SP(NV12)图像转化为ROS2 Topic. 已经调试好ISP的MIPI摄像头参考文档即可.

- [TROS手册(中文社区)](https://developer.d-robotics.cc/rdk_doc/Robot_development/quick_demo/demo_sensor#mipi%E5%9B%BE%E5%83%8F%E9%87%87%E9%9B%86)
- [TROS手册(GitHub)](https://d-robotics.github.io/rdk_doc/en/Robot_development/quick_demo/demo_sensor/#mipi-camera)
- [GitHub](https://github.com/D-Robotics/hobot_mipi_cam)

#### USB摄像头节点

这个节点负责将来自USB摄像头的MJPEG图像转化为ROS2 Topic, 默认使用v4l2取流.

- [TROS手册(中文社区)](https://developer.d-robotics.cc/rdk_doc/Robot_development/quick_demo/demo_sensor#usb%E5%9B%BE%E5%83%8F%E9%87%87%E9%9B%86)
- [TROS手册(GitHub)](https://d-robotics.github.io/rdk_doc/en/Robot_development/quick_demo/demo_sensor/#usb-camera)
- [GitHub](https://github.com/D-Robotics/hobot_usb_cam)


#### Codec节点

Codec节点负责调用RDK的硬件编解码器, 其中, JPU可以硬件解码MJPEG为YUV420SP(NV12), 硬件编码YUV420SP(NV12)为MJPEG, VPU可以硬件解码H264/H265为YUV420SP(NV12), VPU可以硬件编码YUV420SP(NV12)为H264/H265. 

| RDK | Performance | Max Instance |
| --- | --- | --- |
| S100 JPU | 4K@90fps | 64 |
| S100 VPU | 4K@90fps | 32 |
| X5 JPU | 4K@60fps | 64 |
| X5 VPU | 4K@60fps | 32 |


- [TROS手册(中文社区)](https://developer.d-robotics.cc/rdk_doc/Robot_development/quick_demo/hobot_codec)
- [TROS手册(GitHub)](https://d-robotics.github.io/rdk_doc/en/Robot_development/quick_demo/hobot_codec)
- [GitHub](https://github.com/D-Robotics/hobot_codec)

#### hobot_cv 图像处理节点

hobot_cv是一个图像处理节点, 主要负责图像的预处理和后处理, 并以ROS2 Topic的形式发布出处理后的图像. 其中, 缩放, 裁剪, 旋转, padding等功能是调用硬件的VPS, VSE, Primary等硬件加速的. 硬件资源使用完了之后, 才会调用CPU进行图像处理.

- [TROS手册(中文社区)](https://developer.d-robotics.cc/rdk_doc/Robot_development/quick_demo/demo_cv#resize)
- [TROS手册(GitHub)](https://d-robotics.github.io/rdk_doc/en/Robot_development/quick_demo/demo_cv#resize)
- [GitHub](https://github.com/D-Robotics/hobot_cv)


#### hobot_dnn 算法推理节点 (重点)

订阅YUV420SP(NV12)的图像, 使用hobot_cv的硬件加速来进行图像前处理, 送入BPU进行推理, 并以ROS2 Topic的形式发布出算法推理结果.

目前支持: YOLOv5, v8, v10, 11, 12的Detect, Seg, KeyPoints等多种功能算法后处理, 也可以集成自己的算法后处理, 节点内自带推理多线程, 运行时通过不同的workconfig.json文件来区分不同的算法, 配置不同的算法参数.

- [TROS手册(中文社区)](https://developer.d-robotics.cc/rdk_doc/Robot_development/boxs/detection/yolo)
- [TROS手册(GitHub)](https://d-robotics.github.io/rdk_doc/en/Robot_development/boxs/detection/yolo)
- [GitHub](https://github.com/D-Robotics/hobot_dnn)

#### websocket 可视化节点

订阅图像进行可视化, 或订阅图像和算法结果, 时间同步后进行可视化. 可视化页面为Web页面，可以在任意一台局域网访问板卡的IP:8000号端口进行访问. 支持订阅多个ROS2 DOMAIN的话题来进行可视化.

- [TROS手册(中文社区)](https://developer.d-robotics.cc/rdk_doc/Robot_development/quick_demo/demo_render#web%E5%B1%95%E7%A4%BA)
- [TROS手册(GitHub)](https://d-robotics.github.io/rdk_doc/en/Robot_development/quick_demo/demo_render/#web)
- [GitHub](https://github.com/D-Robotics/hobot_websocket)


#### AI MSG 算法结果的自定义Topic  (重点)

所有由dnn_node发布的算法结果由这个功能包管理, 包括检测、分割、关键点等算法结果，可以阅读这个功能包的源码来编写订阅Topic的程序.

- [GitHub](https://github.com/D-Robotics/hobot_msgs)



## 如何在RDK的BPU上加速YOLO?

RDK 开发板使用了BPU（Brain Process Unit）来进行神经网络加速, 这种处理器属于 NPU（神经网络处理单元）的一种. 我们的 BPU 使用方式与 NVIDIA 的 TensorRT 非常相似, 并且我们支持两种神经网络量化方案：**训练后量化（Post-Training Quantization, PTQ）** 和 **感知量化训练（Quantization-Aware Training, QAT）**. 通常情况下, PTQ 所能达到的精度已经足够满足日常应用需求, 而 QAT 更适合对精度要求更高的高级应用场景. 

在 PTQ 方案下, 我们使用 Ultralytics 的 YOLO 导出一个 ONNX 模型. 该 ONNX 模型经过量化和编译之后, 就可以被转换为可在 BPU 上加速运行的模型. 

Ours:

![](imgs/OpenExplore_AI_ToolChain.jpg)

NVIDIA Tensor RT:

![](imgs/nvidia_tensorrt.PNG)

NVIDIA Tensor RT: https://developer.nvidia.cn/tensorrt


## RDK Model Zoo

如何编译出高效率的BPU模型给TROS使用？这里需要参考RDK Model Zoo中的YOLO的部分.

注意, 与TensorRT一致, BPU的工具链是开放的工具链, 部署的方式并不唯一, 您完全可以根据您自己的经验和实际需要来设计不同的部署方式, 编写适合您自己的部署方式的后处理程序. Model Zoo维护了相对较高性能的一些等价修改方法.

### RDK S100 Plus / RDK S100 Model Zoo

![](imgs/rdk_model_zoo_s.jpeg)

https://github.com/D-Robotics/rdk_model_zoo_s


### RDK X5 Model Zoo

![](imgs/rdk_model_zoo.jpg)

https://github.com/D-Robotics/rdk_model_zoo

### 使用方法

**当你参考Model Zoo的方法获得一个模型后, 修改TROS的dnn_node的workconfig.json文件即可. **



## 结语

本文介绍了如何在RDK的机器人操作系统(ROS2)中高效使用YOLO算法, 主要思想是使用Ultralytics YOLO去训练, 使用Model Zoo去转化和验证, 使用RDK TROS去高效Runtime.

