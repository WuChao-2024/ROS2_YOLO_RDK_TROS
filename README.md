# How to Efficiently Use Ultralytics YOLO in RDK's Robot Operating System (ROS)？


## Demo Videos



## Introduction to ROS2

The Robot Operating System (ROS) is a set of software libraries and tools for building robot applications.  From drivers and state-of-the-art algorithms to powerful developer tools, ROS has the open source tools you need for your next robotics project.

Since ROS was started in 2007, a lot has changed in the robotics and ROS community.  The goal of the ROS 2 project is to adapt to these changes, leveraging what is great about ROS 1 and improving what isn’t.


## Introduction to Ultralytics YOLO

![](imgs/ultralytics_YOLO.jpg)

Introducing Ultralytics YOLO11, the latest version of the acclaimed real-time object detection and image segmentation model. YOLO11 is built on cutting-edge advancements in deep learning and computer vision, offering unparalleled performance in terms of speed and accuracy. Its streamlined design makes it suitable for various applications and easily adaptable to different hardware platforms, from edge devices to cloud APIs.

![](imgs/ultralytics_YOLO_tasks.jpg)

Ultralytics YOLO11 is a versatile AI framework that supports multiple computer vision tasks. The framework can be used to perform detection, segmentation, obb, classification, and pose estimation. Each of these tasks has a different objective and use case, allowing you to address various computer vision challenges with a single framework.

- [Ultralytics](https://www.ultralytics.com/)
- [Ultralytics YOLO Document](https://docs.ultralytics.com/)
- [Ultralytics YOLO GitHub](https://github.com/ultralytics/ultralytics)


## Inteoration to D-RObotics RDK

### D-Robotics

![](imgs/d-robotics.png)

地瓜机器人是专注机器人智能进化的底层基础设施提供商，通过提供高性能计算芯片和软硬协同、端云一体的全链路开发平台，全方位助力机器人开发与规模化落地，为人们带来更智能、更美好的体验。

- [地瓜机器人中文官网](https://d-robotics.cc/)
- [地瓜机器人英文官网](https://en.d-robotics.cc/)
- [地瓜开发者社区](https://developer.d-robotics.cc/)


### RDK S100

RDK S100P / RDK S100
- CPU: 6 x A78AE @ 2.0GHz / 6 x A78AE @ 1.5GHz
- BPU: 128TOPs @ int8 / 80TOPs @ int8
- DDR: LPDDR5@6400，96bit / LPDDR5@6400，96bit

![](imgs/rdk_s100.png)

RDK S100 是地瓜机器人面向具身智能和机器人场景推出的高性能、大算力模组, 独特的异构设计可以同时兼顾感知推理和实时运动控制的需求, 减少控制系统的体积和复杂度.

- [RDK S100 介绍(中文)](https://developer.d-robotics.cc/rdks100)

### RDK X5

- CPU: 8 x A55 @ 1.5 GHz
- BPU: 10 TOPs @ int8
- DDR: LPDDR4@4266, 32bit

![](imgs/rdk_x5.jpg)

D-Robotics RDK X5搭载Sunrise 5智能计算芯片，可提供高达10 Tops的算力，是一款面向智能计算与机器人应用的全能开发套件，接口丰富，极致易用，支持Transformer、RWKV、Occupancy、Stereo Perception等多种复杂模型和最新算法，加速智能化应用快速落地。

- [RDK X5 介绍(中文)](https://developer.d-robotics.cc/en/rdkx5)
- [RDK X5 介绍(英文)](https://developer.d-robotics.cc/rdkx5)


## Inteoration to TogetheROS.Bot (TROS)




## How to Accerate YOLO on RDK's BPU?

The RDK board utilizes a BPU (Brain Processing Unit) for neural network acceleration, which falls under the category of NPUs. Our BPU's usage method is very similar to that of NVIDIA's TensorRT, and we support both Post-Training Quantization (PTQ) and Quantization-Aware Training (QAT) schemes for neural networks. Generally, the precision achieved with PTQ is adequate for daily use, whereas QAT is more suitable for advanced applications.

Under the PTQ plan, we use Ultralytics' YOLO to export an ONNX model. This ONNX model, after quantization and compilation, can be transformed into a BPU-accelerated model. 

Ours: 

![](imgs/OpenExplore_AI_ToolChain.jpg)

NVIDIA Tensor RT
![](imgs/nvidia_tensorrt.PNG)

NVIDIA Tensor RT: https://developer.nvidia.cn/tensorrt

