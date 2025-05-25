# How to Efficiently Use Ultralytics YOLO in RDK's Robot Operating System (ROS)？


## Demo Videos

### RDK S100

#### YOLO11x, 640x640, 80 Classes, e2e 110FPS

https://www.bilibili.com/video/BV1EUE4zgE2V

#### 8 Channel YOLO12n, 640x640, 80 Classes, e2e 8 x 30 FPS

#### YOLO12n, 1280x1280, 80 Classes, e2e 30FPS

#### YOLOE 11 l Seg, Prompt Free, 640x640, 4585 Classes, e2e 30FPS

### RDK X5

#### YOLO11n, 640x640, 80 Classes, e2e 200FPS

https://www.bilibili.com/video/BV1NN91Y1EBP

#### YOLOv8n, 640x640, 80 Classes, e2e 220FPS

https://www.bilibili.com/video/BV12h41eYEer

#### YOLOv8n Seg, 640x640, 80 Classes, e2e 166FPS

https://www.bilibili.com/video/BV1xE4UeuEJh

<iframe src="//player.bilibili.com/player.html?isOutside=true&aid=113140175996419&bvid=BV1xE4UeuEJh&cid=25867061303&p=1" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"></iframe>


## Introduction to ROS2


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




## 如何在RDK的BPU上加速YOLO?

以下是你提供内容的中文翻译：

---

RDK 开发板使用了BPU（Brain Process Unit）来进行神经网络加速，这种处理器属于 NPU（神经网络处理单元）的一种。我们的 BPU 使用方式与 NVIDIA 的 TensorRT 非常相似，并且我们支持两种神经网络量化方案：**训练后量化（Post-Training Quantization, PTQ）** 和 **感知量化训练（Quantization-Aware Training, QAT）**。通常情况下，PTQ 所能达到的精度已经足够满足日常应用需求，而 QAT 更适合对精度要求更高的高级应用场景。

在 PTQ 方案下，我们使用 Ultralytics 的 YOLO 导出一个 ONNX 模型。该 ONNX 模型经过量化和编译之后，就可以被转换为可在 BPU 上加速运行的模型。

Ours:

![](imgs/OpenExplore_AI_ToolChain.jpg)

NVIDIA Tensor RT 示例图：

![](imgs/nvidia_tensorrt.PNG)

NVIDIA Tensor RT: https://developer.nvidia.cn/tensorrt


