# SAM TensorRT 推理优化项目

## 项目简介

本项目是基于TensorRT的SAM（Segment Anything Model）推理优化实现。它包括了编码器和解码器的TensorRT引擎封装，以及图像处理和推理流水线的实现。

## 项目结构
├── include
│ ├── encoder_decoder_pipeline.h
│ └── trt_infer.h
├── src
│ ├── encoder_decoder_pipeline.cpp
│ ├── main.cpp
│ └── trt_infer.cpp
├── test_images
│ └── ... (测试图像文件)
├── CMakeLists.txt
└── README.md

## 主要功能

1. TensorRT引擎封装（`TRTEngine`类）
2. 图像预处理（`ImageProcessor`类）
3. 编码器和解码器的推理实现（`Encoder`和`Decoder`类）
4. 编码器-解码器pipeline（`EncoderDecoderPipeline`类）

## 依赖项

- CUDA
- TensorRT
- OpenCV

## 编译和运行

1. 确保已安装所有依赖项。
2. 使用make构建项目：
    -make
3. 运行程序：
bash
./sam_trt_infer <encoder_model_path> <decoder_model_path>
其中，`<encoder_model_path>`和`<decoder_model_path>`分别是编码器和解码器的TensorRT模型文件路径。

## 使用说明

1. 将待处理的图像放入`test_images`目录。
2. 运行程序，它将自动处理`test_images`目录中的所有图像。
3. 程序将输出编码器和解码器的推理结果，包括推理时间和部分输出数据。

## 主要类说明

- `TRTEngine`: 封装TensorRT引擎的创建和管理。
- `ImageProcessor`: 处理图像预处理，包括缩放和归一化。
- `Encoder`: 实现SAM模型的编码器部分。
- `Decoder`: 实现SAM模型的解码器部分。
- `EncoderDecoderPipeline`: 管理整个推理流程，包括编码器和解码器的串联执行。

## 注意事项

- 确保TensorRT模型文件与代码中的输入输出配置一致。
- 图像预处理中使用了固定的均值和标准差，可能需要根据实际情况调整。
- 当前实现使用了固定的点坐标(800, 440)作为示例，实际使用时需要根据需求修改。
- 程序使用了异步处理方式，编码器和解码器在不同的线程中执行。

## 性能优化

- 使用CUDA流（cudaStream_t）来实现异步操作。
- 实现了编码器和解码器的并行处理，提高了整体吞吐量。
- 使用TensorRT进行模型优化，提高了推理速度。

## 未来改进

- 实现更灵活的点坐标输入机制
- 添加后处理步骤，如可视化分割结果
- 优化内存管理，减少不必要的内存拷贝
- 添加更多的错误处理和日志输出
- 实现批处理功能，进一步提高吞吐量
- 当前只实现了single label, single point的推理，未来将进一步完善代码

## References:

 1. SAM2 Repository: https://github.com/facebookresearch/segment-anything-2
 2. ONNX-SAM2-Segment-Anything：https://github.com/ibaiGorordo/ONNX-SAM2-Segment-Anything
## 贡献

欢迎提交问题和合并请求，一起改进这个项目！

## 许可证

本项目采用 Apache 2.0 license. 许可证。详情请见 [LICENSE](LICENSE) 文件。
