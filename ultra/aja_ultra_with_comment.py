import os
from argparse import ArgumentParser

# 导入Holoscan框架的核心类和操作符
from holoscan.core import Application
from holoscan.operators import (
    AJASourceOp,  # AJA输入操作符，用于从AJA设备捕获视频流
    FormatConverterOp,  # 格式转换操作符，用于转换图像格式
    HolovizOp,  # 可视化操作符，用于显示结果
    InferenceOp,  # 推理操作符，用于执行模型推理
    SegmentationPostprocessorOp,  # 分割后处理操作符，用于处理推理结果
)
from holoscan.resources import BlockMemoryPool, CudaStreamPool, MemoryStorageType

# 定义一个超声波分割应用程序类，继承自Holoscan的Application类
class UltrasoundApp(Application):
    def __init__(self, data):
        """初始化超声波分割应用程序"""

        super().__init__()

        # 设置应用程序名称
        self.name = "Ultrasound App"

        # 如果未指定数据路径，使用环境变量中的默认路径
        if data == "none":
            data = os.environ.get("HOLOHUB_DATA_PATH", "")

        # 存储样本数据路径
        self.sample_data_path = data

        # 定义模型路径映射，加载ONNX格式的超声波分割模型
        self.model_path_map = {
            "ultrasound_seg": os.path.join(self.sample_data_path, "us_unet_256x256_nhwc.onnx"),
        }

    # 定义应用程序的操作符和数据流
    def compose(self):
        # 图像属性和内存计算参数
        n_channels = 4  # 通道数（RGBA）
        bpp = 4  # 每个像素的字节数

        # 创建一个CUDA流池，用于管理并发CUDA计算
        cuda_stream_pool = CudaStreamPool(
            self,
            name="cuda_stream",  # CUDA流池的名称
            dev_id=0,  # 使用第一个GPU设备
            stream_flags=0,  # CUDA流标志
            stream_priority=0,  # CUDA流优先级
            reserved_size=1,  # 预留的CUDA流数量
            max_size=5,  # 最大CUDA流数量
        )

        # 创建AJA输入操作符，从AJA设备捕获视频流
        source = AJASourceOp(self, name="aja", **self.kwargs("aja"))
        print("source:\n", source)

        # 计算丢弃Alpha通道的内存块大小和数量
        drop_alpha_block_size = 1280 * 720 * n_channels * bpp  # 更新为720p分辨率
        drop_alpha_num_blocks = 2  # 内存块数量
        # 创建格式转换操作符，用于丢弃Alpha通道
        drop_alpha_channel = FormatConverterOp(
            self,
            name="drop_alpha_channel",  # 操作符的名称
            pool=BlockMemoryPool(
                self,
                storage_type=MemoryStorageType.DEVICE,  # 内存存储类型为设备内存
                block_size=drop_alpha_block_size,  # 内存块大小
                num_blocks=drop_alpha_num_blocks,  # 内存块数量
            ),
            cuda_stream_pool=cuda_stream_pool,  # 使用CUDA流池
            **self.kwargs("drop_alpha_channel"),
        )

        # 预处理操作符的参数设置
        width_preprocessor = 1280  # 预处理宽度（更新为720p宽度）
        height_preprocessor = 720  # 预处理高度（更新为720p高度）
        preprocessor_block_size = width_preprocessor * height_preprocessor * n_channels * bpp
        preprocessor_num_blocks = 3  # 内存块数量
        # 创建预处理操作符，将图像格式调整为适合推理的格式
        segmentation_preprocessor = FormatConverterOp(
            self,
            name="segmentation_preprocessor",  # 操作符的名称
            pool=BlockMemoryPool(
                self,
                storage_type=MemoryStorageType.DEVICE,  # 内存存储类型为设备内存
                block_size=preprocessor_block_size,  # 内存块大小
                num_blocks=preprocessor_num_blocks,  # 内存块数量
            ),
            cuda_stream_pool=cuda_stream_pool,  # 使用CUDA流池
            **self.kwargs("segmentation_preprocessor"),
        )

        # 推理操作符的参数设置
        n_channels_inference = 2  # 推理输入通道数
        width_inference = 256  # 推理输入宽度
        height_inference = 256  # 推理输入高度
        bpp_inference = 4  # 每个像素的字节数
        inference_block_size = (
            width_inference * height_inference * n_channels_inference * bpp_inference
        )
        inference_num_blocks = 2  # 内存块数量
        # 创建推理操作符，执行超声波分割模型的推理
        segmentation_inference = InferenceOp(
            self,
            name="segmentation_inference_holoinfer",  # 操作符的名称
            backend="trt",  # 使用TensorRT作为后端
            allocator=BlockMemoryPool(
                self,
                storage_type=MemoryStorageType.DEVICE,  # 内存存储类型为设备内存
                block_size=inference_block_size,  # 内存块大小
                num_blocks=inference_num_blocks,  # 内存块数量
            ),
            model_path_map=self.model_path_map,  # 模型路径映射
            pre_processor_map={"ultrasound_seg": ["source_video"]},  # 预处理映射
            inference_map={"ultrasound_seg": "inference_output_tensor"},  # 推理输出映射
            in_tensor_names=["source_video"],  # 输入张量名称
            out_tensor_names=["inference_output_tensor"],  # 输出张量名称
            enable_fp16=False,  # 禁用FP16
            input_on_cuda=True,  # 输入在CUDA设备上
            output_on_cuda=True,  # 输出在CUDA设备上
            transmit_on_cuda=True,  # 在CUDA设备上传输数据
        )
        print("Model:", segmentation_inference)
        print("segmentation_inference_vars:", vars(segmentation_inference))

        # 计算后处理内存块大小和数量
        postprocessor_block_size = width_inference * height_inference
        postprocessor_num_blocks = 2  # 内存块数量
        # 创建分割后处理操作符，处理推理结果
        segmentation_postprocessor = SegmentationPostprocessorOp(
            self,
            name="segmentation_postprocessor",  # 操作符的名称
            allocator=BlockMemoryPool(
                self,
                storage_type=MemoryStorageType.DEVICE,  # 内存存储类型为设备内存
                block_size=postprocessor_block_size,  # 内存块大小
                num_blocks=postprocessor_num_blocks,  # 内存块数量
            ),
            **self.kwargs("segmentation_postprocessor"),
        )

        # 创建可视化操作符，用于显示分割结果
        segmentation_visualizer = HolovizOp(
            self,
            name="segmentation_visualizer",  # 操作符的名称
            cuda_stream_pool=cuda_stream_pool,  # 使用CUDA流池
            **self.kwargs("segmentation_visualizer"),
        )

        # 定义操作符之间的数据流
        self.add_flow(source, segmentation_visualizer, {("video_buffer_output", "receivers")})
        self.add_flow(source, drop_alpha_channel, {("video_buffer_output", "")})
        self.add_flow(drop_alpha_channel, segmentation_preprocessor)
        self.add_flow(segmentation_preprocessor, segmentation_inference, {("", "receivers")})
        self.add_flow(segmentation_inference, segmentation_postprocessor, {("transmitter", "")})
        self.add_flow(
            segmentation_postprocessor,
            segmentation_visualizer,
            {("", "receivers")},
        )


if __name__ == "__main__":
    # 使用ArgumentParser解析命令行参数
    parser = ArgumentParser(description="Ultrasound segmentation demo application.")
    parser.add_argument(
        "-d", "--data", default="none", help="Set the data path for the model"
    )
    args = parser.parse_args()  # 解析命令行参数

    # 创建并运行超声波分割应用程序实例
    app = UltrasoundApp(data=args.data)
    # 加载配置文件
    app.config(os.path.join(os.path.dirname(__file__), "ultrasound_segmentation.yaml"))
    app.run()  # 运行应用程序
