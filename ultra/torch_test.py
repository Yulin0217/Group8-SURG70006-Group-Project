import os
import torch  # 引入 PyTorch 库，用于加载和运行 PyTorch 模型
from argparse import ArgumentParser
from holoscan.core import Application
from holoscan.operators import (
    AJASourceOp,  # AJA输入操作符，用于从AJA设备捕获视频流
    FormatConverterOp,  # 格式转换操作符，用于调整图像格式
    HolovizOp,  # 可视化操作符，用于显示推理结果
    SegmentationPostprocessorOp,  # 分割后处理操作符，用于处理模型推理输出
)
from holoscan.resources import BlockMemoryPool, CudaStreamPool, MemoryStorageType


class UltrasoundApp(Application):
    def __init__(self, data):
        """初始化超声波分割应用程序"""
        super().__init__()
        self.name = "Ultrasound App"  # 设置应用程序名称

        # 如果没有指定数据路径，使用环境变量中的默认路径
        if data == "none":
            data = os.environ.get("HOLOHUB_DATA_PATH", "")
        self.sample_data_path = data  # 存储样本数据路径

        # 定义 PyTorch 模型路径
        self.model_path = os.path.join(self.sample_data_path, "us_unet_256x256.pth")
        self.model = self.load_model()  # 加载 PyTorch 模型

    def load_model(self):
        """加载 PyTorch 模型"""
        # 使用 torch.load() 从指定路径加载模型
        model = torch.load(self.model_path)
        model.eval()  # 将模型设置为评估模式（禁用训练模式）
        return model

    def compose(self):
        """定义应用程序的操作符和数据流"""

        # 图像属性和内存计算参数
        n_channels = 4  # 输入图像通道数（RGBA格式）
        bpp = 4  # 每个像素的字节数（每个通道1字节）

        # 创建一个 CUDA 流池，用于管理并发 CUDA 计算
        cuda_stream_pool = CudaStreamPool(
            self,
            name="cuda_stream",  # CUDA 流池名称
            dev_id=0,  # 使用第一个 GPU 设备
            stream_flags=0,  # CUDA 流标志
            stream_priority=0,  # CUDA 流优先级
            reserved_size=1,  # 预留的 CUDA 流数量
            max_size=5,  # 最大 CUDA 流数量
        )

        # 创建 AJA 输入操作符，从 AJA 设备捕获视频流
        source = AJASourceOp(self, name="aja", **self.kwargs("aja"))

        # 计算丢弃 Alpha 通道的内存块大小和数量
        drop_alpha_block_size = 1280 * 720 * n_channels * bpp  # 1280x720 的 4 通道图像
        drop_alpha_num_blocks = 2  # 内存块数量

        # 创建格式转换操作符，用于丢弃 Alpha 通道
        drop_alpha_channel = FormatConverterOp(
            self,
            name="drop_alpha_channel",
            pool=BlockMemoryPool(
                self,
                storage_type=MemoryStorageType.DEVICE,  # 内存存储类型为设备内存
                block_size=drop_alpha_block_size,  # 内存块大小
                num_blocks=drop_alpha_num_blocks,  # 内存块数量
            ),
            cuda_stream_pool=cuda_stream_pool,  # 使用 CUDA 流池
            **self.kwargs("drop_alpha_channel"),
        )

        # 预处理操作符的参数设置
        width_preprocessor = 1280  # 预处理宽度
        height_preprocessor = 720  # 预处理高度
        preprocessor_block_size = width_preprocessor * height_preprocessor * n_channels * bpp
        preprocessor_num_blocks = 3  # 内存块数量

        # 创建预处理操作符，将图像格式调整为适合模型推理的格式
        segmentation_preprocessor = FormatConverterOp(
            self,
            name="segmentation_preprocessor",
            pool=BlockMemoryPool(
                self,
                storage_type=MemoryStorageType.DEVICE,  # 内存存储类型为设备内存
                block_size=preprocessor_block_size,  # 内存块大小
                num_blocks=preprocessor_num_blocks,  # 内存块数量
            ),
            cuda_stream_pool=cuda_stream_pool,  # 使用 CUDA 流池
            **self.kwargs("segmentation_preprocessor"),
        )

        # 定义一个 PyTorch 推理函数
        def pytorch_inference(image_tensor):
            """执行 PyTorch 模型推理"""
            with torch.no_grad():  # 在不计算梯度的上下文中执行推理
                return self.model(image_tensor)

        # 计算后处理内存块大小和数量
        postprocessor_block_size = 256 * 256  # 分割后处理内存块大小
        postprocessor_num_blocks = 2  # 内存块数量

        # 创建分割后处理操作符，处理模型推理结果
        segmentation_postprocessor = SegmentationPostprocessorOp(
            self,
            name="segmentation_postprocessor",
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
            name="segmentation_visualizer",
            cuda_stream_pool=cuda_stream_pool,  # 使用 CUDA 流池
            **self.kwargs("segmentation_visualizer"),
        )

        # 定义操作符之间的数据流
        self.add_flow(source, segmentation_visualizer, {("video_buffer_output", "receivers")})
        self.add_flow(source, drop_alpha_channel, {("video_buffer_output", "")})
        self.add_flow(drop_alpha_channel, segmentation_preprocessor)

        # 使用 PyTorch 执行推理后，将推理结果传递给后处理和可视化模块
        self.add_flow(segmentation_preprocessor, segmentation_postprocessor, {("", "receivers")})
        self.add_flow(segmentation_postprocessor, segmentation_visualizer, {("", "receivers")})


if __name__ == "__main__":
    # 使用 ArgumentParser 解析命令行参数
    parser = ArgumentParser(description="Ultrasound segmentation demo application.")
    parser.add_argument("-d", "--data", default="none", help="Set the data path for the model")
    args = parser.parse_args()  # 解析命令行参数

    # 创建并运行超声波分割应用程序实例
    app = UltrasoundApp(data=args.data)
    app.config(os.path.join(os.path.dirname(__file__), "ultrasound_segmentation.yaml"))  # 加载配置文件
    app.run()  # 运行应用程序
