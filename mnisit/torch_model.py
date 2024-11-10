import os
import torch
import torch.nn.functional as F
from argparse import ArgumentParser
from holoscan.core import Application
from holoscan.operators import (
    AJASourceOp,  # 操作符，用于从 AJA 设备捕获视频流
    FormatConverterOp,  # 操作符，用于格式转换
    HolovizOp,  # 操作符，用于实时显示图像
)
from holoscan.resources import BlockMemoryPool, CudaStreamPool, MemoryStorageType


class HandwrittenDigitApp(Application):
    def __init__(self, data):
        """初始化手写数字识别应用程序"""
        super().__init__()
        self.name = "Handwritten Digit Recognition App"  # 应用程序名称

        # 如果没有指定数据路径，使用默认路径
        if data == "none":
            data = os.environ.get("HOLOHUB_DATA_PATH", "")
        self.sample_data_path = data  # 存储数据路径

        # 定义 PyTorch 模型路径
        self.model_path = os.path.join(self.sample_data_path, "digit_recognition_model.pth")
        self.model = self.load_model()  # 加载 PyTorch 模型

    def load_model(self):
        """加载 PyTorch 模型"""
        model = torch.load(self.model_path)
        model.eval()  # 将模型设置为评估模式（禁用训练模式）
        return model

    def compose(self):
        """定义应用程序的操作符和数据流"""

        n_channels = 1  # 输入图像通道数（灰度图像）
        bpp = 1  # 每个像素的字节数（灰度图像）

        # 创建一个 CUDA 流池，用于管理并发 CUDA 计算
        cuda_stream_pool = CudaStreamPool(
            self,
            name="cuda_stream",
            dev_id=0,
            stream_flags=0,
            stream_priority=0,
            reserved_size=1,
            max_size=5,
        )

        # 创建 AJA 输入操作符，从 AJA 设备捕获实时视频流
        source = AJASourceOp(self, name="aja", **self.kwargs("aja"))

        # 计算格式转换内存块大小：1280x720 分辨率，单通道灰度图像
        drop_alpha_block_size = 1280 * 720 * n_channels * bpp
        drop_alpha_num_blocks = 2  # 内存块数量

        # 创建格式转换操作符，将图像转换为灰度格式
        drop_alpha_channel = FormatConverterOp(
            self,
            name="drop_alpha_channel",
            pool=BlockMemoryPool(
                self,
                storage_type=MemoryStorageType.DEVICE,
                block_size=drop_alpha_block_size,
                num_blocks=drop_alpha_num_blocks,
            ),
            cuda_stream_pool=cuda_stream_pool,
            **self.kwargs("drop_alpha_channel"),
        )

        # 定义 PyTorch 推理函数
        def pytorch_inference(image_tensor):
            """执行 PyTorch 模型推理并打印识别结果"""
            image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)  # 添加批次和通道维度
            image_tensor = image_tensor / 255.0  # 将像素值归一化到 [0, 1]
            with torch.no_grad():  # 禁用梯度计算
                output = self.model(image_tensor)
                predicted_digit = torch.argmax(F.softmax(output, dim=1)).item()
                print(f"识别的数字: {predicted_digit}")  # 将识别的数字输出到终端
                return output

        # 创建可视化操作符，用于实时显示捕获的图像或处理结果
        digit_visualizer = HolovizOp(
            self,
            name="digit_visualizer",  # 操作符名称
            cuda_stream_pool=cuda_stream_pool,
            **self.kwargs("digit_visualizer"),
        )

        # 定义操作符之间的数据流
        self.add_flow(source, digit_visualizer, {("video_buffer_output", "receivers")})
        self.add_flow(source, drop_alpha_channel, {("video_buffer_output", "")})
        self.add_flow(drop_alpha_channel, digit_visualizer, {("", "receivers")})


if __name__ == "__main__":
    # 使用 ArgumentParser 解析命令行参数
    parser = ArgumentParser(description="Handwritten digit recognition demo application.")
    parser.add_argument("-d", "--data", default="none", help="Set the data path for the model")
    args = parser.parse_args()

    # 创建并运行手写数字识别应用程序实例
    app = HandwrittenDigitApp(data=args.data)
    app.config(os.path.join(os.path.dirname(__file__), "digit_recognition_config.yaml"))
    app.run()
