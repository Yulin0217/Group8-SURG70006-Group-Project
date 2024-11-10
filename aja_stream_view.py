import os
from argparse import ArgumentParser

# 导入Holoscan框架的核心类和操作符
from holoscan.core import Application
from holoscan.operators import (
    AJASourceOp,  # AJA输入操作符，用于从AJA设备捕获视频流
    FormatConverterOp,  # 格式转换操作符，用于将图像格式转换为模型可接受的格式
    InferenceOp,  # 推理操作符，用于执行模型推理
    HolovizOp,  # 可视化操作符，用于显示推理结果
    SegmentationPostprocessorOp,  # 假设用于分割模型的后处理操作符
)
from holoscan.resources import BlockMemoryPool, CudaStreamPool, MemoryStorageType

# 定义一个自定义的应用程序类，继承自Holoscan的Application类
class CustomModelApp(Application):
    def __init__(self, data="none", video_device="none"):
        """初始化自定义模型应用程序"""

        super().__init__()  # 调用父类的构造函数

        # 设置应用程序的名称
        self.name = "AJA Application"

        # 如果未指定数据路径，使用默认路径
        if data == "none":
            data = os.path.join(os.environ.get("HOLOHUB_DATA_PATH", "../data"), "custom_model_data")

        # 存储数据路径和视频设备路径
        self.sample_data_path = data
        self.video_device = video_device

    # 定义应用程序的数据流和操作符
    def compose(self):
        # 创建一个设备内存池，用于管理内存分配
        pool = BlockMemoryPool(
            self,
            storage_type=MemoryStorageType.DEVICE,  # 内存存储类型为设备内存
            block_size=16 * 1024 * 1024,  # 每个内存块大小为16MB
            num_blocks=4,  # 内存池包含4个块
        )
        # 创建一个CUDA流池，用于管理并发CUDA计算
        stream_pool = CudaStreamPool(
            self,
            name="stream_pool",  # CUDA流池的名称
            dev=0,  # 使用第一个GPU设备
            stream_flags=0,  # CUDA流标志
            num_streams=4,  # CUDA流的数量
        )

        # 输入数据类型设置为RGBA8888，因为AJA设备输出此格式
        in_dtype = "rgba8888"

        # 配置AJA输入操作符，用于捕获视频流数据
        aja_args = self.kwargs("aja")  # 从配置文件或其他地方获取AJA参数
        if self.video_device != "none":  # 如果指定了视频设备路径，更新参数
            aja_args["device"] = self.video_device
        # 创建AJA输入操作符实例
        source = AJASourceOp(
            self,
            name="aja",  # 操作符的名称
            allocator=pool,  # 内存池分配器
            **aja_args,  # 传递AJA参数
        )
        source_output = "signal"  # 指定AJA操作符的输出信号名称

        # 配置图像格式转换操作符，将RGBA8888转换为模型可接受的格式
        preprocessor_args = self.kwargs("preprocessor")  # 获取预处理参数
        preprocessor = FormatConverterOp(
            self,
            name="preprocessor",  # 操作符的名称
            pool=pool,  # 内存池分配器
            in_dtype=in_dtype,  # 输入数据类型
            **preprocessor_args,  # 传递预处理参数
        )

        # 配置推理操作符，用于运行自定义模型
        inference_args = self.kwargs("inference")  # 获取推理参数
        # 设置模型路径映射，加载ONNX格式的自定义模型
        inference_args["model_path_map"] = {
            "custom_model": os.path.join(self.sample_data_path, "your_custom_model.onnx")
        }
        # 创建推理操作符实例
        inference = InferenceOp(
            self,
            name="inference",  # 操作符的名称
            allocator=pool,  # 内存池分配器
            cuda_stream_pool=stream_pool,  # 使用CUDA流池
            **inference_args,  # 传递推理参数
        )

        # 配置后处理操作符，假设用于处理分割模型的输出
        postprocessor_args = self.kwargs("postprocessor")  # 获取后处理参数
        postprocessor_args["image_width"] = preprocessor_args["resize_width"]  # 设置图像宽度
        postprocessor_args["image_height"] = preprocessor_args["resize_height"]  # 设置图像高度
        # 创建后处理操作符实例
        postprocessor = SegmentationPostprocessorOp(
            self,
            name="postprocessor",  # 操作符的名称
            allocator=pool,  # 内存池分配器
            **postprocessor_args,  # 传递后处理参数
        )

        # 配置可视化操作符，用于显示推理和后处理的结果
        holoviz = HolovizOp(
            self,
            allocator=pool,  # 内存池分配器
            name="holoviz",  # 操作符的名称
            **self.kwargs("holoviz"),  # 获取可视化参数
        )

        # 定义操作符之间的数据流
        self.add_flow(source, holoviz, {(source_output, "receivers")})  # AJA输入连接到可视化
        self.add_flow(source, preprocessor)  # AJA输入连接到预处理器
        self.add_flow(preprocessor, inference, {("", "receivers")})  # 预处理器连接到推理
        self.add_flow(inference, postprocessor, {("transmitter", "in")})  # 推理连接到后处理器
        self.add_flow(postprocessor, holoviz, {("out", "receivers")})  # 后处理器连接到可视化

# 主函数
if __name__ == "__main__":
    # 使用ArgumentParser解析命令行参数
    parser = ArgumentParser(description="AJA Model Application.")
    # 添加数据路径参数
    parser.add_argument(
        "-d", "--data", default="none", help="Set the data path for the model"
    )
    # 添加视频设备参数
    parser.add_argument(
        "-v", "--video_device", default="none", help="Specify the AJA video device"
    )
    args = parser.parse_args()  # 解析命令行参数

    # 创建并运行自定义模型应用程序实例
    app = CustomModelApp(data=args.data, video_device=args.video_device)
    app.run()  # 运行应用程序
