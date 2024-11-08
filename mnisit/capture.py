import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from queue import Queue
from holoscan.core import Application, Operator
from holoscan.operators import AJASourceOp, HolovizOp

# Initialize a global frames queue
frames_queue = Queue(maxsize=10)

# 使用预训练的手写数字识别模型
class PretrainedDigitRecognizer(torch.nn.Module):
    def __init__(self):
        super(PretrainedDigitRecognizer, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(64 * 7 * 7, 128)
        self.fc2 = torch.nn.Linear(128, 10)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 自定义数据集类，用于接收帧
class VideoStreamDataset(Dataset):
    def __init__(self, frames_queue):
        self.frames_queue = frames_queue

    def __len__(self):
        return 1000000

    def __getitem__(self, idx):
        if not self.frames_queue.empty():
            frame = self.frames_queue.get()
            # 将帧转换为灰度图像，并调整大小为 28x28
            frame_gray = np.mean(frame, axis=2)
            frame_resized = np.resize(frame_gray, (28, 28))
            tensor_frame = torch.tensor(frame_resized, dtype=torch.float32).unsqueeze(0) / 255.0
            return tensor_frame
        else:
            return torch.zeros((1, 28, 28), dtype=torch.float32)

# 自定义 Operator 类，用于处理帧数据
class FrameProcessorOp(Operator):
    def __init__(self, app, frames_queue):
        super().__init__(app, "frame_processor")
        self.frames_queue = frames_queue

    def setup(self, spec):
        # 设置输入端口
        spec.input("input_frame")

    def compute(self, op_input, op_output, context):
        # 从输入端口获取帧数据
        frame = op_input.receive("input_frame")
        # 将帧存入队列
        self.frames_queue.put(frame)

# Holoscan 应用程序类
class AJACaptureApp(Application):
    def __init__(self, frames_queue):
        super().__init__()
        self.frames_queue = frames_queue

    def compose(self):
        width = 1280
        height = 720

        # 配置 AJASourceOp
        source = AJASourceOp(
            self,
            name="aja",
            width=width,
            height=height,
            rdma=True,
            enable_overlay=False,
            overlay_rdma=True,
        )

        # 创建自定义的 FrameProcessorOp
        frame_processor = FrameProcessorOp(self, self.frames_queue)
        
        # 使用 add_flow() 连接 AJASourceOp 和 FrameProcessorOp
        self.add_flow(source, frame_processor, {("video_buffer_output", "input_frame")})

        # 创建数据加载器
        dataset = VideoStreamDataset(self.frames_queue)
        data_loader = DataLoader(dataset, batch_size=4, shuffle=False)

        # 添加函数来展示数据加载器中的内容
        def show_data_loader_contents(data_loader, num_batches=5):
            print("Showing contents of the data loader:")
            for i, batch in enumerate(data_loader):
                print(f"Batch {i+1}:")
                print(batch)
                if i >= num_batches - 1:
                    break

        # 调用函数展示内容
        show_data_loader_contents(data_loader)

        # 加载预训练模型
        model = PretrainedDigitRecognizer()
        model.load_state_dict(torch.load("pretrained_mnist_model.pth"))
        model.eval()

        # 实时推理
        with torch.no_grad():
            for batch in data_loader:
                outputs = model(batch)
                _, predicted = torch.max(outputs, 1)
                print(f"Predicted Digits: {predicted.numpy()}")

# 主函数，进行实时视频识别
def main():
    # 创建应用程序实例
    app = AJACaptureApp(frames_queue)
    app.run()  # 运行 Holoscan 应用程序，获取并处理视频流

if __name__ == "__main__":
    main()
