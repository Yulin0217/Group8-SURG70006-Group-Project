import os
from argparse import ArgumentParser

from holoscan.core import Application
from holoscan.operators import (
    AJASourceOp,
    FormatConverterOp,
    HolovizOp,
    InferenceOp,
    SegmentationPostprocessorOp,
)
from holoscan.resources import BlockMemoryPool, CudaStreamPool, MemoryStorageType


class UltrasoundApp(Application):
    def __init__(self, data):
        """Initialize the ultrasound segmentation application"""

        super().__init__()

        # Set name
        self.name = "Ultrasound App"

        if data == "none":
            data = os.environ.get("HOLOHUB_DATA_PATH", "")

        self.sample_data_path = data

        self.model_path_map = {
            "ultrasound_seg": os.path.join(self.sample_data_path, "us_unet_256x256_nhwc.onnx"),
        }

    def compose(self):
        n_channels = 4  # RGBA
        bpp = 4  # bytes per pixel

        cuda_stream_pool = CudaStreamPool(
            self,
            name="cuda_stream",
            dev_id=0,
            stream_flags=0,
            stream_priority=0,
            reserved_size=1,
            max_size=5,
        )

        source = AJASourceOp(self, name="aja", **self.kwargs("aja"))
        print("source:\n", source)
        drop_alpha_block_size = 1280 * 720 * n_channels * bpp  # Updated to 720p
        drop_alpha_num_blocks = 2
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

        width_preprocessor = 1280  # Updated to 720p width
        height_preprocessor = 720   # Updated to 720p height
        preprocessor_block_size = width_preprocessor * height_preprocessor * n_channels * bpp
        preprocessor_num_blocks = 3
        segmentation_preprocessor = FormatConverterOp(
            self,
            name="segmentation_preprocessor",
            pool=BlockMemoryPool(
                self,
                storage_type=MemoryStorageType.DEVICE,
                block_size=preprocessor_block_size,
                num_blocks=preprocessor_num_blocks,
            ),
            cuda_stream_pool=cuda_stream_pool,
            **self.kwargs("segmentation_preprocessor"),
        )

        n_channels_inference = 2
        width_inference = 256
        height_inference = 256
        bpp_inference = 4
        inference_block_size = (
            width_inference * height_inference * n_channels_inference * bpp_inference
        )
        inference_num_blocks = 2
        segmentation_inference = InferenceOp(
            self,
            name="segmentation_inference_holoinfer",
            backend="trt",
            allocator=BlockMemoryPool(
                self,
                storage_type=MemoryStorageType.DEVICE,
                block_size=inference_block_size,
                num_blocks=inference_num_blocks,
            ),
            model_path_map=self.model_path_map,
            pre_processor_map={"ultrasound_seg": ["source_video"]},
            inference_map={"ultrasound_seg": "inference_output_tensor"},
            in_tensor_names=["source_video"],
            out_tensor_names=["inference_output_tensor"],
            enable_fp16=False,
            input_on_cuda=True,
            output_on_cuda=True,
            transmit_on_cuda=True,
        )
        print("Model:", segmentation_inference)
        print("segmentation_inference_vars:", vars(segmentation_inference))
        postprocessor_block_size = width_inference * height_inference
        postprocessor_num_blocks = 2
        segmentation_postprocessor = SegmentationPostprocessorOp(
            self,
            name="segmentation_postprocessor",
            allocator=BlockMemoryPool(
                self,
                storage_type=MemoryStorageType.DEVICE,
                block_size=postprocessor_block_size,
                num_blocks=postprocessor_num_blocks,
            ),
            **self.kwargs("segmentation_postprocessor"),
        )

        segmentation_visualizer = HolovizOp(
            self,
            name="segmentation_visualizer",
            cuda_stream_pool=cuda_stream_pool,
            **self.kwargs("segmentation_visualizer"),
        )

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
    # Parse args
    parser = ArgumentParser(description="Ultrasound segmentation demo application.")
    parser.add_argument(
        "-d",
        "--data",
        default="none",
        help=("Set the data path"),
    )
    args = parser.parse_args()

    app = UltrasoundApp(data=args.data)
    app.config(os.path.join(os.path.dirname(__file__), "ultrasound_segmentation.yaml"))
    app.run()
