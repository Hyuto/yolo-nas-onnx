import numpy as np
import cv2
import onnxruntime as ort

from .utils import log_warning


class ORT_LOADER:
    """ONNXRUNTIME model handler"""

    def __init__(self, path: str, gpu: bool = False):
        self._load_model(path, gpu)

    def _load_model(self, path: str, use_gpu: bool):
        """Load model and get model input and output information"""
        is_gpu_available = ort.get_device() == "GPU"
        if not is_gpu_available and use_gpu:
            log_warning("GPU", "GPU is not available, using CPU to process.")
            use_gpu = False

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]  # use cuda if gpu is available
            if use_gpu
            else ["CPUExecutionProvider"]  # use CPU
        )  # get providers
        self.net = ort.InferenceSession(path, providers=providers)  # load session

        net_input = self.net.get_inputs()[0]  # get input info
        self.input_name = net_input.name
        self.input_shape = net_input.shape
        self.output_names = [x.name for x in self.net.get_outputs()]  # get output info

    def assert_input_shape(self, input_shape=None):
        """check input shape"""
        if input_shape:
            if input_shape != self.input_shape:
                log_warning(
                    "Input Shape",
                    "Model have different input shape with the original input shape on metadata! "
                    + "It can lead to lower detection performance.",
                )

    def forward(self, input_):
        """Get model prediction"""
        return self.net.run(self.output_names, {self.input_name: input_})

    def warmup(self):
        "Warming up model"
        for _ in range(3):
            dummy = np.random.rand(*self.input_shape).astype(np.float32)
            _ = self.forward(dummy)


class DNN_LOADER:
    """OPENCV DNN model handler"""

    def __init__(self, path: str, gpu: bool = False):
        self.net = cv2.dnn.readNet(path)  # overload net

        # get input and output info from ort
        net_ort = ORT_LOADER(path)
        self.input_name = net_ort.input_name
        self.input_shape = net_ort.input_shape
        self.output_names = net_ort.output_names

        is_gpu_available = True if cv2.cuda.getCudaEnabledDeviceCount() else False
        if not is_gpu_available and gpu:
            log_warning("GPU", "GPU is not available, using CPU to process.")
            gpu = False

        if gpu:  # use CUDA if available
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:  # use CPU
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def assert_input_shape(self, input_shape=None):
        """check input shape"""
        if input_shape:
            if input_shape != self.input_shape:
                log_warning(
                    "Input Shape",
                    "Model have different input shape with the original input shape on metadata! "
                    + "It can lead to lower detection performance.",
                )

    def forward(self, input_):
        """Get model prediction"""
        self.net.setInput(input_, self.input_name)
        return self.net.forward(self.output_names)

    def warmup(self):
        "Warming up model"
        for _ in range(3):
            dummy = np.random.rand(*self.input_shape).astype(np.float32)
            _ = self.forward(dummy)


def load_net(path, gpu=False, dnn=False):
    """Load NET

    Args:
        path (str): YOLO-NAS ONNX model path
        gpu (bool, optional): Using GPU?. Defaults to False.
        dnn (bool, optional): Use OpenCV DNN?. Defaults to False.

    Returns:
        Optional[ORT_LOADER, DNN_LOADER]: Net loader
    """
    if dnn:  # if using OpenCV DNN
        return DNN_LOADER(path, gpu)
    return ORT_LOADER(path, gpu)
