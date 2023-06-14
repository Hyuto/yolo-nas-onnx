# YOLO-NAS ONNX with Python

<p align="center">
    <img src="../assets/sample-1.jpg" alt="sample" />
</p>

---

Inference yolo-nas onnx model using `onnxruntime` or opencv `dnn`.

**Inference on Image**

```bash
python detect.py -m <YOLO-NAS-ONNX-MODEL-PATH> -i <IMAGE-INPUT-PATH>
```

**Inference on Video**

<p align="center">
    <video src="../assets/sample-vid-1.mp4"></video>
</p>

```bash
python detect.py -m <YOLO-NAS-ONNX-MODEL-PATH> -v <VIDEO-INPUT-PATH>
```

Note: you can pass `0` on `VIDEO-INPUT-PATH` to direct processing from webcam.

## Custom Trained YOLO-NAS Models

Run custom trained YOLO-NAS model.

1. Generate custom model metadata.

   Generate custom model metadata to provide additional information from torch model.
   Please follow these steps to generate it [Generate Custom Metadata](https://github.com/Hyuto/yolo-nas-onnx#custom-model).

2. Do inferencing with `detect.py`

   Do inferencing with `detect.py` as usual with additional `--custom-metadata` argument to pass the path metadata generated from previous step.

   **Example**

   ```bash
   python detect.py -m <YOLO-NAS-ONNX-MODEL-PATH> -i <IMAGE-INPUT-PATH> --custom-metadata <PATH-TO-METADATA>
   ```

## Run With GPU

Run ONNXRUNTIME or OpenCV DNN with GPU. For ONNXRUNTIME backend if you want to run with GPU you need to install `onnxruntime-gpu` with the same version as your `onnxruntime` lib. OpenCV DNN need more long way to go for GPU inference, you need to build it from the source and enable CUDA.
