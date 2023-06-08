# YOLO-NAS with Python

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
    <video src="../assets/sample-vid-1.mp4" />
</p>

```bash
python detect.py -m <YOLO-NAS-ONNX-MODEL-PATH> -v <VIDEO-INPUT-PATH>
```

Note: you can pass `0` on `VIDEO-INPUT-PATH` to direct processing from webcam.

## Custom Model

Run custom trained YOLO-NAS model.

1. Generate custom model metadata.

   Generate custom model metadata to provide additional information. Use [custom-nas-model-metadata.py](https://gist.github.com/Hyuto/f3db1c0c2c36308284e101f441c2555f) to generate metadata from torch model.

   **Usage**

   ```bash
   python custom-nas-model-metadata.py -m <CHECKPOINT-PATH> \ # Custom trained YOLO-NAS checkpoint path
                                          -t <MODEL-TYPE> \ # Custom trained YOLO-NAS model type
                                          -n <NUM-CLASSES> # Number of classes
   ```

   After running that it'll generate metadata (json formated) for you

2. Do inferencing with `detect.py`

   Do inferencing with `detect.py` as usual with additional `--custom-metadata` argument to pass the path metadata generated from previous step.

   **Example**

   ```bash
   python detect.py -m <YOLO-NAS-ONNX-MODEL-PATH> -i <IMAGE-INPUT-PATH> --custom-metadata <PATH-TO-METADATA>
   ```

## Run With GPU

Run ONNXRUNTIME or OpenCV DNN with GPU. For ONNXRUNTIME backend if you want to run with GPU you need to install `onnxruntime-gpu` with the same version as your `onnxruntime` lib. OpenCV DNN need more long way to go for GPU inference, you need to build it from the source and enable CUDA.
