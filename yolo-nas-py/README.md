# YOLO-NAS with Python

Inference yolo-nas onnx model using `onnxruntime` or opencv `dnn`.

**Inference on Image**

```bash
python detect.py -m <YOLO-NAS-ONNX-MODEL-PATH> -i <IMAGE-INPUT-PATH>
```

**Inference on Video**

```bash
python detect.py -m <YOLO-NAS-ONNX-MODEL-PATH> -v <VIDEO-INPUT-PATH>
```

Note: you can pass `0` on `VIDEO-INPUT-PATH` to direct processing from webcam.