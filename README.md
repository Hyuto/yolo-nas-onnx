# YOLO-NAS ONNX

<p align="center">
    <img src="./assets/sample-4.jpg" alt="sample" />
</p>

**_Image Source_**: https://www.pinterest.com/pin/784752303797219490/

---

Run YOLO-NAS models with ONNX **without using Pytorch**. Inferencing YOLO-NAS ONNX models with ONNXRUNTIME or OpenCV DNN.

## Generate ONNX Model

Generate YOLO-NAS ONNX model. You can convert the model using the following code after installing `super_gradients` library.

```python
from super_gradients.training import models

net = models.get("yolo_nas_s", pretrained_weights="coco")
models.convert_to_onnx(model=net, input_shape=(3,640,640), out_path="yolo_nas_s.onnx")
```

## References

- https://github.com/Deci-AI/super-gradients
