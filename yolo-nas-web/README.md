# YOLO-NAS with onnxruntime-web

<p align="center">
  <img src="./sample.png" />
</p>

![love](https://img.shields.io/badge/Made%20with-🖤-white)
![react](https://img.shields.io/badge/React-blue?logo=react)
![onnxruntime-web](https://img.shields.io/badge/onnxruntime--web-white?logo=onnx&logoColor=black)
![opencv.js](https://img.shields.io/badge/opencv.js-green?logo=opencv)

---

Object Detection application right in your browser.
Serving YOLO-NAS in browser using onnxruntime-web with `wasm` backend.

## Setup

```bash
yarn install # Install dependencies
```

**Copy YOLO-NAS to model directory**

1. Copy YOLO-NAS ONNX model to `./public/model`
2. Update `modelName` in `App.jsx` to new model name
   ```jsx
   ...
   // configs
   const modelName = "<YOLO-NAS-MODELS>.onnx"; // change to new model name
   const modelInputShape = [1, 3, 640, 640];
   const topk = 100;
   const iouThreshold = 0.4;
   const scoreThreshold = 0.2;
   ...
   ```
3. Done! 😊

**Note: Custom Trained YOLO-NAS Models**

Please update `src/utils/labels.json` with your custom YOLO-NAS classes.

## Scripts

```bash
yarn start # Start dev server
yarn build # Build for productions
```

## Additional Models

**NMS**

ONNX model to perform NMS operator [CUSTOM].

[![nms-yolo-nas.onnx](https://img.shields.io/badge/nms--yolo--nas.onnx-black?logo=onnx)](https://netron.app/?url=https://raw.githubusercontent.com/Hyuto/yolo-nas-onnx/master/yolo-nas-web/public/model/nms-yolo-nas.onnx)

## Reference

- https://github.com/Deci-AI/super-gradients
- https://github.com/Hyuto/yolov8-onnxruntime-web
