import React, { useState, useRef } from "react";
import cv from "@techstark/opencv-js";
import { Tensor, InferenceSession } from "onnxruntime-web";
import Loader from "./components/loader";
import { detectImage } from "./utils/detect";
import { Configs, download } from "./utils/utils";
import "./style/App.css";

const App = () => {
  const [session, setSession] = useState(null);
  const [loading, setLoading] = useState({ text: "Loading OpenCV.js", progress: null });
  const [image, setImage] = useState(null);
  const inputImage = useRef(null);
  const imageRef = useRef(null);
  const canvasRef = useRef(null);

  // configs
  const modelName = "test.onnx";
  const configs = new Configs(
    [1, 3, 640, 640], // input shape
    0.25, // score threshold
    0.45, // IOU threshold
    100, // topk
    "custom-yolo_nas_s-metadata.json" // custom metadata
  );

  // wait until opencv.js initialized
  cv["onRuntimeInitialized"] = async () => {
    await configs.init();

    // create session
    const arrBufNet = await download(
      `${configs.baseModelURL}/${modelName}`, // url
      ["Loading YOLO-NAS model", setLoading] // logger
    ); // get model arraybuffer
    const yoloNAS = await InferenceSession.create(arrBufNet);
    const arrBufNMS = await download(
      `${configs.baseModelURL}/nms-yolo-nas.onnx`, // url
      ["Loading NMS model", setLoading] // logger
    ); // get nms model arraybuffer
    const nms = await InferenceSession.create(arrBufNMS);

    // warmup main model
    setLoading({ text: "Warming up model...", progress: null });
    const tensor = new Tensor(
      "float32",
      new Float32Array(configs.inputShape.reduce((a, b) => a * b)),
      configs.inputShape
    );
    await yoloNAS.run({ "input.1": tensor });

    setSession({ net: yoloNAS, nms: nms });
    setLoading(null);
  };

  return (
    <div className="App">
      {loading && (
        <Loader>
          {loading.progress ? `${loading.text} - ${loading.progress}%` : loading.text}
        </Loader>
      )}
      <div className="header">
        <h1>YOLO-NAS Object Detection App</h1>
        <p>
          YOLO-NAS object detection application live on browser powered by{" "}
          <strong>onnxruntime-web</strong>
        </p>
        <p>
          Serving : <code>{modelName}</code>
        </p>
      </div>

      <div className="content">
        <img
          ref={imageRef}
          src="#"
          alt=""
          style={{ display: image ? "block" : "none" }}
          onLoad={() => {
            detectImage(
              imageRef.current,
              canvasRef.current,
              session,
              configs.topk,
              configs.iouThresh,
              configs.scoreThresh,
              configs.inputShape
            );
          }}
        />
        <canvas
          id="canvas"
          width={configs.inputShape[2]}
          height={configs.inputShape[3]}
          ref={canvasRef}
        />
      </div>

      <input
        type="file"
        ref={inputImage}
        accept="image/*"
        style={{ display: "none" }}
        onChange={(e) => {
          // handle next image to detect
          if (image) {
            URL.revokeObjectURL(image);
            setImage(null);
          }

          const url = URL.createObjectURL(e.target.files[0]); // create image url
          imageRef.current.src = url; // set image source
          setImage(url);
        }}
      />
      <div className="btn-container">
        <button
          onClick={() => {
            inputImage.current.click();
          }}
        >
          Open local image
        </button>
        {image && (
          /* show close btn when there is image */
          <button
            onClick={() => {
              inputImage.current.value = "";
              imageRef.current.src = "#";
              URL.revokeObjectURL(image);
              setImage(null);
            }}
          >
            Close image
          </button>
        )}
      </div>
    </div>
  );
};

export default App;
