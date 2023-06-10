import cv from "@techstark/opencv-js";
import { Tensor } from "onnxruntime-web";
import { renderBoxes } from "./renderBox";

/**
 * Detect Image
 * @param {HTMLImageElement} image Image to detect
 * @param {HTMLCanvasElement} canvas canvas to draw boxes
 * @param {ort.InferenceSession} session YOLO-NAS onnxruntime session
 * @param {Number} topk Integer representing the maximum number of boxes to be selected per class
 * @param {Number} iouThreshold Float representing the threshold for deciding whether boxes overlap too much with respect to IOU
 * @param {Number} scoreThreshold Float representing the threshold for deciding when to remove boxes based on score
 * @param {Number[]} inputShape model input shape. Normally in YOLO model [batch, channels, width, height]
 */
export const detectImage = async (image, canvas, session, { preProcessing, postProcessing }) => {
  canvas.width = image.width;
  canvas.height = image.height;

  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height); // clean canvas
  ctx.drawImage(image, 0, 0, image.width, image.height);

  const img = cv.imread(image);
  const [input, metadata] = preProcessing.run(img);

  const tensor = new Tensor("float32", input.data32F, session.inputShape); // to ort.Tensor
  const config = new Tensor(
    "float32",
    new Float32Array([
      postProcessing.topk, // topk per class
      postProcessing.iouThresh, // iou threshold
      postProcessing.scoreThresh, // score threshold
    ])
  ); // nms config tensor
  const outNames = session.net.outputNames;
  const output = await session.net.run({ "input.1": tensor }); // run session and get output layer
  const { selected } = await session.nms.run({
    bboxes: output[outNames[0]],
    scores: output[outNames[1]],
    config: config,
  }); // perform nms and filter boxes

  const boxes = [];

  // looping through output
  for (let idx = 0; idx < selected.dims[1]; idx++) {
    const data = selected.data.slice(idx * selected.dims[2], (idx + 1) * selected.dims[2]); // get rows
    const [box, score, label] = postProcessing.run(data, [...metadata]);

    boxes.push({
      label: label,
      probability: score,
      bounding: box, // upscale box
    }); // update boxes to draw later
  }

  renderBoxes(ctx, boxes, postProcessing.labels); // Draw boxes
  input.delete(); // delete unused Mat
};
