import cv from "@techstark/opencv-js";
import { Tensor } from "onnxruntime-web";
import { renderBoxes } from "./renderBox";
import { PreProcessing } from "./processing";

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
export const detectImage = async (
  image,
  canvas,
  session,
  topk,
  iouThreshold,
  scoreThreshold,
  inputShape
) => {
  const [modelWidth, modelHeight] = inputShape.slice(2);
  const [input, xRatio, yRatio] = preprocessing(image, modelWidth, modelHeight);

  const tensor = new Tensor("float32", input.data32F, inputShape); // to ort.Tensor
  const config = new Tensor(
    "float32",
    new Float32Array([
      topk, // topk per class
      iouThreshold, // iou threshold
      scoreThreshold, // score threshold
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
    const box = data.slice(0, 4);
    const scores = data.slice(4); // classes probability scores
    const score = Math.max(...scores); // maximum probability scores
    const label = scores.indexOf(score); // class id of maximum probability scores

    const [x, y, w, h] = [
      box[0] * xRatio, // upscale left
      box[1] * yRatio, // upscale top
      (box[2] - box[0]) * xRatio, // upscale width
      (box[3] - box[1]) * yRatio, // upscale height
    ]; // keep boxes in maxSize range

    boxes.push({
      label: label,
      probability: score,
      bounding: [x, y, w, h], // upscale box
    }); // update boxes to draw later
  }

  renderBoxes(canvas, boxes); // Draw boxes
  input.delete(); // delete unused Mat
};

/**
 * Preprocessing image
 * @param {HTMLImageElement} source image source
 * @param {Number} modelWidth model input width
 * @param {Number} modelHeight model input height
 * @return preprocessed image and configs
 */
const preprocessing = (source, modelWidth, modelHeight) => {
  try {
    const test = new PreProcessing([], [640, 640]);
    const mat = cv.imread(source); // read from img tag
    const matC3 = new cv.Mat(mat.rows, mat.cols, cv.CV_32FC3); // new image matrix
    cv.cvtColor(mat, matC3, cv.COLOR_RGBA2BGR); // RGBA to BGR
    const testMat = new cv.Mat();
    test._normalize(matC3, testMat, [123.675, 116.28, 103.53], [58.395, 57.12, 57.375]);
    console.log(testMat.data32F);

    // padding image to [n x n] dim
    const maxSize = Math.max(matC3.rows, matC3.cols); // get max size from width and height
    const xPad = maxSize - matC3.cols, // set xPadding
      xRatio = maxSize / matC3.cols; // set xRatio
    const yPad = maxSize - matC3.rows, // set yPadding
      yRatio = maxSize / matC3.rows; // set yRatio
    const matPad = new cv.Mat(); // new mat for padded image
    cv.copyMakeBorder(matC3, matPad, 0, yPad, 0, xPad, cv.BORDER_CONSTANT); // padding black

    const input = cv.blobFromImage(
      matPad,
      1 / 255.0, // normalize
      new cv.Size(modelWidth, modelHeight), // resize to model input size
      new cv.Scalar(0, 0, 0),
      true, // swapRB
      false // crop
    ); // preprocessing image matrix

    // release mat opencv
    mat.delete();
    matC3.delete();
    matPad.delete();

    return [input, xRatio, yRatio];
  } catch (e) {
    console.log(cv.exceptionFromPtr(e).msg);
  }
};
