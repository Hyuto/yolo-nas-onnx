import cv from "@techstark/opencv-js";

export class PreProcessing {
  constructor(steps, output_shape) {
    this.steps = steps;
    this.output_shape = output_shape;
  }

  __rescale_img(img, dst, out_shape) {
    const size = new cv.Size(...out_shape);
    cv.resize(img, dst, size, 0, 0, cv.INTER_LINEAR);
  }

  _standarize(img, dst, max_value) {
    const maxMat = new cv.Mat(
      img.rows,
      img.cols,
      cv.CV_32FC3,
      new cv.Scalar(max_value, max_value, max_value)
    );
    cv.divide(img, maxMat, dst, 1, cv.CV_32FC3);
    maxMat.delete();
    return null;
  }

  _normalize(img, dst, mean, std) {
    const meanMat = new cv.Mat(img.rows, img.cols, cv.CV_32FC3, new cv.Scalar(...mean));
    const stdMat = new cv.Mat(img.rows, img.cols, cv.CV_32FC3, new cv.Scalar(...std));
    cv.subtract(img, meanMat, dst, new cv.Mat(), cv.CV_32FC3);
    cv.divide(dst, stdMat, dst, 1, cv.CV_32FC3);

    meanMat.delete();
    stdMat.delete();
    return null;
  }
}
