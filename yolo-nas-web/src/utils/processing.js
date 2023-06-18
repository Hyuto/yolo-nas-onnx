import cv from "@techstark/opencv-js";

/**
 * Preprocessing class
 * Handle preprocessing steps on image before running model inference
 * @param {Object[]} steps Preprocessing steps to do.
 * @param {Number[]} output_shape Image output shape after preprocessing phase.
 */
export class PreProcessing {
  constructor(steps, output_shape) {
    this.steps = steps;
    this.output_shape = output_shape;
  }

  /**
   * Default func to rescale image
   * @param {cv.Mat} img Image/source mat
   * @param {cv.Mat} dst Destination Mat to store result
   * @param {Number[]} out_shape Image size to rescale
   */
  __rescale_img(img, dst, out_shape) {
    const size = new cv.Size(...out_shape);
    cv.resize(img, dst, size, 0, 0, cv.INTER_LINEAR);
  }

  /**
   * Standardize image corresponding to max_value
   * @param {cv.Mat} img Image/source mat
   * @param {cv.Mat} dst Destination Mat to store result
   * @param {Number} max_value Number representing max value
   * @returns metadata
   */
  _standarize(img, dst, { max_value }) {
    img.convertTo(dst, cv.CV_32F, 1.0 / max_value, 0);
    return null;
  }

  /**
   * Normalize image corresponding to mean and stdev
   * @param {cv.Mat} img Image/source mat
   * @param {cv.Mat} dst Destination Mat to store result
   * @param {Number} mean Number representing mean
   * @param {Number} std Number representing stdev
   * @returns metadata
   */
  _normalize(img, dst, { mean, std }) {
    const meanMat = new cv.Mat(img.rows, img.cols, img.type(), new cv.Scalar(...mean));
    cv.subtract(img, meanMat, dst, new cv.Mat(), cv.CV_32F);
    dst.convertTo(dst, cv.CV_32F, 1.0 / std, 0);

    meanMat.delete();
    return null;
  }

  /**
   * Rescale image to output_shape size
   * @param {cv.Mat} img Image/source mat
   * @param {cv.Mat} dst Destination Mat to store result
   * @returns metadata (scale factors)
   */
  _det_rescale(img, dst) {
    const scale_factors = [this.output_shape[1] / img.rows, this.output_shape[0] / img.cols];
    this.__rescale_img(img, dst, this.output_shape);

    return { scale_factors: scale_factors };
  }

  /**
   * Rescale image to output_shape based on minimum scale factor
   * @param {cv.Mat} img Image/source mat
   * @param {cv.Mat} dst Destination Mat to store result
   * @returns metadata (scale factors)
   */
  _det_long_max_rescale(img, dst) {
    const scale_factor = Math.min(
      (this.output_shape[1] - 4) / img.rows,
      (this.output_shape[0] - 4) / img.cols
    );

    if (scale_factor !== 1.0) {
      const new_height = Math.round(img.rows * scale_factor);
      const new_width = Math.round(img.cols * scale_factor);
      this.__rescale_img(img, dst, [new_width, new_height]);
    }

    return { scale_factors: [scale_factor, scale_factor] };
  }

  /**
   * Padding bottom and right part of image
   * @param {cv.Mat} img Image/source mat
   * @param {cv.Mat} dst Destination Mat to store result
   * @param {Number} pad_value Number to fill the pad
   * @returns metadata (scale factors)
   */
  _bot_right_pad(img, dst, { pad_value }) {
    const pad_height = this.output_shape[1] - img.rows;
    const pad_width = this.output_shape[0] - img.cols;
    cv.copyMakeBorder(
      img,
      dst,
      0,
      pad_height,
      0,
      pad_width,
      cv.BORDER_CONSTANT,
      new cv.Scalar(pad_value, pad_value, pad_value)
    );

    return { padding: [0, pad_height, 0, pad_width] };
  }

  /**
   * Place image to the center and padding the rest part
   * @param {cv.Mat} img Image/source mat
   * @param {cv.Mat} dst Destination Mat to store result
   * @param {Number} pad_value Number to fill the pad
   * @returns metadata (scale factors)
   */
  _center_pad(img, dst, { pad_value }) {
    const pad_height = this.output_shape[1] - img.rows;
    const pad_width = this.output_shape[0] - img.cols;
    const pad_top = Math.floor(pad_height / 2);
    const pad_left = Math.floor(pad_width / 2);
    cv.copyMakeBorder(
      img,
      dst,
      pad_top,
      pad_height - pad_top,
      pad_left,
      pad_width - pad_left,
      cv.BORDER_CONSTANT,
      new cv.Scalar(pad_value, pad_value, pad_value)
    );
    return { padding: [pad_top, pad_height - pad_top, pad_left, pad_width - pad_left] };
  }

  /**
   * Call preprocessing func by it's name
   * @param {String} name name of the step
   * @returns preprocessing step
   */
  _call_fn(name) {
    const mapper = {
      Standardize: this._standarize.bind(this),
      DetRescale: this._det_rescale.bind(this),
      DetLongMaxRescale: this._det_long_max_rescale.bind(this),
      BotRightPad: this._bot_right_pad.bind(this),
      CenterPad: this._center_pad.bind(this),
      Normalize: this._normalize.bind(this),
    };
    return mapper[name];
  }

  /**
   * Run preprocessing based on given steps on constructor
   * @param {cv.Mat} img Source Mat
   * @returns Ready to use cv.Mat and preprocessing metadata.
   */
  run(img) {
    const inputImage = new cv.Mat(img.rows, img.cols, cv.CV_32FC3); // initiate dst mat
    cv.cvtColor(img, inputImage, cv.COLOR_RGBA2BGR); // RGBA to BGR

    const metadata = [];
    // loop through preprocessing steps
    this.steps.forEach((st) => {
      // if step isn't null
      if (st) {
        const [name, kwargs] = Object.entries(st)[0]; // name and kwargs
        // call preprocessing step function
        const meta = kwargs
          ? this._call_fn(name)(inputImage, inputImage, kwargs)
          : this._call_fn(name)(inputImage, inputImage);
        metadata.push(meta); // append metadata
      }
    });

    const input_ = cv.blobFromImage(inputImage, 1, new cv.Size(), new cv.Scalar(), true, false); // image to blob [1, c, h, w] (RGB)
    inputImage.delete(); // free memory
    return [input_, metadata];
  }
}

/**
 * Postprocessing class
 * Handle postprocessing on boxes before rendering outputs
 * @param {Object[]} steps Preprocessing steps to do.
 * @param {Number} iouThresh Float representing the threshold for deciding whether boxes overlap too much with respect to IOU.
 * @param {Number} scoreThresh Float representing the threshold for deciding when to remove boxes.
 * @param {Number} topk TopK.
 * @param {String[]} labels Array of string represent labels of the model.
 */
export class PostProcessing {
  constructor(steps, iouThresh, scoreThresh, topk, labels) {
    this.steps = steps;
    this.iouThresh = iouThresh;
    this.scoreThresh = scoreThresh;
    this.topk = topk;
    this.labels = labels;
  }

  /**
   * Rescale boxes respect to scale factors
   * @param {Number[4]} box Box to rescale
   * @param {Object} metadata metadata object contain scale factors
   * @returns Rescaled box
   */
  _rescale_boxes(box, metadata) {
    const [scale_factors_w, scale_factors_h] = metadata.scale_factors;
    box[0] /= scale_factors_w;
    box[2] /= scale_factors_w;
    box[1] /= scale_factors_h;
    box[3] /= scale_factors_h;
    return box;
  }

  /**
   * Shift box because of padding process in preprocessing steps.
   * @param {Number[4]} box Box to rescale
   * @param {Object} metadata metadata object contain scale factors
   * @returns Shifted box
   */
  _shift_bboxes(box, metadata) {
    box[0] -= metadata.padding[2];
    box[2] -= metadata.padding[2];
    box[1] -= metadata.padding[0];
    box[3] -= metadata.padding[0];
    return box;
  }

  /**
   * Call postprocessing func by it's name
   * @param {String} name name of the step
   * @returns postprocessing step
   */
  _call_fn(name) {
    const mapper = {
      DetRescale: this._rescale_boxes.bind(this),
      DetLongMaxRescale: this._rescale_boxes.bind(this),
      BotRightPad: this._shift_bboxes.bind(this),
      CenterPad: this._shift_bboxes.bind(this),
      Standardize: null,
      Normalize: null,
    };
    return mapper[name];
  }

  /**
   * Run postprocessing based on given steps on constructor
   * @param {Float32Array} output_row A row of output from model inference
   * @returns Ready to use box, score, and label
   */
  run(output_row, metadata) {
    let box = output_row.slice(0, 4); // slice box from output row [index 0 to 3]
    let scores = output_row.slice(4); // slice scores from output row [index 4 to end]

    // loop through reversed processing steps
    this.steps.toReversed().forEach((st) => {
      // if step isn't null
      if (st) {
        const name = Object.entries(st)[0][0]; // get step name
        const meta = metadata.pop(); // get preprocessing metadata
        // if called function isn't null
        if (this._call_fn(name)) {
          box = this._call_fn(name)(box, meta); // process box
        }
      }
    });

    // xyxy to xywh
    box[2] -= box[0];
    box[3] -= box[1];

    const score = Math.max(...scores); // maximum probability scores
    const label = scores.indexOf(score); // class id of maximum probability scores
    return [box, score, label];
  }
}
