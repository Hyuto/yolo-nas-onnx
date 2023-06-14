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

  _standarize(img, dst, { max_value }) {
    img.convertTo(dst, cv.CV_32F, 1.0 / max_value, 0);
    return null;
  }

  _normalize(img, dst, { mean, std }) {
    const meanMat = new cv.Mat(img.rows, img.cols, img.type(), new cv.Scalar(...mean));
    cv.subtract(img, meanMat, dst, new cv.Mat(), cv.CV_32F);
    dst.convertTo(dst, cv.CV_32F, 1.0 / std, 0);

    meanMat.delete();
    return null;
  }

  _det_rescale(img, dst) {
    const scale_factors = [this.output_shape[1] / img.rows, this.output_shape[0] / img.cols];
    this.__rescale_img(img, dst, this.output_shape);

    return { scale_factors: scale_factors };
  }

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

  run(img) {
    const inputImage = new cv.Mat(img.rows, img.cols, cv.CV_32FC3);
    cv.cvtColor(img, inputImage, cv.COLOR_RGBA2BGR); // RGBA to BGR

    const metadata = [];
    this.steps.forEach((st) => {
      if (st) {
        const [name, kwargs] = Object.entries(st)[0];
        const meta = kwargs
          ? this._call_fn(name)(inputImage, inputImage, kwargs)
          : this._call_fn(name)(inputImage, inputImage);
        metadata.push(meta);
      }
    });

    const input_ = cv.blobFromImage(inputImage, 1, new cv.Size(), new cv.Scalar(), true, false);
    inputImage.delete();
    return [input_, metadata];
  }
}

export class PostProcessing {
  constructor(steps, iouThresh, scoreThresh, topk, labels) {
    this.steps = steps;
    this.iouThresh = iouThresh;
    this.scoreThresh = scoreThresh;
    this.topk = topk;
    this.labels = labels;
  }

  _rescale_boxes(box, metadata) {
    const [scale_factors_w, scale_factors_h] = metadata.scale_factors;
    box[0] /= scale_factors_w;
    box[2] /= scale_factors_w;
    box[1] /= scale_factors_h;
    box[3] /= scale_factors_h;
    return box;
  }

  _shift_bboxes(box, metadata) {
    box[0] -= metadata.padding[2];
    box[2] -= metadata.padding[2];
    box[1] -= metadata.padding[0];
    box[3] -= metadata.padding[0];
    return box;
  }

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

  run(output_row, metadata) {
    let box = output_row.slice(0, 4);
    let scores = output_row.slice(4);

    this.steps.toReversed().forEach((st) => {
      if (st) {
        const name = Object.entries(st)[0][0];
        const meta = metadata.pop();
        if (this._call_fn(name)) {
          box = this._call_fn(name)(box, meta);
        }
      }
    });

    box[2] -= box[0];
    box[3] -= box[1];

    const score = Math.max(...scores); // maximum probability scores
    const label = scores.indexOf(score); // class id of maximum probability scores
    return [box, score, label];
  }
}
