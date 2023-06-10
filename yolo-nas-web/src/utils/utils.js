import COCO_LABELS from "./labels.json";

export class Configs {
  baseModelURL = `${process.env.PUBLIC_URL}/model`;
  prepSteps = [
    { DetLongMaxRescale: null },
    { CenterPad: { pad_value: 114 } },
    { Standardize: { max_value: 255.0 } },
  ]; // default YOLO-NAS preprocessing steps
  labels = COCO_LABELS;

  constructor(inputShape, scoreThresh, iouThresh, topk, customMetadata = null) {
    this.inputShape = inputShape;
    this.scoreThresh = scoreThresh;
    this.iouThresh = iouThresh;
    this.topk = topk;
    this.customMetadata = customMetadata;
  }

  async _loadMetadata() {
    const res = await fetch(`${this.baseModelURL}/${this.customMetadata}`);
    const metadata = await res.json();

    this.inputShape = metadata["original_insz"];
    this.scoreThresh = metadata["score_thres"];
    this.iouThresh = metadata["iou_thres"];
    this.prepSteps = metadata["prep_steps"];
    this.labels = metadata["labels"];
  }

  async init() {
    if (this.customMetadata) await this._loadMetadata();
  }
}

export const download = (url, logger = null) => {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("GET", url, true);
    request.responseType = "arraybuffer";
    if (logger) {
      const [log, setState] = logger;
      request.onprogress = (e) => {
        const progress = (e.loaded / e.total) * 100;
        setState({ text: log, progress: progress.toFixed(2) });
      };
    }
    request.onload = function () {
      if (this.status >= 200 && this.status < 300) {
        resolve(request.response);
      } else {
        reject({
          status: this.status,
          statusText: request.statusText,
        });
      }
      resolve(request.response);
    };
    request.onerror = function () {
      reject({
        status: this.status,
        statusText: request.statusText,
      });
    };
    request.send();
  });
};
