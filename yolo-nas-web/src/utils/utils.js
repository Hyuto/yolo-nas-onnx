export class Configs {
  baseModelURL = `${process.env.PUBLIC_URL}/model`;

  constructor(inputShape, scoreThresh, iouThresh, topk, customMetadata = null) {
    this.inputShape = inputShape;
    this.scoreThresh = scoreThresh;
    this.iouThresh = iouThresh;
    this.topk = topk;
    this.customMetadata = customMetadata;
  }

  async _loadMetadata() {
    const res = await fetch(`${this.baseModelURL}/${this.customMetadata}`);
    this.metadata = await res.json();

    this.inputShape = this.metadata["original_insz"];
    this.scoreThresh = this.metadata["score_thres"];
    this.iouThresh = this.metadata["iou_thres"];
  }

  init() {
    if (this.customMetadata) this._loadMetadata();
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
