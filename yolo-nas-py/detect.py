import numpy as np
import cv2

from yolo_nas import preprocess, postprocess
from yolo_nas.models import ORT_LOADER, DNN_LOADER
from yolo_nas.cli import parse_opt


def main(opt):
    net = DNN_LOADER(opt.model, opt.gpu) if opt.dnn else ORT_LOADER(opt.model, opt.gpu)

    if opt.image:
        img = cv2.imread(opt.image)
        input_, ratios = preprocess(img, input_size=opt.imgsz)
        outputs = net.forward(input_)
        postprocess(outputs, ratios, opt.score_tresh, opt.iou_tresh, opt.topk)
    elif opt.video:
        pass


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

    """ providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]  # use cuda if gpu is available
        if ort.get_device() == "GPU"
        else ["CPUExecutionProvider"]
    )  # get providers
    net = ort.InferenceSession("yolo_nas_s.onnx", providers=providers)  # load session

    img = cv2.imread("sugamo-shoping-street-s678745378.jpg")
    source_height, source_width, _ = img.shape
    max_size = max(source_width, source_height)  # get max size
    source_padded = np.zeros((max_size, max_size, 3), dtype=np.uint8)  # initial zeros mat
    source_padded[:source_height, :source_width] = img.copy()  # place original image

    ## ratios
    x_ratio = max_size / 640
    y_ratio = max_size / 640

    # run model
    input_img = cv2.dnn.blobFromImage(
        source_padded,
        1 / 255.0,
        (640, 640),
        swapRB=False,
        crop=False,
    )  # normalize and resize: [h, w, 3] => [1, 3, h, w]
    boxes, raw_scores = net.run(None, {"input.1": input_img})

    # box preprocessing
    boxes[0, :, 2] = (boxes[0, :, 2] - boxes[0, :, 0]) * x_ratio
    boxes[0, :, 3] = (boxes[0, :, 3] - boxes[0, :, 1]) * y_ratio
    boxes[0, :, 0] *= x_ratio
    boxes[0, :, 1] *= y_ratio

    boxes = np.squeeze(boxes, 0)
    scores = raw_scores.max(axis=2).flatten()
    classes = np.argmax(raw_scores, axis=2).flatten()

    selected = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, top_k=100)

    for i in selected:
        box = boxes[i, :].astype(np.int32).flatten()
        score = scores[i]
        label = classes[i]

        cv2.rectangle(img, box, colors(label, bgr=True), 2)  # draw box
        (label_width, label_height), _ = cv2.getTextSize(
            f"{labels[label]}",
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            1,
        )
        cv2.rectangle(
            img,
            (box[0] - 1, box[1] - label_height - 6),
            (box[0] + label_width + 1, box[1]),
            colors(label, bgr=True),
            -1,
        )
        cv2.putText(
            img,
            f"{labels[label]}",
            (box[0], box[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            [255, 255, 255],
            1,
        )

    # cv2.imwrite("res.jpg", img)
    cv2.imshow("test", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
 """
