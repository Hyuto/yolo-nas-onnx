from pathlib import Path

import numpy as np
import cv2

from yolo_nas import preprocess, postprocess
from yolo_nas.models import ORT_LOADER, DNN_LOADER
from yolo_nas.draw import draw_box
from yolo_nas.cli import parse_opt
from yolo_nas.utils import Labels


def detect(net, source, input_size, labels):
    net_input = source.copy()
    input_, prep_meta = preprocess(net_input, input_size)
    outputs = net.forward(input_)

    boxes, scores, classes = postprocess(outputs, prep_meta)
    selected = cv2.dnn.NMSBoxes(boxes, scores, opt.score_tresh, opt.iou_tresh)

    for i in selected:
        box = boxes[i, :].astype(np.int32).flatten()
        score = float(scores[i]) * 100
        label, color = labels(classes[i], use_bgr=True)

        draw_box(source, box, label, score, color)
    return source


def main(opt):
    net = DNN_LOADER(opt.model, opt.gpu) if opt.dnn else ORT_LOADER(opt.model, opt.gpu)
    _, _, input_height, input_width = net.input_shape  # [b, c, h, w]
    net.warmup()

    labels = Labels(opt.labels)

    if opt.image:
        img = cv2.imread(opt.image)
        img = detect(net, img, (input_width, input_height), labels)

        if opt.export:
            print("🚀 \033[1m\033[94mExporting Image: \033[0m" + opt.export)
            cv2.imwrite(opt.export, img)

        name = Path(opt.image).stem
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.imshow(name, img)
        cv2.waitKey(0)
    elif opt.video:
        # Video processing
        vid_source = 0 if opt.video == "0" else opt.video
        cap = cv2.VideoCapture(vid_source)

        name = "Webcam" if opt.video == "0" else Path(opt.video).stem
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        print("\033[1m\033[94mProcessing video,\033[0m press 'q' to exit.")

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            frame = detect(net, frame, (input_width, input_height), labels)
            cv2.imshow(name, frame)

            if cv2.waitKey(1) == ord("q"):
                break

        cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
