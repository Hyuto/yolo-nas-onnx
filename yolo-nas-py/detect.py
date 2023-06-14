from pathlib import Path

import numpy as np
import cv2

from yolo_nas.models import load_net
from yolo_nas.processing import Preprocessing, Postprocessing
from yolo_nas.draw import draw_box
from yolo_nas.cli import get_configs
from yolo_nas.utils import Labels, VideoWriter, export_image, log_info


def detect(net, source, pre_process, post_process, labels):
    net_input = source.copy()
    input_, prep_meta = pre_process(net_input)
    outputs = net.forward(input_)

    boxes, scores, classes = post_process(outputs, prep_meta)
    selected = cv2.dnn.NMSBoxes(boxes, scores, post_process.score_thres, post_process.iou_thres)

    for i in selected:
        box = boxes[i, :].astype(np.int32).flatten()
        score = float(scores[i]) * 100
        label, color = labels(classes[i], use_bgr=True)

        draw_box(source, box, label, score, color)
    return source


def main(configs):
    net = load_net(configs.net.path, configs.net.gpu, configs.net.dnn)
    net.assert_input_shape(configs.processing.input_shape)
    net.warmup()

    _, _, input_height, input_width = net.input_shape  # [b, c, h, w]

    pre_process = Preprocessing(configs.processing.prep_steps, (input_height, input_width))
    post_process = Postprocessing(
        configs.processing.prep_steps,
        configs.processing.iou_thres,
        configs.processing.score_thres,
    )
    labels = Labels(configs.net.labels)

    if configs.source.type == "image":
        img = cv2.imread(configs.source.path)
        img = detect(net, img, pre_process, post_process, labels)

        name = Path(configs.source.path).stem
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)

        cv2.imshow(name, img)
        export_image(img, configs.export)
        cv2.waitKey(0)
    elif configs.source.type == "video":
        # Video processing
        vid_source = 0 if configs.source.path == "0" else configs.source.path
        cap = cv2.VideoCapture(vid_source)
        writer = VideoWriter(cap, configs.export)

        name = "Webcam" if configs.source.path == "0" else Path(configs.source.path).stem
        cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
        log_info("Processing video", "press 'q' to exit.")

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            frame = detect(net, frame, pre_process, post_process, labels)
            cv2.imshow(name, frame)
            writer.write(frame)

            if cv2.waitKey(1) == ord("q"):
                break

        cap.release()
        writer.close()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    configs = get_configs()
    main(configs)
