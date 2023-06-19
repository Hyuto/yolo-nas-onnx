from pathlib import Path

import numpy as np
import cv2

from yolo_nas.models import load_net
from yolo_nas.processing import Preprocessing, Postprocessing
from yolo_nas.draw import draw_box
from yolo_nas.cli import get_configs
from yolo_nas.utils import Labels, VideoWriter, export_image, log_info


def detect(net, source, pre_process, post_process, labels):
    """Detect Image/Frame array"""
    net_input = source.copy()  # copy source array
    input_, prep_meta = pre_process(net_input)  # run preprocess
    outputs = net.forward(input_)  # forward

    boxes, scores, classes = post_process(outputs, prep_meta)  # postprocess output
    selected = cv2.dnn.NMSBoxes(
        boxes, scores, post_process.score_thres, post_process.iou_thres
    )  # run nms to filter boxes

    for i in selected:  # loop through selected idx
        box = boxes[i, :].astype(np.int32).flatten()  # get box
        score = float(scores[i]) * 100  # percentage score
        label, color = labels(classes[i], use_bgr=True)  # get label and color class_id

        draw_box(source, box, label, score, color)  # draw boxes
    return source  # Image array after draw process


def main(configs):
    net = load_net(configs.net.path, configs.net.gpu, configs.net.dnn)  # load net
    net.assert_input_shape(configs.processing.input_shape)  # check input shape
    net.warmup()  # warmup net

    _, _, input_height, input_width = net.input_shape  # get input height and width [b, c, h, w]

    pre_process = Preprocessing(
        configs.processing.prep_steps, (input_height, input_width)
    )  # get preprocess
    post_process = Postprocessing(
        configs.processing.prep_steps,
        configs.processing.iou_thres,
        configs.processing.score_thres,
    )  # get postprocess

    labels = Labels(configs.net.labels)

    if configs.source.type == "image":  # image processing
        img = cv2.imread(configs.source.path)  # read image
        img = detect(net, img, pre_process, post_process, labels)  # detect image

        name = Path(configs.source.path).stem  # get filename
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)  # window config
        cv2.imshow(name, img)  # show image
        log_info("Image Processing", "press 'q' to exit.")

        export_image(img, configs.export)  # export image if configs.export isn't None
        cv2.waitKey(0)
    elif configs.source.type == "video":  # video processing
        # Video processing
        vid_source = (
            int(configs.source.path) if configs.source.path.isnumeric() else configs.source.path
        )  # get video source webcam or file
        cap = cv2.VideoCapture(vid_source)  # VideoCapture
        writer = VideoWriter(cap, configs.export)  # VideoWriter

        name = (
            f"Webcam: {configs.source.path}"  # if webcam
            if configs.source.path.isnumeric()
            else Path(configs.source.path).stem  # if file
        )  # get name for window
        cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)  # window config
        log_info("Video Processing", "press 'q' to exit.")

        while cap.isOpened():  # loop through every frame
            ret, frame = cap.read()  # get frame

            if not ret:
                break

            frame = detect(net, frame, pre_process, post_process, labels)  # detect frame
            cv2.imshow(name, frame)  # show frame
            writer.write(frame)  # write frame if export isn't None

            if cv2.waitKey(1) == ord("q"):
                break

        cap.release()  # release VideoCapture
        writer.close()  # close VideoWriter

    cv2.destroyAllWindows()  # close all OpenCV windows


if __name__ == "__main__":
    configs = get_configs()
    main(configs)
