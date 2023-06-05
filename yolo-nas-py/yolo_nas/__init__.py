from collections import namedtuple

import numpy as np
import cv2

Padding = namedtuple("Padding", "top bottom left right")
PrepMetadata = namedtuple("PrepMetadata", "scale_factors padding")


def preprocess(img, input_size):
    # https://github.com/Deci-AI/super-gradients/blob/46120860cdc392d76382ea0cd7c5941d7ec1e360/src/super_gradients/training/processing/processing.py#L308

    ## DetectionLongestMaxSizeRescale
    height, width = img.shape[:2]
    scale_factor = min((input_size[1] - 4) / height, (input_size[0] - 4) / width)

    if scale_factor != 1.0:
        new_height, new_width = round(height * scale_factor), round(width * scale_factor)
        img = cv2.resize(img, dsize=(new_width, new_height), interpolation=cv2.INTER_LINEAR).astype(
            np.uint8
        )

    ## DetectionCenterPadding
    pad_height, pad_width = input_size[1] - img.shape[0], input_size[0] - img.shape[1]
    pad_top, pad_left = pad_height // 2, pad_width // 2
    img = cv2.copyMakeBorder(
        img,
        pad_top,
        pad_height - pad_top,
        pad_left,
        pad_width - pad_left,
        cv2.BORDER_CONSTANT,
        value=[114, 114, 114],
    )

    ## StandardizeImage
    img = (img / 255.0).astype(np.float32)

    ## ImagePermute
    ### BGR (OpenCV default) -> RGB (YOLO-NAS default)
    img = cv2.dnn.blobFromImage(img, swapRB=True)
    return img, PrepMetadata(
        scale_factors=scale_factor,
        padding=Padding(
            top=pad_top,
            bottom=pad_height - pad_top,
            left=pad_left,
            right=pad_width - pad_left,
        ),
    )


def postprocess(outputs, metadata):
    boxes, raw_scores = outputs
    boxes = np.squeeze(boxes, 0)

    ## DetectionCenterPadding
    boxes[:, [0, 2]] += -metadata.padding.left
    boxes[:, [1, 3]] += -metadata.padding.top

    ## DetectionLongestMaxSizeRescale
    boxes /= metadata.scale_factors

    # change xyxy to xywh
    boxes[:, 2] -= boxes[:, 0]
    boxes[:, 3] -= boxes[:, 1]

    scores = raw_scores.max(axis=2).flatten()
    classes = np.argmax(raw_scores, axis=2).flatten()
    return boxes, scores, classes
