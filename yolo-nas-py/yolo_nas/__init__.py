import numpy as np
import cv2


def preprocess(img, input_size):
    # https://github.com/Deci-AI/super-gradients/blob/46120860cdc392d76382ea0cd7c5941d7ec1e360/src/super_gradients/training/processing/processing.py#L308
    source_height, source_width, _ = img.shape
    max_size = max(source_width, source_height)  # get max size
    x_pad, y_pad = max_size - source_width, max_size - source_height
    source_padded = cv2.copyMakeBorder(img, 0, y_pad, 0, x_pad, cv2.BORDER_CONSTANT)

    ## ratios
    x_ratio = max_size / input_size[0]
    y_ratio = max_size / input_size[1]

    # run model
    input_img = cv2.dnn.blobFromImage(
        source_padded,
        1 / 255.0,
        input_size,
        swapRB=True,
    )  # normalize and resize: [h, w, 3] => [1, 3, h, w]

    return input_img, (x_ratio, y_ratio)


def postprocess(outputs, ratios):
    boxes, raw_scores = outputs
    x_ratio, y_ratio = ratios

    boxes[0, :, 2] = (boxes[0, :, 2] - boxes[0, :, 0]) * x_ratio
    boxes[0, :, 3] = (boxes[0, :, 3] - boxes[0, :, 1]) * y_ratio
    boxes[0, :, 0] *= x_ratio
    boxes[0, :, 1] *= y_ratio

    boxes = np.squeeze(boxes, 0)
    scores = raw_scores.max(axis=2).flatten()
    classes = np.argmax(raw_scores, axis=2).flatten()
    return boxes, scores, classes
