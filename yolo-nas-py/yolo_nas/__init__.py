import numpy as np
import cv2


def preprocess(img, input_size):
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
        input_size,
        swapRB=False,
        crop=False,
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
