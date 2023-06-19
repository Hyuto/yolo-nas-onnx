import numpy as np
import cv2


def draw_box(source, box, label, score, color, alpha=0.25):
    """Draw boxes on images"""
    # fill box
    crop_box = source[
        box[1] : (box[1] + box[3]), box[0] : (box[0] + box[2])
    ]  # crop box from source
    color_box = np.ones([*crop_box.shape[:2], 1], dtype=np.uint8) * np.asarray(
        color, dtype=np.uint8
    )  # color box (same size with crop). [h, w, 1] * [c] => [h, w, c]
    cv2.addWeighted(
        crop_box, 1 - alpha, color_box, alpha, 1.0, crop_box
    )  # weighted from color box to source

    cv2.rectangle(source, box, color, 2)  # draw box

    # measuring text
    size = min(source.shape[:2]) * 0.0007
    thickness = int(min(source.shape[:2]) * 0.001)
    (label_width, label_height), _ = cv2.getTextSize(
        f"{label} - {round(score, 2)}%",
        cv2.FONT_HERSHEY_SIMPLEX,
        size,
        thickness,
    )
    # draw labels (filled rect with text inside)
    cv2.rectangle(
        source,
        (box[0] - 1, box[1] - int(label_height * 2)),
        (box[0] + int(label_width * 1.1), box[1]),
        color,
        cv2.FILLED,
    )
    cv2.putText(
        source,
        f"{label} - {round(score, 2)}%",
        (box[0], box[1] - int(label_height * 0.7)),
        cv2.FONT_HERSHEY_SIMPLEX,
        size,
        [255, 255, 255],
        thickness,
        cv2.LINE_AA,
    )
