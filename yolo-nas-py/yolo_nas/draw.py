import cv2


def draw_box(source, box, label, score, color) -> None:
    """Draw boxes on images"""
    cv2.rectangle(source, box, color, 2)  # draw box
    (label_width, label_height), _ = cv2.getTextSize(
        f"{label} - {round(score, 2)}",
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        1,
    )
    cv2.rectangle(
        source,
        (box[0] - 1, box[1] - label_height - 6),
        (box[0] + label_width + 1, box[1]),
        color,
        -1,
    )
    cv2.putText(
        source,
        f"{label} - {round(score, 2)}",
        (box[0], box[1] - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        [255, 255, 255],
        1,
    )
