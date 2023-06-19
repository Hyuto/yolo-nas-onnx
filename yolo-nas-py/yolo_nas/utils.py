import json

import cv2

# fmt: off
# default coco classes
COCO_DEFAULT_LABELS = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                       "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                       "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
                       "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
                       "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
                       "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                       "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
                       "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", 
                       "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
                       "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
# fmt: on


class Colors:
    """Ultralytics color palette https://ultralytics.com/"""

    def __init__(self):
        # fmt: off
        hexs = ("FF3838", "FF9D97", "FF701F", "FFB21D", "CFD231", "48F90A", "92CC17", "3DDB86",
                "1A9334", "00D4BB", "2C99A8", "00C2FF", "344593", "6473FF", "0018EC", "8438FF",
                "520085", "CB38FF", "FF95C8", "FF37C7")
        # fmt: on
        self.palette = [self.hex2rgb(f"#{c}") for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i: int, bgr: bool = False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h: str):  # rgb order (PIL)
        return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))


class Labels:
    """Labels Handler

    Args:
        labels (List[str]): Models labels (class name)

    Examples:
        Get label (class name) with color from class_id

        >>> labels = Labels(COCO_DEFAULT_LABELS)
        >>> labels(10, use_bgr=True)
    """

    colors = Colors()

    def __init__(self, labels):
        self.labels = labels

    def __call__(self, i, use_bgr=False):
        return self.labels[i], self.colors(i, use_bgr)


class CustomMetadata:
    keys = ["type", "original_insz", "iou_thres", "score_thres", "prep_steps", "labels"]

    def __init__(self, path):
        with open(path) as f:
            data = json.load(f)

        assert (
            list(data.keys()) == self.keys
        ), "Different structure metadata file! Please generate file from https://gist.github.com/Hyuto/f3db1c0c2c36308284e101f441c2555f"

        self.type = data["type"]
        self.original_insz = data["original_insz"]
        self.iou_thres = data["iou_thres"]
        self.score_thres = data["score_thres"]
        self.prep_steps = data["prep_steps"]
        self.labels = data["labels"]


def log_info(header, body):
    print("\033[1m\033[94m" + header + ": \033[0m" + body)


def log_warning(header, body):
    print("⚠️ \033[1m\033[93m" + header + ": \033[0m" + body)


def log_error(header, body):
    print("❌ \033[1m\033[91m" + header + ": \033[0m" + body)


def export_image(img, path):
    """Export numpy.ndarray to image if path is specified (isn't None)

    Args:
        img (numpy.ndarray): Image array
        path (str): Path to export
    """
    if path:
        print("🚀", end=" ")
        log_info("Exporting Image", path)
        cv2.imwrite(path, img)


class VideoWriter:
    """Video writer. Export video frame by frame if path is specified.

    Args:
        cap (cv2.VideoCapture): OpenCV VideoCapture instance
        path (str): Path to export
    """

    def __init__(self, cap, path):
        self.export = path

        if self.export:
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
            else:
                raise NotImplementedError("Opencv Video Capture isn't opened yet!")

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.out = cv2.VideoWriter(path, fourcc, fps, (width, height))

    def write(self, frame):
        """Write frame"""
        if self.export:
            self.out.write(frame)

    def close(self):
        """Close exporter"""
        if self.export:
            print("🚀", end=" ")
            log_info("Exporting video", self.export)
            self.out.release()
