import os
from typing import Any


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
    # fmt: off
    # default coco classes
    labels = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
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
    colors = Colors()

    def __init__(self, from_file=None):
        if from_file:
            self.labels = self._get_labels_from_file(from_file)

    def __call__(self, i, use_bgr=False):
        return self.labels[i], self.colors(i, use_bgr)

    def _get_labels_from_file(self, path):
        with open(path) as f:
            self.labels = [x.strip() for x in f.readlines()]
