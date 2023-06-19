# Inspired from: https://github.com/Deci-AI/super-gradients/blob/3.1.1/src/super_gradients/training/processing/processing.py

import numpy as np
import cv2

YOLO_NAS_DEFAULT_PROCESSING_STEPS = [
    {"DetLongMaxRescale": None},
    {"CenterPad": {"pad_value": 114}},
    {"Standardize": {"max_value": 255.0}},
]


class Preprocessing:
    """Preprocessing Handler

    Args:
        steps (List[Dict]): Preprocessing steps, list of dictionary contains name and args.
        out_shape (Tuple[int]): image out shapes [h, w].

    Examples:
        Simple preprocessing image

        >>> prep = Preprocessing([{"DetLongMaxRescale": None}])
        >>> prep(img)
    """

    def __init__(self, steps, out_shape):
        self.steps = steps
        self.out_shape = out_shape

    @staticmethod
    def __rescale_img(img, out_shape):
        """default rescale func"""
        return cv2.resize(img, dsize=out_shape, interpolation=cv2.INTER_LINEAR).astype(np.uint8)

    def _standarize(self, img, max_value):
        """standarize img based on max value"""
        return (img / max_value).astype(np.float32), None

    def _det_rescale(self, img):
        """Rescale image to output based with scale factors"""
        scale_factor_h, scale_factor_w = (
            self.out_shape[0] / img.shape[0],
            self.out_shape[1] / img.shape[1],
        )  # calc scale factor
        return self.__rescale_img(img, self.out_shape), {
            "scale_factors": (scale_factor_w, scale_factor_h)
        }

    def _det_long_max_rescale(self, img):
        """Rescale image to output based on max size"""
        height, width = img.shape[:2]
        scale_factor = min(
            (self.out_shape[1] - 4) / height, (self.out_shape[0] - 4) / width
        )  # calc scale factor from max size

        if scale_factor != 1.0:  # resize if scale factor isn't 1
            new_height, new_width = round(height * scale_factor), round(width * scale_factor)
            img = self.__rescale_img(img, (new_width, new_height))

        return img, {"scale_factors": (scale_factor, scale_factor)}

    def _bot_right_pad(self, img, pad_value):
        """Pad bottom and right only (palce image on top left)"""
        pad_height, pad_width = self.out_shape[1] - img.shape[0], self.out_shape[0] - img.shape[1]
        return cv2.copyMakeBorder(
            img, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=[pad_value] * img.shape[-1]
        ), {"padding": (0, pad_height, 0, pad_width)}

    def _center_pad(self, img, pad_value):
        """Pad center (palce image on center)"""
        pad_height, pad_width = self.out_shape[1] - img.shape[0], self.out_shape[0] - img.shape[1]
        pad_top, pad_left = pad_height // 2, pad_width // 2
        return cv2.copyMakeBorder(
            img,
            pad_top,
            pad_height - pad_top,
            pad_left,
            pad_width - pad_left,
            cv2.BORDER_CONSTANT,
            value=[pad_value] * img.shape[-1],
        ), {"padding": (pad_top, pad_height - pad_top, pad_left, pad_width - pad_left)}

    def _normalize(self, img, mean, std):
        """Normalize image based on mean and stdev"""
        return (img - np.asarray(mean)) / np.asarray(std), None

    def _call_fn(self, name):
        """Call prep func from string name"""
        mapper = {
            "Standardize": self._standarize,
            "DetRescale": self._det_rescale,
            "DetLongMaxRescale": self._det_long_max_rescale,
            "BotRightPad": self._bot_right_pad,
            "CenterPad": self._center_pad,
            "Normalize": self._normalize,
        }
        return mapper[name]

    def __call__(self, img):
        """Do all preprocessing steps on single image"""
        img = img.copy()  # copy image
        metadata = []  # init metadata list

        for st in self.steps:  # loop processing steps
            if not st:  # if step isn't None
                continue
            name, kwargs = list(st.items())[0]  # name and kwargs from step
            img, meta = (
                self._call_fn(name)(img, **kwargs) if kwargs else self._call_fn(name)(img)
            )  # process image
            metadata.append(meta)  # append metadata

        img = cv2.dnn.blobFromImage(img, swapRB=True)  # image to blob [1, c, h, w] RGB
        return img, metadata


class Postprocessing:
    """Postprocessing Handler

    Args:
        steps (List[Dict]): Preprocessing steps, list of dictionary contains name and args.
        iou_thres (float): Float representing NMS/IOU threshold.
        score_thres (float): image out shapes [h, w].

    Examples:
        Postprocessing outputs (boxes, scores)

        >>> postp = Postprocessing([{"DetLongMaxRescale": None}], o.45, 0.25)
        >>> prep(output, prep_metadata)
    """

    def __init__(self, steps, iou_thres, score_thres):
        self.steps = steps
        self.iou_thres = iou_thres
        self.score_thres = score_thres

    def _rescale_boxes(self, boxes, metadata):
        """Rescale boxes to original image size"""
        scale_factors_w, scale_factors_h = metadata["scale_factors"]
        boxes[:, [0, 2]] /= scale_factors_w
        boxes[:, [1, 3]] /= scale_factors_h
        return boxes

    def _shift_bboxes(self, boxes, metadata):
        """Shift boxes because of padding"""
        pad_top, _, pad_left, _ = metadata["padding"]
        boxes[:, [0, 2]] -= pad_left
        boxes[:, [1, 3]] -= pad_top
        return boxes

    def _call_fn(self, name):
        """Call postp func from string name"""
        mapper = {
            "DetRescale": self._rescale_boxes,
            "DetLongMaxRescale": self._rescale_boxes,
            "BotRightPad": self._shift_bboxes,
            "CenterPad": self._shift_bboxes,
            "Standardize": None,
            "Normalize": None,
        }
        return mapper[name]

    def __call__(self, outputs, metadata):
        """Do all preprocessing steps on single output"""
        boxes, raw_scores = outputs  # get boxes and scores from outputs
        boxes = np.squeeze(boxes, 0)  # squeeze boxes [1, n, 4] => [n, 4]

        metadata = metadata.copy()  # copy preprocessing metadata
        for st in reversed(self.steps):  # reverse looping processing steps
            if not st:  # if step is None
                continue
            name, _ = list(st.items())[0]  # get step name
            meta = metadata.pop()  # get step metadata
            if not self._call_fn(name):  # if step is None
                continue
            boxes = self._call_fn(name)(boxes, meta)  # process boxes

        # change xyxy to xywh
        boxes[:, 2] -= boxes[:, 0]
        boxes[:, 3] -= boxes[:, 1]

        # find max from scores and flatten it [1, n, num_class] => [n]
        scores = raw_scores.max(axis=2).flatten()
        # find index from max scores (class_id) and flatten it [1, n, num_class] => [n]
        classes = np.argmax(raw_scores, axis=2).flatten()
        return boxes, scores, classes
