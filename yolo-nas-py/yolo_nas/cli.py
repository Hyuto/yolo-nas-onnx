import argparse
import os

from collections import namedtuple

from .processing import YOLO_NAS_DEFAULT_PROCESSING_STEPS
from .utils import log_info, CustomMetadata, COCO_DEFAULT_LABELS

Source = namedtuple("Source", "type path")
Net = namedtuple("Net", "path gpu dnn labels")
Processing = namedtuple("Processing", "input_shape prep_steps score_thres iou_thres")
Configs = namedtuple("Configs", "net source processing export")


def get_configs():
    parser = argparse.ArgumentParser(description="Detect using YOLO-NAS model")
    required = parser.add_argument_group("required arguments")
    required.add_argument("-m", "--model", type=str, required=True, help="YOLO-NAS ONNX model path")
    source = parser.add_argument_group("source arguments")
    source.add_argument("-i", "--image", type=str, help="Image source")
    source.add_argument("-v", "--video", type=str, help="Video source")

    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU if available",
    )
    parser.add_argument(
        "--score-thres",
        type=float,
        help="Float representing the threshold for deciding when to remove boxes",
    )
    parser.add_argument(
        "--iou-thres",
        type=float,
        help="Float representing the threshold for deciding whether boxes overlap too much with respect to IOU",
    )
    parser.add_argument(
        "--dnn",
        action="store_true",
        help="Use OpenCV DNN module [if false using onnxruntime] for backend",
    )
    parser.add_argument(
        "--custom-metadata",
        type=str,
        help="Path to metadata file (Generated from https://gist.github.com/Hyuto/f3db1c0c2c36308284e101f441c2555f)",
    )
    parser.add_argument(
        "--export",
        type=str,
        help="Export to a file (path with extension | mp4 is a must for video)",
    )

    opt = parser.parse_args()  # parsing args

    # source checking
    if opt.image is None and opt.video is None:
        raise argparse.ArgumentError("Please specify image or video source!")
    elif opt.image and opt.video:
        raise argparse.ArgumentError("Please specify either image or video source!")

    # path checking
    if not os.path.exists(opt.model):
        raise FileNotFoundError("Wrong path! Not found ONNX model.")
    if opt.image:
        if not os.path.exists(opt.image):
            raise FileNotFoundError("Wrong path! Not found image source.")
    if opt.video:
        if not os.path.exists(opt.video) and opt.video != "0":
            raise FileNotFoundError("Wrong path! Not found video source.")
    if opt.custom_metadata:
        if not os.path.exists(opt.custom_metadata):
            raise FileNotFoundError("Wrong path! Metadata file not found.")
    if opt.export:
        if not os.path.exists(os.path.dirname(opt.export)):
            raise FileNotFoundError("Wrong path! Export directory not found.")

    # Load metadata file if present
    metadata = CustomMetadata(opt.custom_metadata) if opt.custom_metadata else None
    if metadata:
        if not opt.score_thres:
            opt.score_thres = metadata.score_thres
        if not opt.iou_thres:
            opt.iou_thres = metadata.iou_thres

    # default args
    if not opt.score_thres:
        opt.score_thres = 0.25
    if not opt.iou_thres:
        opt.iou_thres = 0.45

    # configs
    source = Source("image" if opt.image else "video", opt.image if opt.image else opt.video)
    net = Net(opt.model, opt.gpu, opt.dnn, metadata.labels if metadata else COCO_DEFAULT_LABELS)
    processing = Processing(
        metadata.original_insz if metadata else None,
        metadata.prep_steps if metadata else YOLO_NAS_DEFAULT_PROCESSING_STEPS,
        opt.score_thres,
        opt.iou_thres,
    )

    # logging
    args = vars(opt).items()
    print("🖼️" if opt.image else "📷", end=" ")
    log_info("Detect", ", ".join([f"{x}={y}" for x, y in args if y is not None]))

    return Configs(net, source, processing, opt.export)
