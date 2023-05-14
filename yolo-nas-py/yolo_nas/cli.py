import argparse
import os


def parse_opt():
    parser = argparse.ArgumentParser(description="Detect using YOLO-NAS model")
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "-m", "--model", type=str, required=True, help="YOLO-NAS Segmentation onnx model path"
    )
    source = parser.add_argument_group("source arguments")
    source.add_argument("-i", "--image", type=str, help="Image source")
    source.add_argument("-v", "--video", type=str, help="Video source")

    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU if available",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=100,
        help="Integer representing the maximum number of boxes to be selected per class",
    )
    parser.add_argument(
        "--score-tresh",
        type=float,
        default=0.25,
        help="Float representing the threshold for deciding when to remove boxes",
    )
    parser.add_argument(
        "--iou-tresh",
        type=float,
        default=0.4,
        help="Float representing the threshold for deciding whether boxes overlap too much with respect to IOU",
    )
    parser.add_argument("--labels", type=str, help="Using custom labels via external txt file")
    parser.add_argument(
        "--dnn",
        action="store_true",
        help="Use OpenCV DNN module [if false using onnxruntime] for backend",
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
    if opt.labels:
        if not os.path.exists(opt.model):
            raise FileNotFoundError("Wrong path! Not found labels file.")
    if opt.image:
        if not os.path.exists(opt.image):
            raise FileNotFoundError("Wrong path! Not found image source.")
    if opt.video:
        if not os.path.exists(opt.video) and opt.video != "0":
            raise FileNotFoundError("Wrong path! Not found video source.")

    # logging
    args = vars(opt).items()
    emoji = "🖼️" if opt.image else "📷"
    print(emoji + " \033[1m\033[94m" + "Detect: " + "\033[0m", end="")
    print(", ".join([f"{x}={y}" for x, y in args if y is not None]))

    return opt
