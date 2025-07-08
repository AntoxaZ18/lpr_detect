import argparse
from time import time

import cv2
from matplotlib import pyplot as plt

from yolo_onnnx import YoloONNX


def bench_ort_onnx(
    model_path: str,
    image_path: str,
    device="cpu",
    batch_size: int = 8,
    imsize: int = 640,
) -> float:
    frame = cv2.imread(image_path)
    batch_images = [frame] * batch_size

    model = YoloONNX(
        model_path, device=device, threads=batch_size, classes=["LPR"], imsize=imsize
    )

    def warmup(onnx_model, images, iterations=20):
        for i in range(iterations):
            _ = onnx_model(images)

    # прогреваем gpu
    if device != "cpu":
        warmup(model, batch_images)

    start = time()
    range_iter = 10

    for _ in range(range_iter):
        frame_boxes = model(batch_images)

    fps = 1 / ((time() - start) / (range_iter * batch_size))

    return fps


if __name__ == "__main__":
    # python onnx_benchmark.py --model model.onnx --image test_image.jpg --imsize 640
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="path to model for benchmarking")
    parser.add_argument("--image", type=str, help="path to image")
    parser.add_argument("--imsize", type=int, help="model size", default=640)
    parser.add_argument("--batch", type=int, help="batch size", default=8)

    args = parser.parse_args()

    model = args.model
    image = args.image
    imsize = args.imsize
    batch_size = args.batch

    output_dir = args.onnx

    fps = bench_ort_onnx(model, image, batch_size=batch_size, imsize=imsize)

    print(f"bencmark {model} with {imsize} image size: {fps} fps")


# model = YoloONNX(model, device='cpu')
# frame = cv2.imread('test.jpg')

# results = model([frame])

# plt.interactive(True)
# plt.draw()
# plt.imshow(results[0], interpolation='nearest')
# plt.show()
