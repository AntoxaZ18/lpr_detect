from yolo_onnnx import YoloONNX
import cv2
from time import time
import os
from matplotlib import pyplot as plt
from tqdm import tqdm

def bench_ort_onnx(model_path: str, image_path: str, device="cpu"):
    batch_size = os.cpu_count() if device == "cpu" else 8
    batch_size = os.cpu_count()

    batch_size = 1

    frame = cv2.imread(image_path)
    batch_images = [frame] * batch_size

    model = YoloONNX(model_path, device=device, threads=batch_size, classes=['LPR'])

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


models_dir = "./onnx_models"

results = {}

for model in tqdm(os.listdir(models_dir)):
    fps = bench_ort_onnx(f"{models_dir}/{model}", "39.bmp")
    results[model] = fps

for model, fps in results.items():
    print(f"{model} mean fps is {fps:.3f} fps")






# model = YoloONNX(model, device='cpu')
# frame = cv2.imread('test.jpg')

# results = model([frame])

# plt.interactive(True)
# plt.draw()
# plt.imshow(results[0], interpolation='nearest')
# plt.show()