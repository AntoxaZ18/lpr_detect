from onnxruntime.quantization import quantize_static, quant_pre_process, CalibrationDataReader, QuantType, QuantFormat, CalibrationMethod
import cv2
import os
import numpy as np
from typing import Tuple

class ImageFolderCalibrationDataReader(CalibrationDataReader):
    def __init__(self, folder_path):
        super().__init__()

        self.folder_path = folder_path
        self.calibration_images = os.listdir(folder_path)
        self.current_item = 0

    def letterbox(
        self, img: np.ndarray, new_shape: Tuple[int, int] = (640, 640)
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Resize and reshape images while maintaining aspect ratio by adding padding.

        Args:
            img (np.ndarray): Input image to be resized.
            new_shape (Tuple[int, int]): Target shape (height, width) for the image.

        Returns:
            (np.ndarray): Resized and padded image.
            (Tuple[int, int]): Padding values (top, left) applied to the image.
        """
        shape = img.shape[:2]  # current shape [height, width]

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = (
            (new_shape[1] - new_unpad[0]) / 2,
            (new_shape[0] - new_unpad[1]) / 2,
        )  # wh padding

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        return img, (top, left)

    

    def get_next(self) -> dict:
        """generate the input data dict in the input format to your ONNX model"""
        if self.current_item == len(self.calibration_images):
            return None  # None signals that the calibration is finished

        image = cv2.imread(os.path.join(self.folder_path, self.calibration_images[self.current_item]))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        image, pad = self.letterbox(image, (640, 640))
        # some image preprocessing to match model's input requirements
        image = image.astype('float32') / 255.

        image = np.transpose(image, (2, 0, 1))  # Channel first

        self.current_item += 1

        print(f"{self.current_item} of {self.__len__()}")

        return {"images": np.expand_dims(image, 0)}

    def __len__(self) -> int:
        """get length of the calibration dataset"""
        return len(self.calibration_images)


if __name__ == "__main__":

    model_initial_path = "./onnx_models/baseline_nano_.onnx"
    model_prep_path = "model.prep.onnx"
    model_quant_path = "model.quant.onnx"

    calibration_dataset = ImageFolderCalibrationDataReader("./val/images")

    # quant_pre_process(model_initial_path, model_prep_path)
    # quantize_static(model_prep_path, model_quant_path, calibration_dataset,  weight_type=QuantType.QInt8)

    # quantize_static(model_initial_path, model_quant_path, calibration_dataset, quant_format=QuantFormat.QDQ, calibrate_method=CalibrationMethod.MinMax, activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)