from ultralytics import YOLO
import os
import shutil

ABLATION_DIR = "./ablation" #folder with ablation experiments models
OUTPUT_DIR = './onnx_models'



def export_to_onnx(
    path: str,
    project_name: str,
    output_folder: str = "",
    export_model_name: str = "",
):
    """
    projects_path - projects folder
    output_folder - output folder
    export_model_name - name of exported model
    """

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    model_path = os.path.join(path, project_name, "train", "weights", "best.pt")

    model = YOLO(model_path)  # загружаем самую лучшую модель

    model.fuse()  # слияние слоев для оптимизации производительности

    converted_path = model.export(
        format="onnx",
        simplify=True,  # упрощаем
        dynamic=True,  # возможность обрабатывать батчами
        device=0,  # gpu
    )

    # перемещаем в папку с моделями
    output_path = os.path.join(
        output_folder, f"{project_name}_{export_model_name}.onnx"
    )
    shutil.move(converted_path, output_path)


if __name__ == "__main__":
    for experiment in os.listdir(ABLATION_DIR):
        export_to_onnx(ABLATION_DIR, experiment, output_folder=OUTPUT_DIR)
        print(f'Converted {experiment} - OK')