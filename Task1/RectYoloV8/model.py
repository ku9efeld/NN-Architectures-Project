"""
Модель YOLOv8-OBB для детекции слов на чеках (ICDAR-2019 SROIE).
"""

from ultralytics import YOLO

def get_yolov8_obb_model(model_size: str = 'n', weights_path: str = None):
    """
    Загружает YOLOv8-OBB модель.
    
    Args:
        model_size (str): 'n', 's', 'm', 'l', 'x'
        weights_path (str, optional): путь к сохранённым весам (.pt)
    
    Returns:
        YOLO: модель Ultralytics YOLOv8-OBB
    """
    if weights_path:
        model = YOLO(weights_path)
    else:
        model = YOLO(f'yolov8{model_size}-obb.pt')  # pretrained OBB модель
    return model