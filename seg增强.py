import warnings

warnings.filterwarnings('ignore')
import os
import shutil
import cv2
from tqdm import tqdm
import numpy as np
import albumentations as A
from PIL import Image

# 类别列表
CLASSES = ['algal leaf', 'Anthracnose', 'bird eye spot', 'brown blight', 'gray light', 'healthy', 'red leaf spot', 'white spot']

# 路径定义
ORIGINAL_DIR = r'C:\Users\zheng\Desktop\labels'  # 原始数据文件夹（图片 + 标签混放）
AUG_DIR = r'C:\Users\zheng\Desktop\labels_aug'    # 增强数据文件夹（图片 + 标签混放）
SHOW_SAVE_PATH = r'C:\Users\zheng\Desktop\aug_visual'  # 可视化保存路径

# 增强循环次数
ENHANCEMENT_LOOP = 3

# 增强策略
ENHANCEMENT_STRATEGY = A.Compose([
    A.Compose([
        A.Affine(scale=[0.5, 1.5], translate_percent=[0.0, 0.3], rotate=[-360, 360], shear=[-45, 45], keep_ratio=True, p=0.5),
        A.BBoxSafeRandomCrop(erosion_rate=0.2, p=0.1),
        A.D4(p=0.1),
        A.ElasticTransform(p=0.1),
        A.HorizontalFlip(p=0.1),
        A.VerticalFlip(p=0.1),
        A.GridDistortion(p=0.1),
        A.Perspective(p=0.1),
    ], p=1.0),

    A.Compose([
        A.GaussNoise(p=0.1),
        A.ISONoise(p=0.1),
        A.ImageCompression(quality_lower=50, quality_upper=100, p=0.1),
        A.RandomBrightnessContrast(p=0.1),
        A.RandomFog(p=0.1),
        A.RandomRain(p=0.1),
        A.RandomSnow(p=0.1),
        A.RandomShadow(p=0.1),
        A.RandomSunFlare(p=0.1),
        A.ToGray(p=0.1),
    ], p=1.0)
], bbox_params=A.BboxParams(format='yolo', min_visibility=0.1, label_fields=['class_labels']))

def draw_detections(box, name, img):
    height, width, _ = img.shape
    xmin, ymin, xmax, ymax = list(map(int, list(box)))

    line_thickness = max(1, int(min(height, width) / 200))
    font_scale = min(height, width) / 500
    font_thickness = max(1, int(min(height, width) / 200))
    text_offset_y = int(min(height, width) / 50)

    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), line_thickness)
    cv2.putText(img, str(name), (xmin, ymin - text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0),
                font_thickness, lineType=cv2.LINE_AA)
    return img

def show_labels(base_dir):
    if not os.path.exists(base_dir):
        print(f"错误：文件夹不存在 → {base_dir}")
        return

    if os.path.exists(SHOW_SAVE_PATH):
        shutil.rmtree(SHOW_SAVE_PATH)
    os.makedirs(SHOW_SAVE_PATH, exist_ok=True)

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG'}

    for images_name in tqdm(os.listdir(base_dir)):
        if not any(images_name.lower().endswith(ext) for ext in image_extensions):
            continue

        file_heads, _ = os.path.splitext(images_name)
        images_path = os.path.join(base_dir, images_name)
        labels_path = os.path.join(base_dir, f'{file_heads}.txt')

        if not os.path.exists(labels_path):
            print(f'{labels_path} label file not found...')
            continue

        with open(labels_path) as f:
            labels = np.array([
                np.array(x.strip().split(), dtype=np.float64)
                for x in f.readlines() if x.strip()
            ], dtype=np.float64)

        images = cv2.imread(images_path)
        if images is None:
            print(f"无法读取图片 {images_path}")
            continue

        height, width, _ = images.shape
        for cls, x_center, y_center, w, h in labels:
            x_center *= width
            y_center *= height
            w *= width
            h *= height
            draw_detections([x_center - w // 2, y_center - h // 2, x_center + w // 2, y_center + h // 2],
                            CLASSES[int(cls)], images)

        save_path = os.path.join(SHOW_SAVE_PATH, images_name)
        cv2.imwrite(save_path, images)
        print(f'{save_path} save success...')

def data_aug_single(images_name):
    file_heads, postfix = os.path.splitext(images_name)
    images_path = os.path.join(ORIGINAL_DIR, images_name)
    labels_path = os.path.join(ORIGINAL_DIR, f'{file_heads}.txt')

    if not os.path.exists(labels_path):
        print(f'{labels_path} label file not found...')
        return

    with open(labels_path) as f:
        labels = np.array([
            np.array(x.strip().split(), dtype=np.float64)
            for x in f.readlines() if x.strip()
        ], dtype=np.float64)

    try:
        images = Image.open(images_path)
    except Exception as e:
        print(f"无法打开图片 {images_name} → {e}")
        return

    for i in range(ENHANCEMENT_LOOP):
        new_file_heads = f'{file_heads}_{i:03d}'
        new_images_name = os.path.join(AUG_DIR, f'{new_file_heads}{postfix}')
        new_labels_name = os.path.join(AUG_DIR, f'{new_file_heads}.txt')

        try:
            transformed = ENHANCEMENT_STRATEGY(
                image=np.array(images),
                bboxes=np.clip(labels[:, 1:], 0.0, 1.0),
                class_labels=labels[:, 0]
            )
        except Exception as e:
            print(f"增强失败 {images_name} 第{i}次 → {e}")
            continue

        transformed_image = transformed['image']
        transformed_bboxes = transformed['bboxes']
        transformed_class_labels = transformed['class_labels']

        cv2.imwrite(new_images_name, cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR))

        with open(new_labels_name, 'w', encoding='utf-8') as f:
            for cls, bbox in zip(transformed_class_labels, transformed_bboxes):
                f.write(f"{int(cls)} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")

        print(f"生成：{new_images_name} 和 {new_labels_name}")

def data_aug():
    if os.path.exists(AUG_DIR):
        shutil.rmtree(AUG_DIR)
    os.makedirs(AUG_DIR, exist_ok=True)

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG'}

    files = [f for f in os.listdir(ORIGINAL_DIR) if any(f.lower().endswith(ext) for ext in image_extensions)]

    for images_name in tqdm(files):
        data_aug_single(images_name)

if __name__ == '__main__':
    data_aug()
    show_labels(ORIGINAL_DIR)
    show_labels(AUG_DIR)