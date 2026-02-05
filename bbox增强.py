import os
import cv2
import numpy as np
import shutil
from tqdm import tqdm
import albumentations as A

# ────────────────────────────────────────────────
# 配置
# ────────────────────────────────────────────────
CLASSES = [
    'algal leaf', 'Anthracnose', 'bird eye spot', 'brown blight',
    'gray light', 'healthy', 'red leaf spot', 'white spot'
]

ORIGINAL_DIR    = r'C:\Users\zheng\Desktop\labels'
AUG_DIR         = r'C:\Users\zheng\Desktop\labels_aug'
SHOW_SAVE_PATH  = r'C:\Users\zheng\Desktop\aug_visual'

ENHANCEMENT_LOOP = 3
SAVE_VISUALIZATION = True
# ────────────────────────────────────────────────

os.makedirs(AUG_DIR, exist_ok=True)
if SAVE_VISUALIZATION:
    os.makedirs(SHOW_SAVE_PATH, exist_ok=True)

def read_yolo_label(label_path):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cls = int(parts[0])
                x, y, w, h = map(float, parts[1:5])
                boxes.append([cls, x, y, w, h])
            except:
                continue
    return boxes

def write_yolo_label(label_path, boxes):
    os.makedirs(os.path.dirname(label_path), exist_ok=True)
    with open(label_path, 'w', encoding='utf-8') as f:
        for box in boxes:
            cls, x, y, w, h = box
            f.write(f"{int(cls)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

def xywh2xyxy(boxes, img_w, img_h):
    new_boxes = []
    for cls, x, y, w, h in boxes:
        x1 = (x - w/2) * img_w
        y1 = (y - h/2) * img_h
        x2 = (x + w/2) * img_w
        y2 = (y + h/2) * img_h
        new_boxes.append([x1, y1, x2, y2, cls])
    return new_boxes

def xyxy2xywh(boxes, img_w, img_h):
    new_boxes = []
    for x1, y1, x2, y2 in boxes:
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        cx = (x1 + x2) / 2 / img_w
        cy = (y1 + y2) / 2 / img_h
        new_boxes.append([cx, cy, w, h])
    return new_boxes

# ─── 增強 pipeline（適配 albumentations 2.0.x） ────────────────────────────────
aug_pipeline = A.Compose([
    # 幾何變換（只允許 90° 倍數旋轉 + 水平/垂直翻轉）
    A.OneOf([
        A.Rotate(limit=0, p=0.4),                   # 占位（不轉）
        A.RandomRotate90(p=0.55),                   # 90/180/270°
        A.HorizontalFlip(p=0.4),
        A.VerticalFlip(p=0.3),
    ], p=0.85),

    # 非等比縮放（只水平或只垂直）
    A.OneOf([
        A.Affine(scale={'x': (0.75, 1.35), 'y': 1.0}, p=0.4),   # 水平拉伸/壓縮
        A.Affine(scale={'x': 1.0, 'y': (0.75, 1.35)}, p=0.4),   # 垂直拉伸/壓縮
    ], p=0.55),

    # 顏色/亮度調整
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.20, contrast_limit=0.20, p=0.5),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=20, p=0.4),
    ], p=0.7),

    # 噪聲 / 模糊
    A.OneOf([
        A.GaussNoise(std_range=(0.10, 0.30), p=0.15),   # 正確寫法，適中噪聲
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3), p=0.15),
        A.Blur(blur_limit=3, p=0.15),
        A.MedianBlur(blur_limit=3, p=0.1),
    ], p=0.35),

    # 隨機遮擋（模擬葉片部分遮擋）
    A.CoarseDropout(
        num_holes_range=(1, 6),
        hole_height_range=(0.03, 0.08),
        hole_width_range=(0.03, 0.08),
        p=0.25
    ),

], bbox_params=A.BboxParams(
    format='pascal_voc',
    label_fields=['class_labels'],
    min_visibility=0.25
))


def augment_one_image(img_path, label_path, output_img_path, output_label_path, vis_path=None):
    img = cv2.imread(img_path)
    if img is None:
        print(f"無法讀取圖片：{img_path}")
        return False

    h, w = img.shape[:2]
    boxes = read_yolo_label(label_path)

    if not boxes:
        boxes_xyxy = []
        class_labels = []
    else:
        boxes_xyxy_with_cls = xywh2xyxy(boxes, w, h)
        class_labels = [int(b[4]) for b in boxes_xyxy_with_cls]
        boxes_xyxy = [b[:4] for b in boxes_xyxy_with_cls]

    # 應用增強
    transformed = aug_pipeline(image=img, bboxes=boxes_xyxy, class_labels=class_labels)

    aug_img    = transformed['image']
    aug_bboxes = transformed['bboxes']
    aug_labels = transformed['class_labels']

    # 轉回 YOLO 格式
    aug_xywh = xyxy2xywh(aug_bboxes, aug_img.shape[1], aug_img.shape[0])
    aug_boxes_yolo = [[int(cls)] + xywh for cls, xywh in zip(aug_labels, aug_xywh)]

    # 保存
    cv2.imwrite(output_img_path, aug_img)
    write_yolo_label(output_label_path, aug_boxes_yolo)

    # 可視化（可選）
    if vis_path and SAVE_VISUALIZATION:
        vis_img = aug_img.copy()
        img_h_new, img_w_new = aug_img.shape[:2]
        for box in aug_boxes_yolo:
            cls_id, x, y, bw, bh = box
            cls_id = int(cls_id)
            x1 = max(0, int((x - bw/2) * img_w_new))
            y1 = max(0, int((y - bh/2) * img_h_new))
            x2 = min(img_w_new, int((x + bw/2) * img_w_new))
            y2 = min(img_h_new, int((y + bh/2) * img_h_new))

            color = (0, 255, 0) if cls_id == 5 else (0, 0, 255)  # healthy 綠色
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)

            label_text = CLASSES[cls_id][:3] if 0 <= cls_id < len(CLASSES) else "???"
            cv2.putText(vis_img, label_text, (x1, max(y1 - 8, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imwrite(vis_path, vis_img)

    return True


def main():
    image_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    image_files = [f for f in os.listdir(ORIGINAL_DIR)
                   if f.lower().endswith(image_extensions)]

    print(f"找到 {len(image_files)} 張圖片，開始增強...")

    for img_name in tqdm(image_files):
        base_name = os.path.splitext(img_name)[0]
        img_path = os.path.join(ORIGINAL_DIR, img_name)
        label_path = os.path.join(ORIGINAL_DIR, base_name + '.txt')

        # 複製原始檔（如果還沒複製）
        dst_img = os.path.join(AUG_DIR, img_name)
        dst_label = os.path.join(AUG_DIR, base_name + '.txt')
        if not os.path.exists(dst_img):
            shutil.copy(img_path, dst_img)
        if os.path.exists(label_path) and not os.path.exists(dst_label):
            shutil.copy(label_path, dst_label)

        # 生成增強版本
        for i in range(ENHANCEMENT_LOOP):
            aug_img_name = f"{base_name}_aug{i+1}{os.path.splitext(img_name)[1]}"
            aug_label_name = f"{base_name}_aug{i+1}.txt"

            output_img = os.path.join(AUG_DIR, aug_img_name)
            output_label = os.path.join(AUG_DIR, aug_label_name)

            vis_name = f"{base_name}_aug{i+1}_vis.jpg" if SAVE_VISUALIZATION else None
            vis_path = os.path.join(SHOW_SAVE_PATH, vis_name) if vis_name else None

            augment_one_image(img_path, label_path, output_img, output_label, vis_path)

    print("增強完成！")
    print(f"增強圖片與標籤保存位置：{AUG_DIR}")
    if SAVE_VISUALIZATION:
        print(f"可視化結果保存位置：{SHOW_SAVE_PATH}")


if __name__ == '__main__':
    main()