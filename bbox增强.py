import os
import cv2
import numpy as np
import shutil
from tqdm import tqdm
import albumentations as A
import random

# ────────────────────────────────────────────────
# 配置（优化后）
# ────────────────────────────────────────────────
CLASSES = ['brown blight', 'disease', 'healthy', 'White spot']
ORIGINAL_DIR = r'C:\Users\zheng\Desktop\labels'
AUG_DIR = r'C:\Users\zheng\Desktop\labels_aug'
SHOW_SAVE_PATH = r'C:\Users\zheng\Desktop\aug_visual'

# 目标图片尺寸
TARGET_SIZE = (960, 960)  # (width, height)
ENHANCEMENT_LOOP = 10
SAVE_VISUALIZATION = True
BALANCE_SPEED = False  # 关闭则最大化增强强度
# 核心配置：强制要求所有输出文件必须包含Bbox
REQUIRE_BBOX = True  # 设置为True时，只保留有Bbox的文件
# 日志文件路径
LOG_PATH = r'C:\Users\zheng\Desktop\aug_log.txt'


# ────────────────────────────────────────────────

# 初始化日志文件
def init_log():
    with open(LOG_PATH, 'w', encoding='utf-8') as f:
        f.write("图片增强日志\n")
        f.write("=" * 50 + "\n")
        f.write(f"目标图片尺寸：{TARGET_SIZE[0]}×{TARGET_SIZE[1]}\n")
        f.write(f"强制要求Bbox：{REQUIRE_BBOX}\n")


# 写入日志
def write_log(content):
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(content + "\n")


os.makedirs(AUG_DIR, exist_ok=True)
if SAVE_VISUALIZATION:
    os.makedirs(SHOW_SAVE_PATH, exist_ok=True)
init_log()


def read_yolo_label(label_path):
    boxes = []
    if not os.path.exists(label_path):
        write_log(f"标签文件不存在：{label_path}")
        return boxes
    with open(label_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        if len(lines) == 0:
            write_log(f"原始标签为空：{label_path}")
            return boxes
        for idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                write_log(f"标签行格式错误（字段不足）：{label_path} 第{idx + 1}行：{line}")
                continue
            try:
                cls = int(parts[0])
                x, y, w, h = map(float, parts[1:5])
                # 增加边界检查（防止无效的bbox值）
                if 0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1:
                    boxes.append([cls, x, y, w, h])
                else:
                    write_log(f"标签值超出范围：{label_path} 第{idx + 1}行：{x, y, w, h}")
            except Exception as e:
                write_log(f"标签解析失败：{label_path} 第{idx + 1}行：{line} 错误：{str(e)}")
                continue
    return boxes


def write_yolo_label(label_path, boxes):
    # 如果强制要求Bbox且无有效boxes，不生成标签文件
    if REQUIRE_BBOX and len(boxes) == 0:
        write_log(f"跳过无Bbox标签文件生成：{label_path}")
        # 如果文件已存在则删除
        if os.path.exists(label_path):
            os.remove(label_path)
        return False

    os.makedirs(os.path.dirname(label_path), exist_ok=True)
    with open(label_path, 'w', encoding='utf-8') as f:
        if boxes:
            for box in boxes:
                cls, x, y, w, h = box
                f.write(f"{int(cls)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    # 校验：确保标签有对应的图片
    img_path = None
    img_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    for ext in img_extensions:
        temp_path = label_path.replace('.txt', ext)
        if os.path.exists(temp_path):
            img_path = temp_path
            break

    if img_path and not os.path.exists(img_path):
        write_log(f"警告：标签文件{label_path}无对应图片文件，已删除标签")
        os.remove(label_path)
        return False

    if not img_path and REQUIRE_BBOX:
        write_log(f"删除无对应图片的标签文件：{label_path}")
        os.remove(label_path)
        return False

    return True


def xywh2xyxy(boxes, img_w, img_h):
    new_boxes = []
    for cls, x, y, w, h in boxes:
        x1 = (x - w / 2) * img_w
        y1 = (y - h / 2) * img_h
        x2 = (x + w / 2) * img_w
        y2 = (y + h / 2) * img_h
        new_boxes.append([x1, y1, x2, y2, cls])
    return new_boxes


def xyxy2xywh(boxes, img_w, img_h):
    new_boxes = []
    for x1, y1, x2, y2, cls in boxes:
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        cx = (x1 + x2) / 2 / img_w
        cy = (y1 + y2) / 2 / img_h
        # 边界检查（防止负数或超过1的值）
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        w = max(0.0, min(1.0, w))
        h = max(0.0, min(1.0, h))
        new_boxes.append([cls, cx, cy, w, h])
    return new_boxes


# 调整图片尺寸并同步更新bbox
def resize_image_and_bboxes(img, boxes, target_size=TARGET_SIZE):
    """
    将图片缩放到目标尺寸，并同步调整bbox坐标
    :param img: 原始图片
    :param boxes: YOLO格式的bbox列表 [cls, x, y, w, h]
    :param target_size: 目标尺寸 (width, height)
    :return: 缩放后的图片，更新后的bbox列表
    """
    orig_h, orig_w = img.shape[:2]
    target_w, target_h = target_size

    # 缩放图片
    resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)

    if not boxes:  # 无bbox时直接返回
        return resized_img, boxes

    # 将YOLO坐标转换为像素坐标
    boxes_xyxy = xywh2xyxy(boxes, orig_w, orig_h)

    # 计算缩放比例
    scale_x = target_w / orig_w
    scale_y = target_h / orig_h

    # 缩放bbox像素坐标
    resized_boxes_xyxy = []
    for x1, y1, x2, y2, cls in boxes_xyxy:
        new_x1 = x1 * scale_x
        new_y1 = y1 * scale_y
        new_x2 = x2 * scale_x
        new_y2 = y2 * scale_y
        resized_boxes_xyxy.append([new_x1, new_y1, new_x2, new_y2, cls])

    # 转换回YOLO格式
    resized_boxes = xyxy2xywh(resized_boxes_xyxy, target_w, target_h)

    return resized_img, resized_boxes


# 自定义非黑色遮挡：在CoarseDropout后手动替换遮挡区域颜色
def add_colorful_dropout(img):
    h, w = img.shape[:2]
    # 生成随机遮挡掩码
    num_holes = random.randint(2, 10)
    hole_height = random.uniform(0.05, 0.12)
    hole_width = random.uniform(0.05, 0.12)

    for _ in range(num_holes):
        # 随机生成孔洞位置
        y = random.randint(0, h - 1)
        x = random.randint(0, w - 1)
        h_hole = int(hole_height * h)
        w_hole = int(hole_width * w)

        # 计算孔洞边界（防止越界）
        y1 = max(0, y - h_hole // 2)
        y2 = min(h, y + h_hole // 2)
        x1 = max(0, x - w_hole // 2)
        x2 = min(w, x + w_hole // 2)

        # 生成随机颜色（非纯黑）
        fill_color = (random.randint(10, 245), random.randint(10, 245), random.randint(10, 245))
        img[y1:y2, x1:x2] = fill_color
    return img


# ─── 优化版增强 pipeline ────────────────────
# 降低变换激进程度，减少bbox过滤
p_geo = 0.7 if BALANCE_SPEED else 0.8  # 降低几何变换概率
p_scale = 0.4 if BALANCE_SPEED else 0.5  # 降低缩放概率
p_color = 0.7 if BALANCE_SPEED else 0.8  # 颜色调整概率
p_noise = 0.4 if BALANCE_SPEED else 0.5  # 噪声概率
p_blur = 0.4 if BALANCE_SPEED else 0.45  # 模糊概率

aug_pipeline = A.Compose([
    # 几何变换（降低旋转/翻转强度）
    A.OneOf([
        A.Rotate(limit=15, p=0.3),
        A.RandomRotate90(p=0.4),
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.2),
    ], p=p_geo),

    # 非等比缩放（降低缩放幅度，减少bbox越界）
    A.OneOf([
        A.Affine(scale={'x': (0.8, 1.2), 'y': 1.0}, p=0.45),
        A.Affine(scale={'x': 1.0, 'y': (0.8, 1.2)}, p=0.45),
    ], p=p_scale),

    # 颜色/亮度调整
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.4),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=25, p=0.35),
        A.RandomBrightnessContrast(brightness_limit=(-0.3, -0.1), contrast_limit=0.1, p=0.25),
    ], p=p_color),

    # 噪声
    A.OneOf([
        A.GaussNoise(std_range=(0.10, 0.30), p=0.4),
        A.ISONoise(color_shift=(0.02, 0.06), intensity=(0.1, 0.4), p=0.4),
    ], p=p_noise),

    # 模糊
    A.OneOf([
        A.Blur(blur_limit=5, p=0.35),
        A.MedianBlur(blur_limit=5, p=0.35),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    ], p=p_blur),

], bbox_params=A.BboxParams(
    format='pascal_voc',
    label_fields=['class_labels'],
    min_visibility=0.1  # 从0.25降低到0.1，减少过滤
))


def augment_one_image(img_path, label_path, output_img_path, output_label_path, vis_path=None):
    img = cv2.imread(img_path)
    if img is None:
        write_log(f"無法讀取圖片：{img_path}")
        return False

    h, w = img.shape[:2]
    boxes = read_yolo_label(label_path)

    # 如果强制要求Bbox且原始无Bbox，直接返回失败
    if REQUIRE_BBOX and len(boxes) == 0:
        write_log(f"跳过无Bbox图片增强：{img_path}")
        # 删除已生成的空文件
        if os.path.exists(output_img_path):
            os.remove(output_img_path)
        if os.path.exists(output_label_path):
            os.remove(output_label_path)
        return False

    original_box_count = len(boxes)

    if not boxes:
        # 原始标签为空，且不强制要求Bbox时才保存
        if not REQUIRE_BBOX:
            resized_img, _ = resize_image_and_bboxes(img, boxes)
            cv2.imwrite(output_img_path, resized_img)
            write_yolo_label(output_label_path, [])
        return True

    # 转换bbox格式
    boxes_xyxy_with_cls = xywh2xyxy(boxes, w, h)
    class_labels = [int(b[4]) for b in boxes_xyxy_with_cls]
    boxes_xyxy = [b[:4] for b in boxes_xyxy_with_cls]

    # 应用增强
    try:
        transformed = aug_pipeline(image=img, bboxes=boxes_xyxy, class_labels=class_labels)
        aug_img = transformed['image']
        aug_bboxes_xyxy = transformed['bboxes']
        aug_labels = transformed['class_labels']

        # 将增强后的bbox转换回带cls的格式
        aug_bboxes_xyxy_with_cls = []
        for i, (x1, y1, x2, y2) in enumerate(aug_bboxes_xyxy):
            aug_bboxes_xyxy_with_cls.append([x1, y1, x2, y2, aug_labels[i]])

    except Exception as e:
        write_log(f"增强失败：{img_path} 错误：{str(e)}")
        # 增强失败时调整原图尺寸后保存（仅当有Bbox时）
        if len(boxes) > 0:
            resized_img, resized_boxes = resize_image_and_bboxes(img, boxes)
            cv2.imwrite(output_img_path, resized_img)
            write_yolo_label(output_label_path, resized_boxes)
        else:
            # 无Bbox且强制要求时删除文件
            if os.path.exists(output_img_path):
                os.remove(output_img_path)
            if os.path.exists(output_label_path):
                os.remove(output_label_path)
        return False

    # 如果增强后无Bbox且强制要求，放弃保存
    if REQUIRE_BBOX and len(aug_bboxes_xyxy) == 0:
        write_log(f"增强后无Bbox，跳过保存：{output_img_path} (原始{original_box_count}个Bbox)")
        return False

    # 记录过滤后的bbox数量
    if len(aug_bboxes_xyxy) < original_box_count:
        write_log(f"bbox被过滤：{img_path} 原始{original_box_count}个 → 增强后{len(aug_bboxes_xyxy)}个")

    # 手动添加彩色遮挡（替代CoarseDropout）
    if random.random() < 0.35:
        aug_img = add_colorful_dropout(aug_img)

    # 调整增强后图片到目标尺寸，并更新bbox
    resized_aug_img, _ = resize_image_and_bboxes(aug_img, [], TARGET_SIZE)
    # 单独处理增强后的bbox缩放
    aug_img_h, aug_img_w = aug_img.shape[:2]
    scaled_bboxes_xyxy = []
    for i, (x1, y1, x2, y2) in enumerate(aug_bboxes_xyxy):
        # 缩放bbox到目标尺寸
        scale_x = TARGET_SIZE[0] / aug_img_w
        scale_y = TARGET_SIZE[1] / aug_img_h
        new_x1 = x1 * scale_x
        new_y1 = y1 * scale_y
        new_x2 = x2 * scale_x
        new_y2 = y2 * scale_y
        scaled_bboxes_xyxy.append([new_x1, new_y1, new_x2, new_y2, aug_labels[i]])
    # 转换回YOLO格式
    final_boxes = xyxy2xywh(scaled_bboxes_xyxy, TARGET_SIZE[0], TARGET_SIZE[1])

    # 保存增强并缩放后的图片和标签（仅当有Bbox时）
    if not (REQUIRE_BBOX and len(final_boxes) == 0):
        cv2.imwrite(output_img_path, resized_aug_img)
        write_success = write_yolo_label(output_label_path, final_boxes)

        # 如果标签写入失败（无Bbox），删除图片
        if not write_success and os.path.exists(output_img_path):
            os.remove(output_img_path)
            write_log(f"删除无Bbox的增强图片：{output_img_path}")
    else:
        write_log(f"跳过无Bbox增强图片保存：{output_img_path}")
        return False

    # 可视化（使用缩放后的图片和bbox）
    if vis_path and SAVE_VISUALIZATION and len(final_boxes) > 0:
        vis_img = resized_aug_img.copy()
        img_h_new, img_w_new = resized_aug_img.shape[:2]
        for box in final_boxes:
            cls_id, x, y, bw, bh = box
            cls_id = int(cls_id)
            x1 = max(0, int((x - bw / 2) * img_w_new))
            y1 = max(0, int((y - bh / 2) * img_h_new))
            x2 = min(img_w_new, int((x + bw / 2) * img_w_new))
            y2 = min(img_h_new, int((y + bh / 2) * img_h_new))

            color = (0, 255, 0) if 0 <= cls_id < len(CLASSES) else (0, 0, 255)
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
            label_text = CLASSES[cls_id][:3] if 0 <= cls_id < len(CLASSES) else "???"
            cv2.putText(vis_img, label_text, (x1, max(y1 - 8, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imwrite(vis_path, vis_img)
    elif vis_path and os.path.exists(vis_path):
        os.remove(vis_path)  # 删除无Bbox的可视化图片

    return True


# 处理原始图片的尺寸调整和标签更新
def resize_original_file(img_path, label_path, output_img_path, output_label_path):
    """调整原始图片到目标尺寸并更新标签"""
    img = cv2.imread(img_path)
    if img is None:
        write_log(f"无法读取原始图片：{img_path}")
        return False

    boxes = read_yolo_label(label_path)

    # 如果强制要求Bbox且无Bbox，不处理
    if REQUIRE_BBOX and len(boxes) == 0:
        write_log(f"跳过无Bbox原始图片：{img_path}")
        return False

    # 调整尺寸并更新bbox
    resized_img, resized_boxes = resize_image_and_bboxes(img, boxes)

    # 仅当有有效Bbox时保存
    if not (REQUIRE_BBOX and len(resized_boxes) == 0):
        cv2.imwrite(output_img_path, resized_img)
        write_yolo_label(output_label_path, resized_boxes)
        return True
    else:
        write_log(f"跳过保存无Bbox的原始图片：{output_img_path}")
        return False


# 清理无Bbox的文件
def clean_empty_bbox_files():
    """清理输出目录中无Bbox的图片和标签文件"""
    write_log("\n开始清理无Bbox的文件...")

    image_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    # 先收集所有标签文件
    label_files = [f for f in os.listdir(AUG_DIR) if f.lower().endswith('.txt')]

    deleted_count = 0
    # 检查每个标签文件
    for label_file in label_files:
        label_path = os.path.join(AUG_DIR, label_file)
        boxes = read_yolo_label(label_path)

        # 如果无Bbox，删除标签和对应图片
        if REQUIRE_BBOX and len(boxes) == 0:
            # 删除标签文件
            os.remove(label_path)
            deleted_count += 1
            write_log(f"删除无Bbox标签文件：{label_file}")

            # 删除对应图片
            base_name = os.path.splitext(label_file)[0]
            for ext in image_extensions:
                img_path = os.path.join(AUG_DIR, base_name + ext)
                if os.path.exists(img_path):
                    os.remove(img_path)
                    write_log(f"删除对应无Bbox图片：{base_name + ext}")
                    break

    # 检查孤立的图片文件（无对应标签）
    img_files = [f for f in os.listdir(AUG_DIR) if f.lower().endswith(image_extensions)]
    for img_file in img_files:
        base_name = os.path.splitext(img_file)[0]
        label_path = os.path.join(AUG_DIR, base_name + '.txt')
        if not os.path.exists(label_path):
            img_path = os.path.join(AUG_DIR, img_file)
            os.remove(img_path)
            deleted_count += 1
            write_log(f"删除无对应标签的图片：{img_file}")

    write_log(f"清理完成：共删除{deleted_count}个无Bbox/孤立文件")
    return deleted_count


def main():
    image_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    image_files = [f for f in os.listdir(ORIGINAL_DIR)
                   if f.lower().endswith(image_extensions)]

    write_log(f"找到 {len(image_files)} 張圖片，開始增強...")
    print(f"找到 {len(image_files)} 張圖片，開始增強...")
    print(f"强制要求所有输出文件包含Bbox：{REQUIRE_BBOX}")

    valid_image_count = 0  # 统计有Bbox的图片数量
    for img_name in tqdm(image_files, desc="处理进度"):
        base_name = os.path.splitext(img_name)[0]
        img_path = os.path.join(ORIGINAL_DIR, img_name)
        label_path = os.path.join(ORIGINAL_DIR, base_name + '.txt')

        # 检查原始图片是否有Bbox
        original_boxes = read_yolo_label(label_path)
        if REQUIRE_BBOX and len(original_boxes) == 0:
            write_log(f"跳过无Bbox原始图片：{img_name}")
            continue

        valid_image_count += 1

        # 複製并调整原始檔到960×960
        dst_img = os.path.join(AUG_DIR, img_name)
        dst_label = os.path.join(AUG_DIR, base_name + '.txt')
        # 调整原始图片尺寸并更新标签
        resize_original_file(img_path, label_path, dst_img, dst_label)

        # 生成增強版本
        for i in range(ENHANCEMENT_LOOP):
            aug_img_name = f"{base_name}_aug{i + 1}{os.path.splitext(img_name)[1]}"
            aug_label_name = f"{base_name}_aug{i + 1}.txt"

            output_img = os.path.join(AUG_DIR, aug_img_name)
            output_label = os.path.join(AUG_DIR, aug_label_name)

            vis_name = f"{base_name}_aug{i + 1}_vis.jpg" if SAVE_VISUALIZATION else None
            vis_path = os.path.join(SHOW_SAVE_PATH, vis_name) if vis_name else None

            augment_one_image(img_path, label_path, output_img, output_label, vis_path)

    # 最终清理无Bbox的文件
    clean_empty_bbox_files()

    write_log(f"\n增强完成！有效图片（含Bbox）数量：{valid_image_count}")
    print("增強完成！")
    print(f"有效图片（含Bbox）数量：{valid_image_count}")
    if SAVE_VISUALIZATION:
        print(f"可視化結果保存位置：{SHOW_SAVE_PATH}")
    print(f"增强日志保存位置：{LOG_PATH}")
    print(f"所有输出图片尺寸均为：{TARGET_SIZE[0]}×{TARGET_SIZE[1]}")
    print(f"所有输出文件均包含有效Bbox：{REQUIRE_BBOX}")


if __name__ == '__main__':
    main()
