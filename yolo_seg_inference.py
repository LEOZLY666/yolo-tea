from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np

# ===================== 配置区域 =====================
MODEL_PATH = r"C:\LEOWork\Pycharm\Projects\yolo\pt\best-seg.pt"     # ← 改成你的分割模型路径
IMG_DIR    = r"C:\Users\zheng\Desktop\img"
SAVE_DIR   = r"C:\Users\zheng\Desktop\img_results_seg"

CONF_THRES = 0.30
IOU_THRES  = 0.45
IMG_SIZE   = 640
DEVICE     = "0"            # "0" 或 "cpu"
MASK_ALPHA = 0.4            # 掩码透明度（0~1）
# ====================================================

def main():
    model = YOLO(MODEL_PATH)
    print(f"分割模型加载完成：{MODEL_PATH}")
    print(f"类别名称：{model.names}\n")

    save_path = Path(SAVE_DIR)
    save_path.mkdir(parents=True, exist_ok=True)

    img_suffix = (".jpg", ".jpeg", ".png", ".bmp")
    img_list = [p for p in Path(IMG_DIR).glob("*") if p.suffix.lower() in img_suffix]

    if not img_list:
        print("没有找到图片！")
        return

    print(f"找到 {len(img_list)} 张图片\n")

    results = model(
        source=img_list,
        conf=CONF_THRES,
        iou=IOU_THRES,
        imgsz=IMG_SIZE,
        device=DEVICE,
        retina_masks=True,      # 更高质量的掩码（推荐）
        save=False,
        verbose=True
    )

    for result, img_path in zip(results, img_list):
        # 原图
        img = cv2.imread(str(img_path))

        # 画框 + 掩码
        annotated_img = result.plot(
            boxes=True,
            masks=True,
            probs=False,
            labels=True,
            masks_alpha=MASK_ALPHA   # 控制掩码透明度
        )

        # 保存
        save_name = save_path / f"{img_path.stem}_seg{img_path.suffix}"
        cv2.imwrite(str(save_name), annotated_img)
        print(f"已保存：{save_name.name}")

    print("\n分割推理全部完成！")


if __name__ == "__main__":
    main()