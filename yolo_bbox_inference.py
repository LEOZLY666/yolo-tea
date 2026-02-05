from ultralytics import YOLO
from pathlib import Path
import cv2

# ===================== 配置区域 =====================
MODEL_PATH = r"C:\LEOWork\Pycharm\Projects\yolo\pt\test23.pt"  # ← 改成你的检测模型路径
IMG_DIR    = r"C:\Users\zheng\Desktop\img"
SAVE_DIR   = r"C:\Users\zheng\Desktop\img_results_bbox"               # 结果保存位置

CONF_THRES = 0.3    
IOU_THRES  = 0.45
IMG_SIZE   = 640
DEVICE     = "0"            # "0" = GPU0, "cpu" = CPU
# ====================================================

def main():
    # 加载模型
    model = YOLO(MODEL_PATH)
    print(f"模型加载完成：{MODEL_PATH}")
    print(f"类别名称：{model.names}\n")

    # 准备保存路径
    save_path = Path(SAVE_DIR)
    save_path.mkdir(parents=True, exist_ok=True)

    # 支持的图片后缀
    img_suffix = (".jpg", ".jpeg", ".png", ".bmp")

    img_list = [p for p in Path(IMG_DIR).glob("*") if p.suffix.lower() in img_suffix]

    if not img_list:
        print("文件夹内没有找到图片！")
        return

    print(f"找到 {len(img_list)} 张图片，开始推理...\n")

    # 批量推理（也可以一张一张来）
    results = model(
        source=img_list,
        conf=CONF_THRES,
        iou=IOU_THRES,
        imgsz=IMG_SIZE,
        device=DEVICE,
        save=False,         # 我们自己控制保存
        save_txt=False,
        save_conf=False,
        verbose=True
    )

    # 逐张处理结果并保存
    for result, img_path in zip(results, img_list):
        # result 是一个 Results 对象
        img = cv2.imread(str(img_path))

        # 画框
        annotated_img = result.plot()   # 自动画框+标签+置信度

        # 保存
        save_name = save_path / f"{img_path.stem}_pred{img_path.suffix}"
        cv2.imwrite(str(save_name), annotated_img)
        print(f"已保存：{save_name.name}")

    print("\n全部推理完成！")


if __name__ == "__main__":
    main()