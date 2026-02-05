import os
import random
from pathlib import Path
import yaml
from collections import defaultdict
from ultralytics import YOLO
import ultralytics

print(f"Ultralytics 版本: {ultralytics.__version__}")  # 确认版本

# ────────────────────────────────────────────────
# 全局配置
# ────────────────────────────────────────────────
DATA_ROOT = r"C:\Users\zheng\Desktop\disea"

# 使用官方 YOLO26s（自动下载最新版）
MODEL = "yolo26s.pt"   # Ultralytics 会自动从 https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt 下载

IMG_SIZE = 640
BATCH = 8
EPOCHS = 100
FOLDS = 5
SEED = 42
DEVICE = "0"

OUTPUT_DIR = os.path.join(DATA_ROOT, "kfold_results")
FOLDS_YAML_DIR = os.path.join(DATA_ROOT, "folds_yaml")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FOLDS_YAML_DIR, exist_ok=True)

CLASS_NAMES = [
    "Black rot of tea",
    "Brown blight of tea",
    "Leaf rust of tea",
    "Red Spider infested tea leaf",
    "Tea Mosquito bug infested leaf",
    "Tea leaf",
    "White spot of tea"
]
NC = len(CLASS_NAMES)
# ────────────────────────────────────────────────


def collect_images_per_class():
    class_images = defaultdict(list)

    for cls_id in range(NC):
        cls_dir = os.path.join(DATA_ROOT, str(cls_id))
        img_dir = os.path.join(cls_dir, "images")
        lbl_dir = os.path.join(cls_dir, "labels")

        if not os.path.isdir(img_dir):
            print(f"警告：类别 {cls_id} 目录不存在 → {img_dir}")
            continue

        for img_name in sorted(os.listdir(img_dir)):
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            stem = Path(img_name).stem
            label_path = os.path.join(lbl_dir, stem + ".txt")
            if os.path.isfile(label_path):
                class_images[cls_id].append(img_name)
            else:
                print(f"跳过无标签样本：{cls_id}/images/{img_name}")

    # 打印数量分布
    total = 0
    for cls_id, imgs in sorted(class_images.items()):
        print(f"类别 {cls_id} ({CLASS_NAMES[cls_id]}): {len(imgs)} 张")
        total += len(imgs)
    print(f"总有效样本数：{total}\n")

    return class_images


def stratified_kfold_split(class_images, n_splits=5, seed=42):
    random.seed(seed)
    fold_val_sets = [[] for _ in range(n_splits)]

    for cls_id, img_list in class_images.items():
        if not img_list:
            continue
        random.shuffle(img_list)
        n = len(img_list)
        fold_sizes = [n // n_splits + (1 if i < n % n_splits else 0) for i in range(n_splits)]

        start = 0
        for fold in range(n_splits):
            size = fold_sizes[fold]
            val_slice = img_list[start : start + size]
            fold_val_sets[fold].extend(val_slice)
            start += size

    return fold_val_sets


def create_fold_yaml(fold_idx, train_paths, val_paths):
    train_txt = os.path.join(FOLDS_YAML_DIR, f"fold_{fold_idx+1}_train.txt")
    val_txt   = os.path.join(FOLDS_YAML_DIR, f"fold_{fold_idx+1}_val.txt")

    with open(train_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(train_paths) + "\n")

    with open(val_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(val_paths) + "\n")

    yaml_dict = {
        "path": DATA_ROOT,
        "train": train_txt,
        "val": val_txt,
        "nc": NC,
        "names": CLASS_NAMES
    }

    yaml_path = os.path.join(FOLDS_YAML_DIR, f"fold_{fold_idx+1}.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_dict, f, allow_unicode=True, sort_keys=False)

    print(f"  已生成 yaml: {yaml_path}")
    return yaml_path


def main():
    print("正在收集每类图片...")
    class_images = collect_images_per_class()

    print(f"开始 {FOLDS}-fold 分层划分...")
    fold_val_imgnames = stratified_kfold_split(class_images, FOLDS, SEED)

    # name → full path 映射
    name_to_path = {}
    for cls_id in range(NC):
        cls_img_dir = os.path.join(DATA_ROOT, str(cls_id), "images")
        for name in class_images[cls_id]:
            name_to_path[name] = os.path.join(cls_img_dir, name)

    fold_results = []

    for fold_idx in range(FOLDS):
        print(f"\n━━━━━━━━━━ Fold {fold_idx+1}/{FOLDS} ━━━━━━━━━━")

        val_names = set(fold_val_imgnames[fold_idx])
        val_paths = [name_to_path[name] for name in val_names if name in name_to_path]
        train_paths = [p for name, p in name_to_path.items() if name not in val_names]

        print(f"  训练集: {len(train_paths)} 张")
        print(f"  验证集: {len(val_paths)} 张")

        yaml_path = create_fold_yaml(fold_idx, train_paths, val_paths)

        # 加载 YOLO26s
        try:
            model = YOLO(MODEL)  # 自动下载官方 yolo26s.pt 如果本地没有
            print("模型加载成功")
        except Exception as e:
            print(f"模型加载失败: {e}")
            print("请运行: pip install -U ultralytics")
            continue

        # 训练
        try:
            results = model.train(
                data=yaml_path,
                epochs=EPOCHS,
                imgsz=IMG_SIZE,
                batch=BATCH,
                device=DEVICE,
                project=OUTPUT_DIR,
                name=f"fold_{fold_idx+1}",
                exist_ok=True,
                seed=SEED,
                patience=30,
                optimizer="auto",
                lr0=0.01,
                lrf=0.01,
                cos_lr=True,
                amp=True,
                workers=0,
                cache=False,
                mosaic=1.0,
                mixup=0.0,
                hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
                degrees=5.0,
                translate=0.1,
                scale=0.5,
                shear=0.0,
                flipud=0.5,
                fliplr=0.5,
                close_mosaic=10,
            )

            metrics = results.results_dict
            fold_results.append({
                "fold": fold_idx + 1,
                "mAP50": metrics.get("metrics/mAP50(B)", 0),
                "mAP": metrics.get("metrics/mAP50-95(B)", 0),
            })

            print(f"  Fold {fold_idx+1} 完成    mAP50: {metrics.get('metrics/mAP50(B)', 0):.4f}   mAP: {metrics.get('metrics/mAP50-95(B)', 0):.4f}")

        except Exception as e:
            print(f"  Fold {fold_idx+1} 训练失败！{e}")
            continue

    # 汇总
    if fold_results:
        print("\n" + "═" * 50)
        print("5-Fold 交叉验证汇总结果 (YOLO26s)")
        print("═" * 50)

        avg_map50 = sum(r["mAP50"] for r in fold_results) / len(fold_results)
        avg_map   = sum(r["mAP"]   for r in fold_results) / len(fold_results)

        print(f"平均 mAP@50     : {avg_map50:.4f}")
        print(f"平均 mAP@50:95   : {avg_map:.4f}\n")

        for r in sorted(fold_results, key=lambda x: x["fold"]):
            print(f"Fold {r['fold']:2d} : mAP50 = {r['mAP50']:.4f} | mAP = {r['mAP']:.4f}")

        print(f"\n结果保存于: {OUTPUT_DIR}")
    else:
        print("所有折训练失败，请检查日志")


if __name__ == '__main__':
    random.seed(SEED)
    main()