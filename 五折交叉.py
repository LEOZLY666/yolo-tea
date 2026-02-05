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
AUG_DIR = r"C:\Users\zheng\Desktop\labels_aug"          # 混合图片+标签目录

DATA_ROOT = AUG_DIR                                     # yaml里用的path

MODEL = r"C:\LEOWork\Pycharm\Projects\yolo\model\yolo26s.pt"

IMG_SIZE = 640
BATCH = 8
EPOCHS = 100
FOLDS = 5
SEED = 42
DEVICE = "0"

OUTPUT_DIR = os.path.join(AUG_DIR, "kfold_results")
FOLDS_YAML_DIR = os.path.join(AUG_DIR, "folds_yaml")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FOLDS_YAML_DIR, exist_ok=True)

CLASS_NAMES = [
    'algal leaf',
    'Anthracnose',
    'bird eye spot',
    'brown blight',
    'gray light',
    'healthy',
    'red leaf spot',
    'white spot'
]
NC = len(CLASS_NAMES)
# ────────────────────────────────────────────────


def collect_images_per_class():
    """
    从同一个文件夹读取所有图片 + txt
    根据标签文件**第一行**的类别编号来分类（因为你说每个图片只有一个标签）
    """
    class_images = defaultdict(list)
    img_dir = AUG_DIR
    lbl_dir = AUG_DIR

    if not os.path.isdir(img_dir):
        print(f"错误：目录不存在 → {img_dir}")
        return class_images

    for img_name in sorted(os.listdir(img_dir)):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        stem = Path(img_name).stem
        label_path = os.path.join(lbl_dir, stem + ".txt")

        if not os.path.isfile(label_path):
            print(f"跳过无标签：{img_name}")
            continue

        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]

            if not lines:
                print(f"空标签文件，跳过：{img_name}")
                continue

            # 只看第一行，获取类别id
            first_class_str = lines[0].split()[0]
            cls_id = int(first_class_str)

            if not (0 <= cls_id < NC):
                print(f"类别编号 {cls_id} 非法（应为0~{NC-1}），跳过：{img_name}")
                continue

            class_images[cls_id].append(img_name)

        except (ValueError, IndexError) as e:
            print(f"标签格式错误 {label_path}：{e} → 跳过 {img_name}")
        except Exception as e:
            print(f"读取标签失败 {label_path}：{e} → 跳过")

    # 打印分布
    total = 0
    print("\n类别分布：")
    print("-" * 60)
    for cls_id in range(NC):
        cnt = len(class_images[cls_id])
        print(f"  {cls_id:2d} | {CLASS_NAMES[cls_id]:<18} | {cnt:5d} 张")
        total += cnt
    print("-" * 60)
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

    print(f"  已生成: {yaml_path}")
    return yaml_path


def main():
    print("正在扫描并按类别统计图片（基于标签第一行）...")
    class_images = collect_images_per_class()

    if sum(len(v) for v in class_images.values()) == 0:
        print("没有找到任何有效带标签的图片，程序退出。")
        return

    print(f"开始 {FOLDS}-fold 分层交叉验证划分...")
    fold_val_imgnames = stratified_kfold_split(class_images, FOLDS, SEED)

    # 图片名 → 完整路径
    name_to_path = {}
    for img_name in os.listdir(AUG_DIR):
        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            name_to_path[img_name] = os.path.join(AUG_DIR, img_name)

    fold_results = []

    for fold_idx in range(FOLDS):
        print(f"\n━━━━━━━━━━ Fold {fold_idx+1} / {FOLDS} ━━━━━━━━━━")

        val_names = set(fold_val_imgnames[fold_idx])
        val_paths = [name_to_path.get(name) for name in val_names if name in name_to_path]
        val_paths = [p for p in val_paths if p is not None]

        train_paths = [p for name, p in name_to_path.items() if name not in val_names]

        print(f"  训练集样本数: {len(train_paths):5d}")
        print(f"  验证集样本数: {len(val_paths):5d}")

        if len(val_paths) == 0 or len(train_paths) == 0:
            print("  该折数据集为空，跳过训练")
            continue

        yaml_path = create_fold_yaml(fold_idx, train_paths, val_paths)

        try:
            model = YOLO(MODEL)
            print("模型加载成功")
        except Exception as e:
            print(f"模型加载失败: {e}")
            print("建议：pip install -U ultralytics")
            continue

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

            print(f"  Fold {fold_idx+1} 完成    mAP50: {metrics.get('metrics/mAP50(B)', 0):.4f}"
                  f"   mAP50-95: {metrics.get('metrics/mAP50-95(B)', 0):.4f}")

        except Exception as e:
            print(f"  Fold {fold_idx+1} 训练失败：{e}")
            continue

    # ── 汇总 ────────────────────────────────────────
    if fold_results:
        print("\n" + "═" * 60)
        print("5-Fold 交叉验证结果汇总 (YOLO26s)")
        print("═" * 60)

        avg_map50 = sum(r["mAP50"] for r in fold_results) / len(fold_results)
        avg_map   = sum(r["mAP"]   for r in fold_results) / len(fold_results)

        print(f"平均 mAP@50     : {avg_map50:.4f}")
        print(f"平均 mAP@50:95   : {avg_map:.4f}\n")

        for r in sorted(fold_results, key=lambda x: x["fold"]):
            print(f"Fold {r['fold']:2d} : mAP50 = {r['mAP50']:.4f} | mAP = {r['mAP']:.4f}")

        print(f"\n所有权重/结果保存在: {OUTPUT_DIR}")
    else:
        print("所有折均训练失败，请检查上面日志")


if __name__ == '__main__':
    random.seed(SEED)
    main()
