import os
import random
from pathlib import Path
import yaml
from collections import defaultdict
from ultralytics import YOLO
import ultralytics
import torch

print(f"Ultralytics 版本: {ultralytics.__version__}")
print(f"PyTorch 版本: {torch.__version__}")

# ────────────────────────────────────────────────
# 全局配置（仅保留核心，确保路径绝对且无冗余）
# ────────────────────────────────────────────────
AUG_DIR = Path(r"C:\Users\zheng\Desktop\labels_aug")
DATA_ROOT = AUG_DIR

# 本地yolo26s.pt绝对路径（resolve()确保路径规范化）
MODEL_PATH = Path(r"C:\LEOWork\Pycharm\Projects\yolo\model\yolo26s.pt").resolve()
print(f"本地权重文件绝对路径：{MODEL_PATH}")
print(f"权重文件是否存在：{MODEL_PATH.exists()}")

IMG_SIZE = 960
BATCH = 8
EPOCHS = 20
FOLDS = 5
SEED = 42
DEVICE = "0"  # GPU 0，改为"cpu"可使用CPU

OUTPUT_DIR = AUG_DIR / "kfold_results"
FOLDS_YAML_DIR = AUG_DIR / "folds_yaml"
OUTPUT_DIR.mkdir(exist_ok=True)
FOLDS_YAML_DIR.mkdir(exist_ok=True)

CLASS_NAMES = [
    'brown blight', 'disease', 'healthy', 'White spot'
]

# CLASS_NAMES = [
#     'algal leaf', 'Anthracnose', 'bird eye spot', 'brown blight',
#     'gray light', 'healthy', 'red leaf spot', 'white spot'
# ]
NC = len(CLASS_NAMES)


# ────────────────────────────────────────────────
# 权重校验：仅验证存在性和大小，不修改Ultralytics内部配置
# ────────────────────────────────────────────────
def check_pretrained_weight(weight_path):
    """仅校验本地权重的基础有效性，兼容8.4.11版本"""
    if not weight_path.exists():
        raise FileNotFoundError(f"权重文件不存在：{weight_path}")

    # 计算文件大小（MB）
    file_size = weight_path.stat().st_size / (1024 * 1024)
    # yolo26s.pt官方版本约10-20MB，排除过小的无效文件
    if file_size < 5:
        raise ValueError(f"本地权重文件过小（{file_size:.2f}MB），不是有效yolo26s.pt（应≥10MB）")

    # 验证文件可被PyTorch加载（确保不是损坏文件）
    try:
        torch.load(weight_path, map_location="cpu", weights_only=False)
        print(f"✅ 本地yolo26s.pt校验成功 | 大小：{file_size:.2f}MB")
        return True
    except Exception as e:
        raise RuntimeError(f"本地权重文件损坏，无法加载：{e}")


# ────────────────────────────────────────────────
# 收集图片并按类别分类
# ────────────────────────────────────────────────
def collect_images_per_class():
    class_images = defaultdict(list)
    img_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

    if not AUG_DIR.is_dir():
        print(f"错误：数据集目录不存在 → {AUG_DIR}")
        return class_images

    for img_path in AUG_DIR.glob("*"):
        if img_path.suffix.lower() not in img_extensions:
            continue
        img_name = img_path.name
        stem = img_path.stem
        label_path = AUG_DIR / f"{stem}.txt"

        if not label_path.exists():
            print(f"空标签：{img_name}")
            continue

        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
            if not lines:
                print(f"空标签：{img_name}")
                continue

            first_line_parts = lines[0].split()
            if len(first_line_parts) < 1:
                print(f"标签格式错误（无类别）：{img_name}")
                continue
            cls_id = int(first_line_parts[0])

            if not (0 <= cls_id < NC):
                print(f"类别编号非法（{cls_id}）：{img_name}")
                continue

            class_images[cls_id].append(img_name)
        except Exception as e:
            print(f"读取标签失败：{img_name} → {e}")

    # 打印类别分布
    total = 0
    print("\n类别分布：")
    print("-" * 60)
    for cls_id in range(NC):
        cnt = len(class_images[cls_id])
        print(f"  {cls_id:2d} | {CLASS_NAMES[cls_id]:<18} | {cnt:5d} 张")
        total += cnt
    print("-" * 60)
    print(f"总有效样本：{total}\n")

    return class_images


# ────────────────────────────────────────────────
# 分层K折划分
# ────────────────────────────────────────────────
def stratified_kfold_split(class_images, n_splits=5, seed=42):
    random.seed(seed)
    fold_val_sets = [[] for _ in range(n_splits)]

    for cls_id, img_list in class_images.items():
        if not img_list:
            continue
        random.shuffle(img_list)
        n = len(img_list)
        # 按折数均匀划分样本
        fold_sizes = [n // n_splits + (1 if i < n % n_splits else 0) for i in range(n_splits)]

        start = 0
        for fold in range(n_splits):
            size = fold_sizes[fold]
            fold_val_sets[fold].extend(img_list[start:start + size])
            start += size

    return fold_val_sets


# ────────────────────────────────────────────────
# 生成每个折的YAML配置文件
# ────────────────────────────────────────────────
def create_fold_yaml(fold_idx, train_paths, val_paths):
    train_txt = FOLDS_YAML_DIR / f"fold_{fold_idx + 1}_train.txt"
    val_txt = FOLDS_YAML_DIR / f"fold_{fold_idx + 1}_val.txt"

    # 写入训练/验证集路径（转为字符串，避免Path对象序列化问题）
    with open(train_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(str(p) for p in train_paths) + "\n")
    with open(val_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(str(p) for p in val_paths) + "\n")

    # 构建YOLO训练用的YAML字典
    yaml_dict = {
        "path": str(DATA_ROOT),
        "train": str(train_txt),
        "val": str(val_txt),
        "nc": NC,
        "names": CLASS_NAMES
    }

    yaml_path = FOLDS_YAML_DIR / f"fold_{fold_idx + 1}.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_dict, f, allow_unicode=True, sort_keys=False, indent=2)

    print(f"  已生成：{yaml_path}")
    return yaml_path


# ────────────────────────────────────────────────
# 主训练逻辑（核心：简化模型加载，无内部配置修改）
# ────────────────────────────────────────────────
def main():
    # 第一步：校验本地权重（仅验证有效性，不修改Ultralytics）
    try:
        check_pretrained_weight(MODEL_PATH)
    except Exception as e:
        print(f"权重校验失败：{e}")
        return

    # 第二步：收集并统计图片
    print("正在扫描并按类别统计图片...")
    class_images = collect_images_per_class()
    total_samples = sum(len(v) for v in class_images.values())
    if total_samples == 0:
        print("错误：无有效带标签的样本，程序退出")
        return

    # 第三步：分层K折划分
    print(f"开始 {FOLDS}-fold 分层交叉验证划分...")
    fold_val_imgnames = stratified_kfold_split(class_images, FOLDS, SEED)

    # 构建图片名到绝对路径的映射
    name_to_path = {}
    img_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    for img_path in AUG_DIR.glob("*"):
        if img_path.suffix.lower() in img_extensions:
            name_to_path[img_path.name] = img_path

    fold_results = []

    # 第四步：逐折训练（核心修改：仅加载本地路径，无内部配置修改）
    for fold_idx in range(FOLDS):
        print(f"\n━━━━━━━━━━ Fold {fold_idx + 1} / {FOLDS} ━━━━━━━━━━")

        # 划分训练/验证集
        val_names = set(fold_val_imgnames[fold_idx])
        val_paths = [name_to_path[name] for name in val_names if name in name_to_path]
        train_paths = [p for name, p in name_to_path.items() if name not in val_names]

        # 校验数据集非空
        if len(val_paths) == 0 or len(train_paths) == 0:
            print(f"  警告：训练/验证集为空，跳过该折")
            continue
        print(f"  训练集：{len(train_paths)} | 验证集：{len(val_paths)}")

        # 生成该折的YAML配置
        yaml_path = create_fold_yaml(fold_idx, train_paths, val_paths)

        # 关键：直接加载本地权重（无任何内部配置修改，兼容8.4.11）
        try:
            # 转为字符串路径，避免Path对象解析问题
            model = YOLO(str(MODEL_PATH))
            print(f"  ✅ 成功加载本地yolo26s.pt")
        except Exception as e:
            print(f"  ❌ 加载本地权重失败：{e}")
            print(f"  排查建议：1. 确认{MODEL_PATH}是官方yolo26s.pt；2. 重新下载官方权重")
            continue

        # 开始训练（仅保留8.4.11支持的参数）
        try:
            results = model.train(
                data=yaml_path,
                epochs=EPOCHS,
                imgsz=IMG_SIZE,
                batch=BATCH,
                device=DEVICE,
                project=str(OUTPUT_DIR),
                name=f"fold_{fold_idx + 1}",
                exist_ok=True,
                seed=SEED,
                patience=30,
                optimizer="auto",
                lr0=0.01,
                lrf=0.01,
                cos_lr=True,
                amp=True,  # 混合精度训练，报错可改为False
                workers=0,  # Windows必须设为0，避免多进程错误
                cache=False,
                mosaic=1.0,
                mixup=0.0,
                hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
                degrees=5.0, translate=0.1, scale=0.5, shear=0.0,
                flipud=0.5, fliplr=0.5, close_mosaic=10,
                verbose=False,
                save=True  # 自动保存best.pt和last.pt，无需save_best
            )

            # 提取训练指标
            metrics = results.results_dict
            fold_result = {
                "fold": fold_idx + 1,
                "mAP50": metrics.get("metrics/mAP50(B)", 0.0),
                "mAP50-95": metrics.get("metrics/mAP50-95(B)", 0.0),
            }
            fold_results.append(fold_result)

            print(f"  ✅ Fold {fold_idx + 1} 训练完成")
            print(f"     mAP50: {fold_result['mAP50']:.4f} | mAP50-95: {fold_result['mAP50-95']:.4f}")

        except Exception as e:
            print(f"  ❌ Fold {fold_idx + 1} 训练失败：{e}")
            continue

    # ── 训练结果汇总 ────────────────────────────────────────
    if fold_results:
        print("\n" + "═" * 60)
        print("5-Fold 交叉验证结果汇总（YOLO26s）")
        print("═" * 60)
        avg_map50 = sum(r["mAP50"] for r in fold_results) / len(fold_results)
        avg_map50_95 = sum(r["mAP50-95"] for r in fold_results) / len(fold_results)
        print(f"平均 mAP@50     : {avg_map50:.4f}")
        print(f"平均 mAP@50:95   : {avg_map50_95:.4f}\n")

        for r in sorted(fold_results, key=lambda x: x["fold"]):
            print(f"Fold {r['fold']:2d} : mAP50 = {r['mAP50']:.4f} | mAP50-95 = {r['mAP50-95']:.4f}")

        print(f"\n训练结果保存路径：{OUTPUT_DIR}")
    else:
        print("\n❌ 所有折训练失败，请检查上述错误日志")


if __name__ == '__main__':
    # 固定随机种子，保证结果可复现
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    main()
