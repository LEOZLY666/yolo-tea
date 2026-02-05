import os
import glob
from PIL import Image
import numpy as np
from ultralytics import YOLO


def two_stage_detection(
        model1_path='models/test23.pt',
        model2_path='models/best_tld.pt'
):
    """
    两阶段检测：先用模型1检测目标区域，再用模型2对区域进行细分识别（修复坐标越界问题）
    """
    # 输入文件夹（存放茶树图片）
    image_input_dir = r'C:\Users\zheng\Desktop\img'

    # 检查输入文件夹
    if not os.path.exists(image_input_dir):
        print(f"❌ 目标文件夹不存在：{image_input_dir}")
        return

    # 两阶段结果保存目录
    base_output_dir = r'C:\Users\zheng\Desktop\YOLOv12_TwoStage_Results'
    stage1_dir = os.path.join(base_output_dir, 'Stage1_Disease_Detection')
    stage2_dir = os.path.join(base_output_dir, 'Stage2_Detail_Classification')
    os.makedirs(stage1_dir, exist_ok=True)
    os.makedirs(stage2_dir, exist_ok=True)
    print(f"📁 第一阶段结果保存：{stage1_dir}")
    print(f"📁 第二阶段结果保存：{stage2_dir}\n")

    # 加载模型1并检查
    if not os.path.exists(model1_path):
        raise FileNotFoundError(f"❌ 模型1文件不存在：{model1_path}")
    model1 = YOLO(model1_path)
    model1_classes = model1.names
    print(f"✅ 模型1加载成功，包含类别：{model1_classes}")

    # 加载模型2并检查
    if not os.path.exists(model2_path):
        raise FileNotFoundError(f"❌ 模型2文件不存在：{model2_path}")
    model2 = YOLO(model2_path)
    model2_classes = model2.names
    model2_class_list = list(model2_classes.values())  # 模型2类别列表
    print(f"✅ 模型2加载成功，包含类别：{model2_class_list}\n")

    # 确认目标类别是否存在（容错处理）
    model1_targets = ['bug', 'fertilizer']
    for cls in model1_targets:
        if cls not in model1_classes.values():
            print(f"⚠️ 模型1中未找到'{cls}'类别，可能导致计数为0（请检查拼写）")

    model2_targets = [
        'Black rot of tea', 'Brown blight of tea', 'Leaf rust of tea',
        'Red Spider infested tea leaf', 'Tea Mosquito bug infested leaf',
        'Tea leaf', 'White spot of tea', 'disease'
    ]
    for cls in model2_targets:
        if cls not in model2_class_list:
            print(f"⚠️ 模型2中未找到'{cls}'类别，可能导致计数为0（请检查拼写）")

    # 获取所有图片路径
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(image_input_dir, ext)))
    image_paths = list(set(image_paths))  # 去重

    if not image_paths:
        print(f"❌ 未在 {image_input_dir} 找到图片")
        return

    print(f"🔍 找到 {len(image_paths)} 张图片，开始两阶段识别...\n")

    # 遍历图片处理
    for img_idx, img_path in enumerate(image_paths, 1):
        img_name = os.path.basename(img_path)
        print(f"===== 处理第 {img_idx} 张：{img_name} =====")

        try:
            # 加载原图（确保图片正常打开）
            original_img = Image.open(img_path).convert('RGB')
            original_np = np.array(original_img)  # 转为numpy数组（shape: [高, 宽, 3]）
            img_height, img_width = original_np.shape[0], original_np.shape[1]
            print(f"   原图尺寸：宽={img_width}, 高={img_height}")

            # -------------------------- 第一阶段：模型1推理 --------------------------
            model1_results = model1(img_path, verbose=False)  # verbose=False关闭默认推理日志

            # 统计模型1目标数量 + 筛选有效检测框（新增置信度记录）
            model1_counts = {cls: 0 for cls in model1_targets}
            valid_boxes = []  # 新增置信度字段：(x1, y1, x2, y2, 类别名, 框索引, 置信度)
            box_details = []  # 用于记录每个框的详细信息（写入txt）

            for r in model1_results:
                # 遍历所有检测框
                for box_idx, box in enumerate(r.boxes):
                    cls_idx = int(box.cls.item())
                    cls_name = model1_classes[cls_idx]
                    confidence = box.conf.item()  # 获取置信度

                    # 只处理模型1的目标类别（bug/fertilizer）
                    if cls_name not in model1_targets:
                        continue

                    # 获取绝对坐标（YOLO输出为[x1, y1, x2, y2]，对应左上角-右下角）
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                    # ---------------- 关键修复：坐标有效性校验 ----------------
                    # 1. 确保坐标在图片范围内
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(img_width, x2)  # x2不超过图片宽度
                    y2 = min(img_height, y2)  # y2不超过图片高度
                    # 2. 确保裁剪区域有有效尺寸（宽>0且高>0）
                    crop_width = x2 - x1
                    crop_height = y2 - y1
                    if crop_width <= 0 or crop_height <= 0:
                        print(f"   ⚠️ 跳过无效检测框{box_idx}（尺寸异常：宽={crop_width}, 高={crop_height}）")
                        continue

                    # 记录有效框、计数及详细信息
                    model1_counts[cls_name] += 1
                    valid_boxes.append((x1, y1, x2, y2, cls_name, box_idx, confidence))
                    # 保存框详细信息（用于txt）
                    box_details.append({
                        "box_idx": box_idx,
                        "class": cls_name,
                        "confidence": confidence,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "width": crop_width,
                        "height": crop_height
                    })

                # 保存模型1的整体检测结果图
                model1_plot = r.plot()
                model1_result_img = Image.fromarray(model1_plot[..., ::-1])  # BGR转RGB
                model1_output_path = os.path.join(stage1_dir, f"stage1_overall_{img_name}")
                model1_result_img.save(model1_output_path)

            # 输出模型1结果
            print("\n----- 第一阶段检测结果 -----")
            print(f"   bug数量：{model1_counts['bug']}")
            print(f"   fertilizer数量：{model1_counts['fertilizer']}")
            print(f"   有效检测框数量：{len(valid_boxes)}")
            print(f"   整体检测图保存：{model1_output_path}")

            # ---------------- 新增：保存第一阶段推理结果到txt ----------------
            stage1_txt_path = os.path.join(stage1_dir, f"stage1_stats_{os.path.splitext(img_name)[0]}.txt")
            with open(stage1_txt_path, 'w', encoding='utf-8') as f:
                f.write(f"===== 第一阶段推理结果：{img_name} =====\n")
                f.write(f"原图尺寸：宽={img_width}, 高={img_height}\n\n")
                f.write("【类别数量统计】\n")
                for cls, count in model1_counts.items():
                    f.write(f"{cls}：{count}个\n")
                f.write(f"\n有效检测框总数：{len(valid_boxes)}\n\n")

                f.write("【每个检测框详细信息】\n")
                f.write("框索引 | 类别 | 置信度 | 左上角坐标(x1,y1) | 右下角坐标(x2,y2) | 宽 | 高\n")
                f.write("-" * 100 + "\n")
                for detail in box_details:
                    f.write(
                        f"{detail['box_idx']:6d} | {detail['class']:10s} | {detail['confidence']:.4f} | "
                        f"({detail['x1']},{detail['y1']}) | ({detail['x2']},{detail['y2']}) | "
                        f"{detail['width']} | {detail['height']}\n"
                    )
            print(f"   第一阶段结果txt保存：{stage1_txt_path}")

            # 如果没有有效检测框，跳过第二阶段
            if not valid_boxes:
                print("⚠️ 未检测到有效目标区域，跳过第二阶段分析\n")
                continue

            # -------------------------- 第二阶段：模型2推理 --------------------------
            # 初始化模型2类别计数器
            model2_counts = {cls: 0 for cls in model2_class_list}

            # 处理每个有效检测框（添加异常捕获，避免单个框出错导致程序崩溃）
            for (x1, y1, x2, y2, cls1_name, box_idx, confidence) in valid_boxes:  # 新增confidence参数
                try:
                    # 裁剪目标区域（已通过有效性校验，可安全裁剪）
                    cropped_img = original_np[y1:y2, x1:x2]  # [高, 宽, 3]
                    crop_height, crop_width = cropped_img.shape[0], cropped_img.shape[1]
                    print(
                        f"   处理检测框{box_idx}（{cls1_name}，置信度：{confidence:.4f}）：裁剪尺寸={crop_width}x{crop_height}")

                    # 1. 保存第一阶段裁剪的区域
                    stage1_crop = Image.fromarray(cropped_img)
                    crop_name = f"stage1_{img_name}_box{box_idx}_{cls1_name}.png"
                    stage1_crop_path = os.path.join(stage1_dir, crop_name)
                    stage1_crop.save(stage1_crop_path)
                    print(f"     ✅ 第一阶段裁剪图保存：{os.path.basename(stage1_crop_path)}")

                    # 2. 用模型2推理裁剪区域（verbose=False关闭默认日志）
                    model2_results = model2(cropped_img, verbose=False)

                    # 统计模型2类别 + 保存结果图
                    for r2 in model2_results:
                        # 统计细分类别数量
                        for box2 in r2.boxes:
                            cls2_idx = int(box2.cls.item())
                            cls2_name = model2_classes[cls2_idx]
                            model2_counts[cls2_name] += 1

                        # 保存模型2处理后的区域
                        model2_plot = r2.plot()
                        model2_crop = Image.fromarray(model2_plot[..., ::-1])  # BGR转RGB
                        model2_crop_name = f"stage2_{img_name}_box{box_idx}_{cls1_name}.png"
                        model2_crop_path = os.path.join(stage2_dir, model2_crop_name)
                        model2_crop.save(model2_crop_path)
                        print(f"     ✅ 第二阶段细分图保存：{os.path.basename(model2_crop_path)}")

                except Exception as e:
                    print(f"   ❌ 检测框{box_idx}处理失败：{str(e)}（跳过该框，继续处理下一个）")
                    continue

            # 输出模型2结果
            print("\n----- 第二阶段细分结果 -----")
            for cls, count in model2_counts.items():
                print(f"   {cls}：{count}")
            print(f"   细分区域图保存至：{stage2_dir}\n")

        except Exception as e:
            print(f"❌ 图片{img_name}处理失败：{str(e)}（跳过该图片，继续处理下一张）\n")
            continue

    print(f"🎉 所有图片处理完成！")
    print(f"   第一阶段结果（整体检测+裁剪区域+统计txt）：{stage1_dir}")
    print(f"   第二阶段结果（细分识别区域）：{stage2_dir}")


if __name__ == "__main__":
    two_stage_detection()