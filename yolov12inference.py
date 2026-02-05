import os
import glob
from PIL import Image
from ultralytics import YOLO


def count_bug_fertilizer(model_path='models/test23.pt'):
    """
    识别图片中的bug和fertilizer，输出数量到终端，并保存带检测框的图片
    """
    # 输入文件夹（存放茶树图片）
    image_input_dir = r'C:\Users\zheng\Desktop\teaimg'

    # 检查输入文件夹
    if not os.path.exists(image_input_dir):
        print(f"❌ 目标文件夹不存在：{image_input_dir}")
        return

    # 输出结果图片目录
    output_dir = os.path.join(os.path.expanduser('~'), 'Desktop', 'YOLOv12_Results')
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 结果图片保存路径：{output_dir}\n")

    # 加载模型并检查类别
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ 模型文件不存在：{model_path}")
    model = YOLO(model_path)
    class_names = model.names  # 获取模型所有类别
    print(f"✅ 模型加载成功，包含类别：{class_names}")

    # 确认目标类别是否在模型中（容错处理）
    target_classes = ['bug', 'fertilizer']
    for cls in target_classes:
        if cls not in class_names.values():
            print(f"⚠️ 模型中未找到'{cls}'类别，可能导致计数为0（请检查类别名称拼写）")

    # 获取所有图片路径
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(image_input_dir, ext)))
    image_paths = list(set(image_paths))  # 去重

    if not image_paths:
        print(f"❌ 未在 {image_input_dir} 找到图片")
        return

    print(f"🔍 找到 {len(image_paths)} 张图片，开始识别...\n")

    # 遍历图片处理
    for img_idx, img_path in enumerate(image_paths, 1):
        img_name = os.path.basename(img_path)
        print(f"----- 处理第 {img_idx} 张：{img_name} -----")

        # 推理
        results = model(img_path)

        # 统计bug和fertilizer数量
        bug_count = 0
        fertilizer_count = 0

        for r in results:
            # 遍历所有检测框
            for box in r.boxes:
                cls_idx = int(box.cls.item())  # 类别索引
                cls_name = class_names[cls_idx]  # 类别名称

                # 计数目标类别
                if cls_name == 'bug':
                    bug_count += 1
                elif cls_name == 'fertilizer':
                    fertilizer_count += 1

            # 保存带检测框的图片（可选，用于验证）
            im_array = r.plot()
            im = Image.fromarray(im_array[..., ::-1])
            output_path = os.path.join(output_dir, f"result_{img_name}")
            im.save(output_path)

        # 终端输出计数结果
        print(f"✅ 识别结果：")
        print(f"   bug数量：{bug_count}")
        print(f"   fertilizer数量：{fertilizer_count}")
        print(f"   结果图片已保存：{output_path}\n")

    print(f"🎉 所有图片处理完成！结果图片保存至：{output_dir}")


if __name__ == "__main__":
    count_bug_fertilizer()