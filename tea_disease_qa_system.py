import os
import shutil
import requests
import numpy as np
from PIL import Image
from ultralytics import YOLO
from typing import List, Dict, Optional
from datetime import datetime
from flask import Flask, request, jsonify

# 初始化Flask应用
app = Flask(__name__)

# -------------------------- 1. 配置常量（核心修改：路径调整） --------------------------
# 路径配置：改为D盘uploads目录
UPLOADS_ROOT = "D:/uploads"  # 根目录
INPUT_DIR = os.path.join(UPLOADS_ROOT, "tea_images_input")  # 项目上传的图片目录（供参考，实际路径由请求传入）
OUTPUT_DIR = os.path.join(UPLOADS_ROOT, "tea_images_output")  # 处理后输出目录
CACHE_DIR = os.path.join(UPLOADS_ROOT, "tea_images_cache")  # 缓存目录（处理后删除）

# 确保目录存在（自动创建不存在的文件夹）
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# 模型配置（不变）
MODEL1_PATH = "models/test23.pt"  # 第一阶段模型(识别bug和fertilizer)
MODEL2_PATH = "models/best_tld.pt"  # 第二阶段模型(识别具体病虫害)

# FastGPT接口配置（不变）
FASTGPT_API_URL = "http://localhost:8102/api/fastgpt/ask"
HTTP_TIMEOUT = 60

# 加载模型（启动时预加载）
try:
    model1 = YOLO(MODEL1_PATH)
    model2 = YOLO(MODEL2_PATH)
    model1_classes = model1.names
    model2_classes = model2.names
    model2_class_list = list(model2_classes.values())
    print("模型加载成功")
except Exception as e:
    print(f"模型加载失败: {str(e)}")
    raise


# -------------------------- 2. 工具函数（无需修改） --------------------------
def clean_cache():
    """清理缓存文件夹"""
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
    os.makedirs(CACHE_DIR, exist_ok=True)


def copy_original_image(src_path):
    """复制原始图片到输出目录（输出目录已改为D:/uploads/tea_images_output）"""
    img_name = os.path.basename(src_path)
    dest_path = os.path.join(OUTPUT_DIR, img_name)
    shutil.copy2(src_path, dest_path)
    return dest_path


def calculate_avg_confidence(boxes) -> float:
    """计算平均置信度"""
    if not boxes:
        return 0.0
    confidences = [box.conf.item() for box in boxes]
    return sum(confidences) / len(confidences)


# -------------------------- 3. FastGPT接口调用（不变） --------------------------
def call_fastgpt_api(question: str, user_id: str) -> Optional[str]:
    """调用FastGPT接口获取智能建议"""
    try:
        response = requests.post(
            url=FASTGPT_API_URL,
            json={"question": question, "userId": user_id},
            timeout=HTTP_TIMEOUT
        )
        response.raise_for_status()
        response_data = response.json()

        if response_data.get("code") == 200:
            return response_data.get("data", "未获取到回答")
        else:
            print(f"FastGPT接口错误: {response_data.get('message', '未知错误')}")
            return None
    except Exception as e:
        print(f"FastGPT调用失败: {str(e)}")
        return None


def generate_question(detection_result: Dict) -> str:
    """基于识别结果生成问题"""
    img_name = detection_result["image_name"]
    model2_result = detection_result["model2_result"]
    disease_list = [(cls, count) for cls, count in model2_result.items() if count > 0]

    if not disease_list:
        if detection_result["model1_result"]["bug"] > 0:
            return f"图片{img_name}中检测到{detection_result['model1_result']['bug']}处病虫害，但未识别出具体类型，请提供常见茶叶病虫害的防治建议。"
        elif detection_result["model1_result"]["fertilizer"] > 0:
            return f"图片{img_name}中检测到{detection_result['model1_result']['fertilizer']}处需要施肥的区域，请提供茶叶施肥的建议和方法。"
        else:
            return f"图片{img_name}中未检测到明显的病虫害或需要施肥的区域，请提供茶叶日常养护建议。"
    else:
        disease_desc = "、".join([f"{cls}（{count}处）" for cls, count in disease_list])
        return f"图片{img_name}中检测到{disease_desc}，请分别说明这些茶叶病虫害的病因、危害，以及对应的防治方法（包括农业防治、化学防治、生物防治。中文）。"


# -------------------------- 4. 图片处理函数（无需修改，路径已通过常量自动适配） --------------------------
def process_single_image(img_path: str, user_id: str) -> Dict:
    """处理单张图片并返回结果（不包含数据库操作）"""
    img_name = os.path.basename(img_path)
    print(f"处理图片: {img_name}，用户ID: {user_id}")

    # 初始化结果字典
    result = {
        "success": False,
        "image_name": img_name,
        "model1_result": {"bug": 0, "fertilizer": 0},
        "model2_result": {cls: 0 for cls in model2_class_list},
        "has_disease": False,
        "avg_confidence": 0.0,
        "original_image_path": "",
        "boxed_image_path": "",
        "process_text": "",
        "process_type": 0,
        "error": ""
    }

    try:
        # 检查图片是否存在（确保输入图片在D:/uploads/tea_images_input目录）
        if not os.path.exists(img_path):
            result["error"] = f"图片不存在: {img_path}（请确认图片在{INPUT_DIR}目录下）"
            return result

        # 复制原始图片到输出目录（D:/uploads/tea_images_output）
        original_save_path = copy_original_image(img_path)
        result["original_image_path"] = original_save_path

        # 加载图片
        original_img = Image.open(img_path).convert("RGB")
        original_np = np.array(original_img)
        img_height, img_width = original_np.shape[0], original_np.shape[1]

        # 第一阶段：模型1推理
        model1_results = model1(img_path, verbose=False)
        valid_boxes = []
        all_boxes = []

        for r in model1_results:
            all_boxes.extend(r.boxes)  # 收集所有框用于计算平均置信度

            for box_idx, box in enumerate(r.boxes):
                cls_idx = int(box.cls.item())
                cls_name = model1_classes[cls_idx]

                if cls_name not in ["bug", "fertilizer"]:
                    continue

                # 坐标校验
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img_width, x2), min(img_height, y2)

                if (x2 - x1) <= 0 or (y2 - y1) <= 0:
                    print(f"跳过无效检测框{box_idx}（尺寸异常）")
                    continue

                result["model1_result"][cls_name] += 1
                valid_boxes.append((x1, y1, x2, y2, cls_name, box_idx))

            # 保存模型1标注图片到输出目录（D:/uploads/tea_images_output）
            model1_plot = r.plot()
            model1_result_img = Image.fromarray(model1_plot[..., ::-1])
            boxed_save_path = os.path.join(OUTPUT_DIR, f"boxed_{img_name}")
            model1_result_img.save(boxed_save_path)
            result["boxed_image_path"] = boxed_save_path

        # 计算平均置信度
        result["avg_confidence"] = calculate_avg_confidence(all_boxes)

        # 第二阶段：模型2推理（仅处理有效框，缓存到D:/uploads/tea_images_cache）
        if valid_boxes:
            for (x1, y1, x2, y2, cls1_name, box_idx) in valid_boxes:
                try:
                    # 裁剪区域并保存到缓存目录
                    cropped_img = original_np[y1:y2, x1:x2]
                    crop_name = f"crop_{img_name}_box{box_idx}_{cls1_name}.png"
                    crop_path = os.path.join(CACHE_DIR, crop_name)
                    Image.fromarray(cropped_img).save(crop_path)

                    # 模型2推理
                    model2_results = model2(cropped_img, verbose=False)
                    for r2 in model2_results:
                        for box2 in r2.boxes:
                            cls2_idx = int(box2.cls.item())
                            cls2_name = model2_classes[cls2_idx]
                            result["model2_result"][cls2_name] += 1
                            result["has_disease"] = True

                except Exception as e:
                    print(f"检测框{box_idx}处理失败: {str(e)}（跳过）")
                    continue

        # 生成问题并调用FastGPT
        question = generate_question(result)
        print(f"生成问题: {question[:50]}...")

        answer = call_fastgpt_api(question, user_id)
        if not answer:
            answer = "未能获取到智能建议"
        result["process_text"] = answer

        # 确定处理类型 (1-病虫害 2-施肥)
        result["process_type"] = 1 if result["model1_result"]["bug"] > 0 else 2

        # 标记处理成功
        result["success"] = True
        return result

    except Exception as e:
        error_msg = f"图片处理失败: {str(e)}"
        print(error_msg)
        result["error"] = error_msg
        return result


# -------------------------- 5. API接口（不变，路径通过结果自动返回） --------------------------
@app.route('/api/image-processing/process', methods=['POST'])
def process_image():
    """处理图片的API接口（返回D:/uploads下的路径）"""

    try:
        # 获取请求数据（确保imagePath指向D:/uploads/tea_images_input下的图片）
        data = request.get_json()

        # 验证必要参数
        if not data or "userId" not in data or "imagePath" not in data:
            return jsonify({
                "code": 400,
                "message": "参数错误，需要包含userId和imagePath",
                "data": None
            }), 400

        user_id = str(data["userId"])
        image_path = data["imagePath"]  # 此处应传入D:/uploads/tea_images_input/xxx.png

        # 处理图片
        clean_cache()  # 清理缓存
        result = process_single_image(image_path, user_id)
        clean_cache()  # 处理完成后清理缓存

        if result["success"]:
            # 响应中返回的路径为D:/uploads/tea_images_output下的绝对路径（后续由Java接口转为HTTP路径）
            return jsonify({
                "code": 200,
                "message": "处理成功",
                "data": {
                    "timestamp": datetime.now().isoformat(),
                    "processType": result["process_type"],
                    "processText": result["process_text"],
                    "avgConfidence": round(result["avg_confidence"], 4),
                    "uploadImagePath": result["original_image_path"],  # D:/uploads/tea_images_output/xxx.png
                    "boxedImagePath": result["boxed_image_path"],  # D:/uploads/tea_images_output/boxed_xxx.png
                    "readStatus": 0,
                    "solutionTicketId": None
                }
            })
        else:
            return jsonify({
                "code": 500,
                "message": result["error"],
                "data": None
            }), 500

    except Exception as e:
        return jsonify({
            "code": 500,
            "message": f"接口处理错误: {str(e)}",
            "data": None
        }), 500


# -------------------------- 6. 启动服务（打印新路径方便确认） --------------------------
if __name__ == '__main__':
    print("=" * 60)
    print("          茶叶图片智能处理服务          ")
    print("=" * 60)
    print(f"服务启动，监听端口 5000...")
    print(f"输入图片目录: {INPUT_DIR}")  # 打印输入目录
    print(f"输出图片目录: {OUTPUT_DIR}")  # 打印输出目录
    print(f"缓存目录: {CACHE_DIR}")  # 打印缓存目录
    app.run(host='0.0.0.0', port=5000, debug=True)  # 0.0.0.0允许外部访问