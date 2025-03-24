import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import sys

sys.path.append("D:\YOLO_E\Streamlit")  # 确保 biformer.py 在这个路径下
# streamlit run D:\YOLO_E\Streamlit\web3_.py
# 加载 YOLO 模型
model = YOLO(r"best.pt", task="detect")

# Streamlit 页面设置
st.set_page_config(page_title="YOLOv8-BCD 肺部结节检测", layout="wide")
st.title("🩺 YOLOv8-BCD 肺部结节检测系统")
st.write("**上传 CT 图片，模型将自动检测肺部结节，并提供检测框和置信度信息。**")

# 侧边栏参数
st.sidebar.header("🔧 设置")
conf_threshold = st.sidebar.slider("置信度阈值", 0.1, 1.0, 0.5, 0.05)

# **支持多张图片上传**
uploaded_files = st.file_uploader("📂 上传图片（支持多张）", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    for i, uploaded_file in enumerate(uploaded_files):
        st.subheader(f"📷 处理文件：{uploaded_file.name}")

        # 读取图片
        image = Image.open(uploaded_file)
        image_np = np.array(image)

        # 转换为 OpenCV 格式 (BGR)
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # 运行 YOLO 预测
        results = model(image_cv)[0]

        # 画出检测结果
        detections = []
        for box in results.boxes:
            conf = box.conf[0].item()
            if conf < conf_threshold:
                continue  # 过滤低置信度框

            x1, y1, x2, y2 = map(int, box.xyxy[0])  # 获取坐标
            label = f"Nodule: {conf:.2f}"  # 标签信息

            # 画框
            cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_cv, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            detections.append({"X1": x1, "Y1": y1, "X2": x2, "Y2": y2, "置信度": conf})

        # 转换回 RGB 以适应 Streamlit 显示
        image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        result_image = Image.fromarray(image_rgb)

        # **显示检测结果**
        st.image(result_image, caption=f"📊 {uploaded_file.name} 的检测结果", use_column_width=True)

        # **显示检测框数据**
        if detections:
            st.write("🔍 **检测结果（表格）**")
            st.dataframe(detections)
        else:
            st.warning(f"⚠️ {uploaded_file.name} 没有检测到结节，请尝试调整置信度阈值或上传其他图像。")

        # **保存检测结果**
        result_image.save(f"detection_result_{i}.png")

        # **提供下载按钮**
        with open(f"detection_result_{i}.png", "rb") as file:
            st.download_button(
                label=f"📸 下载 {uploaded_file.name} 的检测结果",
                data=file,
                file_name=f"detection_result_{uploaded_file.name}.png",
                mime="image/png",
                key=f"download_{i}"  # 确保 key 唯一，避免 StreamlitDuplicateElementId 错误
            )

st.sidebar.info("💡 **使用提示**\n- 上传 CT 图片（可多张）\n- 调整置信度阈值\n- 下载带标注的检测结果")
