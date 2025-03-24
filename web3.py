import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import sys

# sys.path.append("D:\YOLO_E\Streamlit")  # 确保 biformer.py 在这个路径下
# streamlit run D:\YOLO_E\Streamlit\web3.py
# 加载 YOLO 模型
model = YOLO("best.pt", task="detect")

# Streamlit 页面设置
st.set_page_config(page_title="YOLOv8-BCD for lung nodule detection", layout="wide")
st.title("🩺 YOLOv8-BCD-based Lung Nodule Detection System")
st.write("**Upload CT images, and the model will automatically detect lung nodules, providing bounding boxes and confidence level information.**")

# 侧边栏参数
st.sidebar.header("🔧 Setting")
conf_threshold = st.sidebar.slider("Confidence threshold", 0.1, 1.0, 0.25, 0.05)

# **支持多张图片上传**
uploaded_files = st.file_uploader("📂 Upload images (multiple images are supported).", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    for i, uploaded_file in enumerate(uploaded_files):
        st.subheader(f"📷 Process files：{uploaded_file.name}")

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
        st.image(result_image, caption=f"📊 The detection results of {uploaded_file.name}", use_container_width=True)

        # **显示检测框数据**
        if detections:
            st.write("🔍 **Detection results(table)**")
            st.dataframe(detections)
        else:
            st.warning(f"⚠️ {uploaded_file.name} No nodules were detected. Please try adjusting the confidence threshold or uploading other images.")

        # **保存检测结果**
        result_image.save(f"detection_result_{i}.png")

        # **提供下载按钮**
        with open(f"detection_result_{i}.png", "rb") as file:
            st.download_button(
                label=f"📸 Download the result of {uploaded_file.name} ",
                data=file,
                file_name=f"detection_result_{uploaded_file.name}.png",
                mime="image/png",
                key=f"download_{i}"  # 确保 key 唯一，避免 StreamlitDuplicateElementId 错误
            )

st.sidebar.info("💡 **Usage tips**\n- Upload images(multiple images are supported)\n- Adjust the confidence threshold\n- Download the annotated detection results")
