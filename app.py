import os
import time

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

CONF_THRESHOLD = 0.6
FALLING_THRESHOLD = 0.85
MOTION_THRESHOLD = 5.0


@st.cache_resource
def load_model():
    checkpoint = torch.load("abnormal_behaviour_cnn.pth", map_location="cpu")
    class_names = checkpoint["class_names"]

    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )

        def forward(self, x):
            return self.block(x)

    class AbnormalBehaviourCNN(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.features = nn.Sequential(
                ConvBlock(3, 32),
                ConvBlock(32, 64),
                ConvBlock(64, 128),
                ConvBlock(128, 256),
            )
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(128, num_classes),
            )

        def forward(self, x):
            x = self.features(x)
            x = self.avgpool(x)
            x = self.classifier(x)
            return x

    model = AbnormalBehaviourCNN(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, class_names


model, class_names = load_model()

TEST_DIR = "./abnormal_behaviour/test"

test_image_paths = []
for cls in class_names:
    folder = os.path.join(TEST_DIR, cls)
    if os.path.isdir(folder):
        for fname in os.listdir(folder):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                full_path = os.path.join(folder, fname)
                test_image_paths.append((cls, full_path))

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def get_falling_index(class_names_list):
    for i, name in enumerate(class_names_list):
        if name.lower() == "falling":
            return i
    return None


def predict(img: Image.Image):
    img_t = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.softmax(outputs, dim=1).cpu().numpy().flatten()

    falling_idx = get_falling_index(class_names)
    if falling_idx is not None:
        probs[falling_idx] *= 0.15
        probs = probs / probs.sum()

    pred_idx = int(np.argmax(probs))
    pred_class = class_names[pred_idx]
    confidence = float(probs[pred_idx])

    return pred_class, confidence, probs


def apply_thresholds(label: str, confidence: float) -> str:
    if confidence < CONF_THRESHOLD:
        return "No abnormal behaviour detected"
    if label.lower() == "falling" and confidence < FALLING_THRESHOLD:
        return "No abnormal behaviour detected"
    return label


st.set_page_config(page_title="Abnormal Behaviour Classification", layout="wide")

st.title("Abnormal Behaviour Classification")
st.write(
    "This demo uses a custom CNN to classify abnormal human behaviours. "
    "You can select an image from the test dataset, upload an image, "
    "take a snapshot, or use auto-detect mode."
)

st.sidebar.header("About the model")
st.sidebar.markdown(
    f"""
- **Model type:** custom CNN  
- **Number of classes:** {len(class_names)}  
- **Classes:** {", ".join(class_names)}  
- **Input size:** 224 × 224  
"""
)

input_mode = st.sidebar.radio(
    "Input source",
    [
        "Test image (from dataset)",
        "Upload image",
        "Webcam snapshot",
        "Auto-detect (live webcam)",
    ],
    index=0,
)

uploaded_file = None
camera_image = None
test_image_file = None

if input_mode == "Test image (from dataset)":
    if not test_image_paths:
        st.error("No test images found in ./abnormal_behaviour/test")
    else:
        classes_for_select = ["Any"] + list(class_names)
        chosen_class = st.selectbox("Select class", classes_for_select)

        if chosen_class == "Any":
            available = test_image_paths
        else:
            available = [item for item in test_image_paths if item[0] == chosen_class]

        selected_path = st.selectbox(
            "Select test image",
            [p for _, p in available],
            format_func=lambda p: f"{os.path.basename(os.path.dirname(p))} / {os.path.basename(p)}",
        )

        test_image_file = selected_path

elif input_mode == "Upload image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

elif input_mode == "Webcam snapshot":
    camera_image = st.camera_input("Take a picture")

elif input_mode == "Auto-detect (live webcam)":
    st.subheader("Live webcam detection")
    start_btn = st.button("Start detection")
    stop_btn = st.button("Stop detection")
    frame_placeholder = st.empty()
    pred_placeholder = st.empty()

    if "run_cam" not in st.session_state:
        st.session_state.run_cam = False
    if "prev_frame" not in st.session_state:
        st.session_state.prev_frame = None

    if start_btn:
        st.session_state.run_cam = True
    if stop_btn:
        st.session_state.run_cam = False

    if st.session_state.run_cam:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not cap.isOpened():
            st.error("Cannot open webcam. Close other apps that use the camera or try a different index.")
            st.session_state.run_cam = False
        else:
            while st.session_state.run_cam:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Cannot grab frame from webcam. Stopping detection.")
                    st.session_state.run_cam = False
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype("float32")

                motion_low = False
                if st.session_state.prev_frame is not None:
                    diff = np.mean(np.abs(frame_gray - st.session_state.prev_frame))
                    if diff < MOTION_THRESHOLD:
                        motion_low = True

                st.session_state.prev_frame = frame_gray

                pil_img = Image.fromarray(frame_rgb)
                label, conf, _ = predict(pil_img)

                display_label = apply_thresholds(label, conf)
                if motion_low:
                    display_label = "No abnormal behaviour detected"

                frame_placeholder.image(
                    frame_rgb,
                    caption=f"{display_label} ({conf:.2f})",
                    channels="RGB",
                    use_container_width=True,
                )
                pred_placeholder.markdown(
                    f"**Prediction:** {display_label} — `{conf:.2f}`"
                )

                time.sleep(0.2)

            cap.release()
            st.session_state.prev_frame = None

    st.stop()

img = None
if test_image_file is not None:
    img = Image.open(test_image_file).convert("RGB")
elif uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
elif camera_image is not None:
    img = Image.open(camera_image).convert("RGB")

if img is not None:
    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.subheader("Input image")
        st.image(img, use_container_width=True)

    pred_label, confidence, probs = predict(img)
    display_label = apply_thresholds(pred_label, confidence)

    with col2:
        st.subheader("Prediction")
    st.markdown(f"**Predicted class:** `{display_label}`")
    st.markdown(f"**Confidence:** `{confidence:.4f}`")

    prob_df = pd.DataFrame({"class": class_names, "probability": probs})
    prob_df = prob_df.sort_values("probability", ascending=False)

    st.subheader("Class probabilities")
    st.bar_chart(prob_df.set_index("class"))
else:
    st.info("Please select a test image, upload an image, take a snapshot, or start auto-detection.")
