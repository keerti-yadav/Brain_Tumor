import streamlit as st
import torch
import cv2
import numpy as np
import os
import gdown

from models.classifier import EfficientNetClassifier
from models.gradcam import GradCAM
from models.segmentation import UNet
from config import *


os.makedirs("models", exist_ok=True)


cls_path = "models/best_model.pth"

if not os.path.exists(cls_path):
    gdown.download(
        "https://drive.google.com/uc?id=13EZKGPAiy8kLlC0fUbZyu-V--oywp6HJ",
        cls_path,
        quiet=False
    )


seg_path = "models/segmentation_model.pth"

if not os.path.exists(seg_path):
    gdown.download(
        "https://drive.google.com/uc?id=1cplS5pKBXVS2za9Kl-2SzkzC3RyK7QZn",
        seg_path,
        quiet=False
    )

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Brain Tumor Analysis", layout="wide")

CLASS_NAMES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    clf = EfficientNetClassifier(NUM_CLASSES)
    clf.load_state_dict(torch.load("models/best_model.pth", map_location=DEVICE))
    clf.to(DEVICE)
    clf.eval()

    seg = UNet()
    seg.load_state_dict(torch.load("models/segmentation_model.pth", map_location=DEVICE))
    seg.eval()

    return clf, seg

model, seg_model = load_models()
gradcam = GradCAM(model)

# ---------------- PREPROCESS ----------------
def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))

    return torch.from_numpy(img).float().unsqueeze(0).to(DEVICE)

# ---------------- UI ----------------
st.markdown("<h1 style='text-align:center;'>Brain Tumor Analysis</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload MRI Scan", type=["jpg", "png", "jpeg"])

if uploaded_file:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, caption="Original MRI", width=300)

    if st.button("Analyze"):

        # -------- CLASSIFICATION --------
        x = preprocess(img)

        with torch.no_grad():
            outputs = model(x)
            probs = torch.softmax(outputs, dim=1)
            confidence, pred = torch.max(probs, 1)

        predicted_class = CLASS_NAMES[pred.item()]
        confidence_val = confidence.item()

        # -------- SEGMENTATION --------
       
        if predicted_class != "no_tumor":

            img_seg = cv2.resize(img, (224,224)) / 255.0
            img_seg = np.transpose(img_seg, (2,0,1))
            img_tensor_seg = torch.tensor(img_seg).float().unsqueeze(0)

            with torch.no_grad():
                seg_pred = seg_model(img_tensor_seg)
                seg_pred = torch.argmax(seg_pred, dim=1).squeeze().numpy()

            overlay_seg = cv2.resize(img, (224,224))
            overlay_seg[seg_pred == 1] = [255, 0, 0]

        else:
           
            seg_pred = np.zeros((224,224))
            overlay_seg = cv2.resize(img, (224,224))

        # -------- GRADCAM --------
        cam = gradcam.generate(x, pred.item())

     
        cam = np.where(cam > 0.4, cam, 0)

      
        cam = cv2.GaussianBlur(cam, (15, 15), 0)

        heatmap = np.uint8(255 * cam)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_TURBO)

        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

        
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        img = img.astype(np.uint8)

       
        overlay_cam = cv2.addWeighted(img, 0.7, heatmap, 0.3, 0)




        # -------- ALL CLASS CONFIDENCE  --------
        st.markdown("### Class Confidence Scores")

        c1, c2, c3, c4 = st.columns(4)
        for i, col in enumerate([c1, c2, c3, c4]):
            col.metric(CLASS_NAMES[i], f"{probs[0][i].item():.2f}")

        # -------- MAIN RESULT BOX --------
        st.markdown("<br>", unsafe_allow_html=True)

        box = st.container()
        with box:
            st.markdown("### Prediction")
            st.success(f"{predicted_class}  |  Confidence: {confidence_val:.2f}")

        st.markdown("<hr>", unsafe_allow_html=True)

        # -------- VISUALS --------
        st.markdown("### Visual Outputs")

        v1, v2, v3, v4 = st.columns(4)

        with v1:
            st.image(img, caption="Original", width=350)

        with v2:
            seg_display = (seg_pred * 255).astype(np.uint8)
            seg_display = cv2.applyColorMap(seg_display, cv2.COLORMAP_PLASMA)

            st.image(seg_display, caption="Segmentation", width=350)
        with v3:
            st.image(overlay_seg, caption="Overlay", width=350)

        with v4:
            st.image(overlay_cam, caption="GradCAM", width=350)
