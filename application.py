import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -------------------------------
# 1. Load Model
# -------------------------------
@st.cache_resource  # cache model so it doesn't reload every time
def load_model():
    return tf.keras.models.load_model("plantDisease.keras")

model = load_model()

# Define your class names (same order as training)
class_names = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Corn_(maize)_Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)Common_rust",
    "Corn_(maize)_Northern_Leaf_Blight",
    "Corn_(maize)_healthy",
    "Pepper_bell__Bacterial_spot",
    "Pepper_bell__healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato_Tomato_YellowLeaf_Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]
idx_to_class = {i: c for i, c in enumerate(class_names)}

# -------------------------------
# 2. Preprocessing
# -------------------------------
def preprocess(img: Image.Image):
    img = img.convert("RGB")
    img = img.resize((224,224))
    arr = np.array(img)/255.0
    arr = np.expand_dims(arr,0)
    return arr

# -------------------------------
# 3. Prediction Function
# -------------------------------
def predict(img: Image.Image):
    x = preprocess(img)
    preds = model.predict(x)[0]
    top3_idx = preds.argsort()[-3:][::-1]
    results = {idx_to_class[int(i)]: float(preds[int(i)]) for i in top3_idx}
    return results

# -------------------------------
# 4. Streamlit UI
# -------------------------------
st.set_page_config(page_title="🌱 Plant Disease Classifier", layout="centered")

st.title("🌱 Plant Disease Classifier")
st.write("Upload a plant leaf image (jpg/png/bmp). The model will predict the disease.")

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("Predicting...")
    results = predict(image)

    st.subheader("Top Predictions")
    for cls, prob in results.items():
        st.write(f"**{cls}** : {prob:.2%}")
