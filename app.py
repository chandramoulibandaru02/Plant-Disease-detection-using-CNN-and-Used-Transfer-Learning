import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# -------------------------------
# 1. Load Model
# -------------------------------
model = tf.keras.models.load_model(r"C:\Users\Chandramouli bandaru\OneDrive\Desktop\plant disease\plantDisease.keras")

# Define your class names in the SAME order used during training
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
]   # <--- replace with your actual class names
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
# 4. Gradio Interface
# -------------------------------
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload leaf image"),
    outputs=gr.Label(num_top_classes=5, label="Top Predictions"),
    title="🌱 Plant Disease Classifier",
    description="Upload a plant leaf image (jpg/png/bmp). Model predicts disease."
)

if __name__ == "__main__":
    iface.launch(share=True)   # gives you a public link
