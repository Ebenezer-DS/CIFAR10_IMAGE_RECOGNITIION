import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import cifar10
from PIL import Image

# Load CIFAR-10 test data directly
(_, _), (x_test, y_test) = cifar10.load_data()

# Normalize x_test for inference
x_test = x_test.astype(np.float32) / 255.0

# Define class names
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Load pre-trained CIFAR-10 model
model_path = 'image_rec_cnn_model.h5'
try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# App title and description
st.title("CIFAR-10 Multi-Class Image Classification")
st.write("This app classifies images from the CIFAR-10 dataset into one of ten classes.")
st.write("**Choose an option below to test the model's performance.**")

# Upload an image or select a random one
option = st.selectbox("Choose an option:", ("Upload an image", "Select a random test image"))

if option == "Upload an image":
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Uploaded Image", use_column_width=True)
        
        # Resize and normalize the image
        image_resized = image.resize((32, 32))
        image_resized = np.array(image_resized).astype(np.float32) / 255.0

        if image_resized.shape != (32, 32, 3):
            st.error("Uploaded image must have dimensions (32x32) and 3 color channels (RGB).")
            st.stop()

        # Make the prediction
        prediction = model.predict(np.expand_dims(image_resized, axis=0))
else:
    random_idx = np.random.randint(0, len(x_test))
    image = x_test[random_idx]

    # Make the prediction
    prediction = model.predict(np.expand_dims(image, axis=0))
    st.image(image, caption="Random Test Image", use_column_width=True)

# Display prediction results
if prediction is not None:
    pred_proba = prediction.flatten()

    # Sort by probability and get top predictions
    top_indices = np.argsort(pred_proba)[::-1]  # Descending order
    top_classes = [class_names[i] for i in top_indices]
    top_probs = pred_proba[top_indices]

    st.subheader("Top Predictions")
    for i in range(len(top_classes)):
        st.write(f"{top_classes[i]}: {top_probs[i]:.2%}")

    st.subheader("Prediction Probabilities")
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=top_classes, y=top_probs, ax=ax, palette="coolwarm")
    ax.set_title("Prediction Probability Distribution", fontsize=16)
    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Probability", fontsize=12)
    ax.set_ylim(0, 1)
    for i, prob in enumerate(top_probs):
        ax.text(i, prob + 0.02, f"{prob:.2f}", ha='center', fontsize=10)
    st.pyplot(fig)
else:
    st.warning("Please select an option and ensure a valid image is uploaded.")
