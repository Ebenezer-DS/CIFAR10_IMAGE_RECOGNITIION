import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
from tensorflow.keras.datasets import cifar10

# Function to load and merge the chunks for x_train
def load_chunks(chunk_path, num_chunks, data_type='x_train'):
    data = []
    for i in range(num_chunks):
        chunk_file = f'{chunk_path}/{data_type}_chunk_{i}.npy'
        if os.path.exists(chunk_file):
            chunk_data = np.load(chunk_file)
            data.append(chunk_data)
        else:
            st.error(f"Chunk {i} of {data_type} does not exist.")
            return None
    return np.concatenate(data, axis=0)

# Path to the folder containing the chunks
chunk_path = 'cifar10'
num_chunks = 4

# Load CIFAR-10 training data (using chunks for x_train)
x_train = load_chunks(chunk_path, num_chunks, data_type='x_train')
y_train = np.load(f'{chunk_path}/y_train.npy')

if x_train is None or y_train is None:
    st.stop()

# Normalize x_train by dividing by 255 (no need to specify astype)
x_train = x_train / 255.0  # Direct normalization

# Load CIFAR-10 test data (full test set)
(_, _), (x_test, y_test) = cifar10.load_data()

# Normalize x_test by dividing by 255 (accessing the correct data in the tuple)
x_test = x_test / 255.0  # Access the image data and normalize

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

# Initialize placeholder for prediction
prediction = None  # Ensure prediction is always defined

# Upload an image or select a random one
option = st.selectbox("Choose an option:", ("Upload an image", "Select a random test image"))

if option == "Upload an image":
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Uploaded Image", use_column_width=True)
        
        # Resize the image to CIFAR-10 dimensions for prediction
        image_resized = image.resize((32, 32))
        image_resized = np.array(image_resized) / 255.0
        
        if len(image_resized.shape) != 3 or image_resized.shape[2] != 3:
            st.error("Uploaded image must have 3 color channels (RGB).")
            st.stop()

        # Display resized image for comparison
        st.image(image_resized, caption="Resized Image (32x32) for Model Input", use_column_width=True, clamp=True)

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