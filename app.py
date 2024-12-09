import gradio as gr
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import cifar10
from PIL import Image
import os

# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize images
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0

# Define class names
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Load pre-trained CIFAR-10 model
model_path = 'image_rec_cnn_model.h5'
try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    raise ValueError(f"Error loading model: {e}")

# Prediction function
def classify_image(source, uploaded_image=None):
    if source == "Upload an image":
        if uploaded_image is None:
            return "Error: No image uploaded.", None, None
        
        # Resize uploaded image
        image_array = np.array(uploaded_image.resize((32, 32))) / 255.0
        displayed_image = uploaded_image.resize((128, 128))
        
        if image_array.shape != (32, 32, 3):
            return "Error: Uploaded image must be 32x32 RGB.", None, None
    
    elif source == "Random image":
        # Randomly select from either train or test set
        dataset_choice = np.random.choice(["train", "test"])
        if dataset_choice == "train":
            index = np.random.randint(0, len(x_train))
            image_array = x_train[index]
        else:
            index = np.random.randint(0, len(x_test))
            image_array = x_test[index]

        # Display image without color modification
        displayed_image = Image.fromarray((image_array * 255).astype(np.uint8)).resize((512, 512))

    # Model prediction using image_array
    prediction = model.predict(np.expand_dims(image_array, axis=0))
    pred_proba = prediction.flatten()

    # Sort predictions and prepare result
    top_indices = np.argsort(pred_proba)[::-1]
    top_classes = [class_names[i] for i in top_indices]
    top_probs = pred_proba[top_indices]

    result = {f"{top_classes[i]}": float(top_probs[i]) for i in range(len(top_classes))}

    # Plot probabilities
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_classes, y=top_probs)
    plt.title("Prediction Probability Distribution")
    plt.xlabel("Class")
    plt.ylabel("Probability")
    plt.ylim(0, 1)
    for i, prob in enumerate(top_probs):
        plt.text(i, prob + 0.02, f"{prob:.2%}", ha='center', fontsize=10)

    return result, plt, displayed_image


# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# CIFAR-10 Multi-Class Image Classification")
    gr.Markdown("This app classifies images into one of ten classes.")

    with gr.Row():
        source_dropdown = gr.Dropdown(
            choices=["Upload an image", "Random image"],
            value="Upload an image",
            label="Image Source"
        )
        image_input = gr.Image(type="pil", label="Upload an image (if selected)")

    with gr.Row():
        output_text = gr.Label(label="Prediction Results")
        output_plot = gr.Plot()
        random_image_display = gr.Image(label="Random/Uploaded Image Display")

    classify_button = gr.Button("Classify Image")
    classify_button.click(
        classify_image,
        inputs=[source_dropdown, image_input],
        outputs=[output_text, output_plot, random_image_display]
    )

demo.launch()
