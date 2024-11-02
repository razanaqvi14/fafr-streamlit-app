import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
import cv2
from PIL import Image
import io
import os

# Load the model
classes = [
    "Cristiano Ronaldo",
    "Erling Haaland",
    "Kylian Mbappe",
    "Lionel Messi",
    "Neymar Jr",
]


def load_model(weights_path):
    model = models.resnet18(weights="IMAGENET1K_V1")
    size_of_last_layer = model.fc.in_features
    model.fc = nn.Linear(size_of_last_layer, len(classes))
    model.load_state_dict(
        torch.load(weights_path),
    )
    model.eval()
    return model


model = load_model(
    os.path.join(
        os.path.dirname(__file__),
        "model",
        "TL_CNN_model_weights.pth",
    )
)

# Image transformations
image_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

softmax = nn.Softmax(dim=1)


# Function to crop the face
def crop_photo(image):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    faces = face_cascade.detectMultiScale(image)
    for x, y, w, h in faces:
        roi_color = image[y : y + h, x : x + w]
        eyes = eyes_cascade.detectMultiScale(roi_color)
        if len(eyes) >= 2:
            return roi_color
    return None


# Prediction function
def predict(cropped_image):
    transformed_image = image_transform(cropped_image).unsqueeze(0)
    with torch.no_grad():
        output = softmax(model(transformed_image))
        probabilities = {
            classes[i]: round(output[0, i].item(), 3) for i in range(len(classes))
        }
        _, prediction = torch.max(output, 1)
    return classes[prediction], probabilities


# Streamlit UI
st.title("Football Player Prediction")
st.write("Upload an image of a football player to get predictions.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    original_image = Image.open(uploaded_file)
    image = original_image.convert("RGB")
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cropped_image = crop_photo(image)

    if cropped_image is None:
        st.error("No clear face detected in the image. Please try a different image.")
    else:
        prediction, probabilities = predict(cropped_image)
        st.image(original_image, caption="Uploaded Image", use_column_width=True)
        st.write(f"**Prediction:** {prediction}")
        st.write("**Probabilities:**")
        st.write(probabilities)
