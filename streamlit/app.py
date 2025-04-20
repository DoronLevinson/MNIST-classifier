import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
import numpy as np
import requests
import io

FASTAPI_URL = "http://fastapi:8000/predict" # local FastAPI endpoint

st.title("MNIST Digit Recognizer")

st.sidebar.header("Choose Input Method")
input_mode = st.sidebar.radio("How would you like to provide the digit?", ["Upload Image", "Draw Digit"])

# --- Helper: send image to FastAPI and get prediction ---
def get_prediction(image: Image.Image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    buffered.seek(0)
    files = {"file": ("digit.png", buffered, "image/png")}
    response = requests.post(FASTAPI_URL, files=files)

    if response.status_code == 200:
        return response.json()["prediction"]
    else:
        st.error("Prediction failed.")
        st.error(response.text)
        return None

# --- Upload Mode ---
if input_mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload a handwritten digit image", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("L")
        st.image(image, caption="Uploaded Image", width=150)
        st.write("Image loaded successfully.")

        prediction = get_prediction(image)
        if prediction is not None:
            st.subheader(f"Predicted Digit: {prediction}")

# --- Draw Mode ---
else:
    st.markdown("### Draw a digit below:")
    canvas_result = st_canvas(
        fill_color="white",
        stroke_width=50,
        stroke_color="black",
        background_color="white",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas"
    )

    if canvas_result.image_data is not None:
        image_data = canvas_result.image_data[:, :, 0].astype(np.uint8)
        image = Image.fromarray(image_data)
        image = ImageOps.invert(image.convert("L"))

        st.image(image.resize((150, 150)), caption="Your Drawing")

        prediction = get_prediction(image)
        if prediction is not None:
            st.subheader(f"Predicted Digit: {prediction}")