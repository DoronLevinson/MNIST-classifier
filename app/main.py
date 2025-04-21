# app/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import torch

from inference import download_model_from_s3, load_model, preprocess_image

app = FastAPI()

# Load model on startup
BUCKET = "doron-general-ml-bucket"
OBJECT_KEY = "simple_mnist_model.pth"
LOCAL_MODEL_PATH = "app/model/weights/simple_mnist_model.pth"

download_model_from_s3(BUCKET, OBJECT_KEY, LOCAL_MODEL_PATH)
model = load_model(LOCAL_MODEL_PATH)

@app.get("/")
def root():
    return {"message": "MNIST FastAPI is running."}

@app.post("/predict")
async def predict_digit(file: UploadFile = File(...)):
    try:
        # Read image as bytes
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("L")

        # Preprocess and predict
        tensor = preprocess_image(image)
        with torch.no_grad():
            output = model(tensor)
            prediction = torch.argmax(output, dim=1).item()

        return JSONResponse(content={"prediction": prediction})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))