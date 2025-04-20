# app/inference.py
import torch
from torchvision import transforms
from PIL import Image
import os
import boto3
from model.architectures.simple_nn import SimpleNN

def download_model_from_s3(bucket_name, object_key, filename_local):
    if os.path.exists(filename_local):
        return

    # ✅ Ensure the directory exists
    os.makedirs(os.path.dirname(filename_local), exist_ok=True)

    # ✅ Download the file
    s3 = boto3.client("s3")
    s3.download_file(bucket_name, object_key, filename_local)

def load_model(filename_local):
    model = SimpleNN()
    model.load_state_dict(torch.load(filename_local, map_location=torch.device("cpu")))
    model.eval()
    return model

def preprocess_image(image: Image.Image) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    tensor = transform(image).unsqueeze(0)  # [1, 1, 28, 28]
    tensor = tensor.view(-1, 28 * 28)        # Flatten to [1, 784]
    return tensor