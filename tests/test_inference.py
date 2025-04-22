import os
from PIL import Image
import torch
from app.inference import load_model, preprocess_image

def test_preprocess_image_shape():
    # Create a dummy 28x28 grayscale image
    image = Image.new("L", (28, 28), color=255)
    tensor = preprocess_image(image)
    
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 784)

def test_load_model_weights():
    model_path = "app/model/weights/simple_mnist_model.pth"
    assert os.path.exists(model_path), "Model file not found"
    
    model = load_model(model_path)
    assert model is not None
    assert hasattr(model, "forward")