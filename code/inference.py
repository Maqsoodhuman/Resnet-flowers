"""
SageMaker Inference Script for ResNet-34 Flower Classification
"""
import io
import json
import logging
import os

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model import ResNet34

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# ImageNet normalization (used during training)
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]

# Class names (102 flower species)
CLASS_NAMES = [f"{i:03d}" for i in range(1, 103)]


def model_fn(model_dir):
    """Load the PyTorch model from the model directory."""
    logger.info(f"Loading model from {model_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Initialize model
    model = ResNet34(num_classes=102)

    # Load weights
    model_path = os.path.join(model_dir, "resnet34_flowers_best.pth")
    logger.info(f"Loading weights from {model_path}")

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    logger.info("Model loaded successfully")
    return model


def input_fn(request_body, request_content_type):
    """Deserialize and preprocess input data."""
    logger.info(f"Received content type: {request_content_type}")

    if request_content_type in ["image/jpeg", "image/png", "image/jpg", "application/x-image"]:
        # Handle raw image bytes
        image = Image.open(io.BytesIO(request_body)).convert("RGB")
    elif request_content_type == "application/json":
        # Handle JSON with base64 encoded image
        import base64
        data = json.loads(request_body)
        if "image" in data:
            image_bytes = base64.b64decode(data["image"])
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        else:
            raise ValueError("JSON must contain 'image' key with base64 encoded image")
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

    # Preprocessing transform (same as evaluation)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
    ])

    tensor = transform(image).unsqueeze(0)  # Add batch dimension
    logger.info(f"Preprocessed image tensor shape: {tensor.shape}")

    return tensor


def predict_fn(input_data, model):
    """Run inference on the preprocessed input."""
    device = next(model.parameters()).device
    input_data = input_data.to(device)

    logger.info("Running inference...")

    with torch.no_grad():
        outputs = model(input_data)
        probabilities = F.softmax(outputs, dim=1)

    return probabilities


def output_fn(prediction, response_content_type):
    """Serialize the prediction output."""
    logger.info(f"Serializing output with content type: {response_content_type}")

    probs = prediction.cpu().numpy()[0]

    # Get top-5 predictions
    top_k = 5
    top_indices = probs.argsort()[-top_k:][::-1]

    result = {
        "predictions": [
            {
                "class_id": int(idx),
                "class_name": CLASS_NAMES[idx],
                "probability": float(probs[idx])
            }
            for idx in top_indices
        ],
        "predicted_class": CLASS_NAMES[top_indices[0]],
        "confidence": float(probs[top_indices[0]])
    }

    if response_content_type == "application/json":
        return json.dumps(result)
    else:
        raise ValueError(f"Unsupported response content type: {response_content_type}")
