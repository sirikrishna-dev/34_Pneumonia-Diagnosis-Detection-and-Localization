"""
CAM generation for Pneumonia Diagnosis and Localization.

Uses the trained CAMNet model from the classification folder.
Supports DICOM chest X-rays and ordinary image files.
"""

import sys
from pathlib import Path

import numpy as np
import pydicom
import torch
import matplotlib.pyplot as plt
from PIL import Image

# Locate the project and classification code.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "classification"))

from classification.model import CAMNet


def load_model(checkpoint_path):
    """Load the trained model and normalization statistics."""

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Loading trained model...")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    model = CAMNet(
        in_channels=checkpoint.get("in_channels", 1)
    ).to(device)

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    mean = np.asarray(
        checkpoint["mean"], dtype=np.float32
    )

    std = np.asarray(
        checkpoint["std"], dtype=np.float32
    )

    print("Model loaded successfully!")

    return model, mean, std, device


def load_grayscale_image(image_path):
    """
    Load a DICOM or ordinary image and resize it to 128x128.

    Matches the image scaling and resizing in
    classification/preprocess.py.
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    if image_path.suffix.lower() == ".dcm":
        dcm = pydicom.dcmread(str(image_path))

        image = dcm.pixel_array.astype(np.float32)

        # Scale each image to 0-255, matching training preprocessing.
        image = (
            (image - image.min())
            / (image.max() - image.min() + 1e-8)
            * 255.0
        )

        image = Image.fromarray(image.astype(np.uint8))

    else:
        image = Image.open(image_path).convert("L")

    image = image.resize(
        (128, 128),
        Image.Resampling.BILINEAR
    )

    return np.asarray(image, dtype=np.uint8)


def preprocess_image(image_path, mean, std, device):
    """Load and normalize an X-ray for the trained model."""

    image = load_grayscale_image(image_path)

    image = image.astype(np.float32)

    # Use the per-pixel statistics saved during training.
    image = (image - mean) / std

    tensor = torch.from_numpy(image)

    # Shape: batch x channel x height x width.
    tensor = tensor.unsqueeze(0).unsqueeze(0)

    return tensor.to(device)


@torch.no_grad()
def generate_cam(model, image_tensor):
    """
    Generate a Class Activation Map for the pneumonia class.

    Returns:
        cam: Normalized 128x128 heatmap.
        predicted_class: 0 for Normal, 1 for Pneumonia.
        probabilities: Predicted class probabilities.
    """

    # Extract the last convolutional feature maps.
    features = model.features(image_tensor)

    # Calculate classification logits.
    logits = model.fc(
        model.gap(features).flatten(1)
    )

    probabilities = torch.softmax(logits, dim=1)

    predicted_class = int(
        logits.argmax(dim=1).item()
    )

    # Class 1 represents pneumonia.
    pneumonia_weights = model.fc.weight[1]

    # Combine feature maps using the pneumonia-class weights.
    cam = torch.einsum(
        "c,bchw->bhw",
        pneumonia_weights,
        features
    )

    # Keep positive activations.
    cam = torch.relu(cam)[0]

    # Normalize the heatmap to [0, 1].
    cam = (
        (cam - cam.min())
        / (cam.max() - cam.min() + 1e-8)
    )

    # Resize the CAM to the original model input dimensions.
    cam = torch.nn.functional.interpolate(
        cam[None, None],
        size=(128, 128),
        mode="bilinear",
        align_corners=False
    )[0, 0]

    return (
        cam.cpu().numpy(),
        predicted_class,
        probabilities[0].cpu().numpy()
    )


def show_cam(image_path, cam, predicted_class, probabilities):
    """Display the X-ray alongside its pneumonia CAM."""

    image = load_grayscale_image(image_path)

    fig, axes = plt.subplots(
        1, 2, figsize=(12, 5)
    )

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original Chest X-ray")
    axes[0].axis("off")

    axes[1].imshow(image, cmap="gray")
    axes[1].imshow(
        cam,
        cmap="jet",
        alpha=0.45,
        vmin=0,
        vmax=1
    )

    axes[1].set_title("Pneumonia CAM Heatmap")
    axes[1].axis("off")

    label = (
        "Pneumonia"
        if predicted_class == 1
        else "Normal"
    )

    fig.suptitle(
        f"Prediction: {label} | "
        f"P(Pneumonia)={probabilities[1]:.3f}"
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    checkpoint_path = (
        PROJECT_ROOT / "classification" / "cnn_best.pt"
    )

    image_path = (
        PROJECT_ROOT
        / "data"
        / "00436515-870c-4b36-a041-de91049b9ab4.dcm"
    )

    # Load the trained classification model.
    model, mean, std, device = load_model(
        checkpoint_path
    )

    print("Device:", device)
    print("Image:", image_path)

    # Preprocess the DICOM image.
    image_tensor = preprocess_image(
        image_path,
        mean,
        std,
        device
    )

    # Generate the pneumonia CAM.
    cam, predicted_class, probabilities = generate_cam(
        model,
        image_tensor
    )

    label = (
        "Pneumonia"
        if predicted_class == 1
        else "Normal"
    )

    print("\nPrediction:", label)
    print(
        "P(Normal):",
        round(float(probabilities[0]), 4)
    )
    print(
        "P(Pneumonia):",
        round(float(probabilities[1]), 4)
    )
    print("CAM shape:", cam.shape)
    print(
        "CAM value range:",
        float(cam.min()),
        "to",
        float(cam.max())
    )

    # Display the X-ray and heatmap.
    show_cam(
        image_path,
        cam,
        predicted_class,
        probabilities
    )