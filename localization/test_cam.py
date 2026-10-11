from pathlib import Path
import matplotlib.pyplot as plt

from localization.cam import (
    load_model,
    preprocess_image,
    generate_cam,
)

ROOT = Path(__file__).resolve().parent.parent

checkpoint = ROOT / "classification" / "cnn_best.pt"
image_path = ROOT / "data" / "00436515-870c-4b36-a041-de91049b9ab4.dcm"

model, mean, std, device = load_model(checkpoint)

image_tensor = preprocess_image(
    image_path, mean, std, device
)

cam, predicted_class, probabilities = generate_cam(
    model, image_tensor
)

print("Prediction:", "Pneumonia" if predicted_class == 1 else "Normal")
print("Probabilities:", probabilities)
print("CAM shape:", cam.shape)

plt.imshow(cam, cmap="jet")
plt.title("Pneumonia CAM Heatmap")
plt.axis("off")
plt.show()