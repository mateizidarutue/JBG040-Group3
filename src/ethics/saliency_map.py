import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

def generate_saliency_map(model, image, class_index=None):
    model.eval()
    image.requires_grad_()

    output = model(image)

    if class_index is None:
        class_index = torch.argmax(output, dim=1).item()

    model.zero_grad()
    output[0, class_index].backward()

    # Get gradient and convert to saliency
    saliency = image.grad.data.abs().squeeze().cpu().numpy()

    # Handle single channel (grayscale)
    if saliency.ndim == 3:
        saliency = np.max(saliency, axis=0)

    # Normalize saliency map
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

    # Resize saliency to match original image
    saliency_resized = cv2.resize(saliency, (128, 128))

    # Convert to heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * saliency_resized), cv2.COLORMAP_HOT)

    # Prepare original image
    orig_img = image.squeeze().detach().cpu().numpy()
    if orig_img.ndim == 3:
        orig_img = orig_img[0]  # first channel if needed
    orig_img = (orig_img - orig_img.min()) / (orig_img.max() - orig_img.min() + 1e-8)
    orig_img = np.stack([orig_img]*3, axis=-1)

    # Blend heatmap + original image
    overlayed = 0.5 * heatmap / 255 + 0.5 * orig_img

    # Plot it
    plt.figure(figsize=(6, 6))
    plt.imshow(overlayed)
    plt.axis("off")
    plt.title(f"Saliency Map for Class {class_index}")
    plt.show()