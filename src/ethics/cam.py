import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt


def generate_cam(model, image, class_index, true_class=None):
    model.eval()
    with torch.no_grad():
        feature_maps, _ = model(image)

    # Get weights of  last fully connected layer
    fclay_weights = model.fc_layers[-1].weight[class_index]
    feature_maps = feature_maps.squeeze(0).cpu().numpy()  # Remove batch dim
    fc_weights = fclay_weights.detach().cpu().numpy()

    # Compute weighted sum of feature maps
    cam = np.zeros(feature_maps.shape[1:], dtype=np.float32)
    for i in range(len(fc_weights)):
        cam += fc_weights[i] * feature_maps[i]
    # normalize cam
    cam = np.maximum(cam, 0)
    if np.max(cam) > 0:
        cam = cam / (np.max(cam) + 1e-8)
    else:
        cam = np.zeros_like(cam)
    cam = np.nan_to_num(cam)

    cam = cv2.resize(cam, (128, 128))

    # convert cam to heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    orig_img = image.squeeze(0).squeeze(0).cpu().numpy()
    # normalize the original image and convert to rgb
    orig_img = (orig_img - orig_img.min()) / (orig_img.max() - orig_img.min())
    orig_img = np.stack([orig_img] * 3, axis=-1)

    overlayed_img = 0.5 * heatmap / 255 + 0.5 * orig_img

    # display
    plt.figure(figsize=(6, 6))
    plt.imshow(overlayed_img)
    plt.axis("off")
    plt.title(f"CAM — True: {true_class} | Predicted: {class_index}")
    plt.savefig(
        f"cam_outputs/cam_true_{true_class}_pred_{class_index}.png", bbox_inches="tight"
    )
    plt.show()
