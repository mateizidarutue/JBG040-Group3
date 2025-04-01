import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

def generate_gradcam(model, image, class_index=None, true_class=None):
    model.eval()

    # --- Step 1: Get the target conv layer ---
    target_layer = model.conv_layers[-1]  # You can change this to any layer

    activations = []
    gradients = []

    # Hook to capture activations (forward pass)
    def forward_hook(module, input, output):
        activations.append(output)

    # Hook to capture gradients (backward pass)
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    # Register hooks
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_backward_hook(backward_hook)

    image = image.requires_grad_()
    output = model(image)

    if class_index is None:
        class_index = torch.argmax(output, dim=1).item()

    score = output[0, class_index]
    model.zero_grad()
    score.backward(retain_graph=True)

    # Detach hooks
    forward_handle.remove()
    backward_handle.remove()

    # --- Step 2: Get feature maps & gradients ---
    act = activations[0].squeeze(0).cpu().detach().numpy()  # shape: [C, H, W]
    grad = gradients[0].squeeze(0).cpu().detach().numpy()   # shape: [C, H, W]

    # Grad-CAM++ weighting
    weights = np.zeros(grad.shape[0], dtype=np.float32)

    for k in range(grad.shape[0]):
        grad_k = grad[k]
        act_k = act[k]

        alpha_num = grad_k ** 2
        alpha_denom = 2 * grad_k ** 2 + act_k * grad_k ** 3
        alpha_denom = np.where(alpha_denom != 0, alpha_denom, 1e-8)
        alpha = alpha_num / alpha_denom
        relu_grad = np.maximum(grad_k, 0)
        weights[k] = np.sum(alpha * relu_grad)

    # --- Step 3: Compute CAM ---
    cam = np.zeros_like(act[0], dtype=np.float32)
    for k, w in enumerate(weights):
        cam += w * act[k]

    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)
    cam = cv2.resize(cam, (128, 128))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    # --- Step 4: Overlay on original image ---
    orig_img = image.squeeze().detach().cpu().numpy()
    if orig_img.ndim == 3:
        orig_img = orig_img[0]
    orig_img = (orig_img - orig_img.min()) / (orig_img.max() - orig_img.min() + 1e-8)
    orig_img = np.stack([orig_img]*3, axis=-1)

    overlayed = 0.5 * heatmap / 255 + 0.5 * orig_img

    # --- Step 5: Plot ---
    plt.figure(figsize=(6, 6))
    plt.imshow(overlayed)
    plt.axis("off")
    plt.title(f"Grad-CAM++ — True: {true_class} | Predicted: {class_index}")
    plt.show()
