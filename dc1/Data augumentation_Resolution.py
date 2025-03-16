import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the images from file
images = np.load("data/X_train.npy") #change the file name for train and test set
print("Original images shape:", images.shape)

#Change the scale_factor to improve resolution (I tried 2 and 4)
scale_factor = 2

def upscale_image(img, scale=2):
    """
    Upscale a single grayscale image using cubic interpolation.
    """
    # Determine new size: (width, height)
    new_size = (img.shape[1] * scale, img.shape[0] * scale)
    upscaled_img = cv2.resize(img, new_size, interpolation=cv2.INTER_CUBIC)
    return upscaled_img


# Process images depending on their shape:
if len(images.shape) == 4:
    # Assuming shape is (N, 1, H, W)
    # Remove the singleton channel dimension for processing.
    images_2d = images[:, 0, :, :]

    upscaled_list = [upscale_image(img, scale=scale_factor) for img in images_2d]
    # Convert list back to numpy array and reintroduce the channel dimension.
    upscaled_images = np.array(upscaled_list)[:, np.newaxis, :, :]
else:
    # Assuming shape is (N, H, W)
    upscaled_list = [upscale_image(img, scale=scale_factor) for img in images]
    upscaled_images = np.array(upscaled_list)

print("Upscaled images shape:", upscaled_images.shape)

# Save the upscaled images to a new file
np.save('images_upscaled.npy', upscaled_images)
print("Upscaled images saved to 'images_upscaled.npy'.")

upscaled_images = np.load('images_upscaled.npy')

#Correct slicing depending on the shape.
def get_image(arr, idx):
    """Return the idx-th image in 2D grayscale form."""
    if len(arr.shape) == 4:
        # Shape is (N, 1, H, W)
        return arr[idx, 0, :, :]
    else:
        # Shape is (N, H, W)
        return arr[idx, :, :]

# 1) Show a single image (e.g., the first one)
plt.figure(figsize=(6, 6))
plt.imshow(get_image(upscaled_images, 0), cmap='gray')
plt.axis('off')
plt.title('Single Image Example')
plt.show()

# 2) Show multiple images (e.g., first 4 images)
plt.figure(figsize=(12, 4))
for i in range(4):
    plt.subplot(1, 4, i + 1)
    plt.imshow(get_image(upscaled_images, i), cmap='gray')
    plt.axis('off')
    plt.title(f'Image {i+1}')
plt.tight_layout()
plt.show()