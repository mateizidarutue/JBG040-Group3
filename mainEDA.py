# Imports
import numpy as np
import matplotlib.pyplot as plt
import torch
from EDA.graphs import *
from dc1.image_dataset import ImageDataset
from torchvision import transforms
from pathlib import Path


# Data loading
data1 = np.load('dc1/data/X_train.npy')
data2 = np.load('dc1/data/Y_train.npy')

# Data shape visualization.
print('shape of data X_train:')
print(data1.shape)
print('shape of data Y_train:')
print(data2.shape)

# Y-label data count visualization.
bar_chart_ylabel(data2).show()

# Coustome index image viewing with label
show_images_around_index(data1, data2, 4000).show()



transform_augment = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(80),  
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
])
    
    # Create an augmented dataset using the cropping transform.
aug_dataset = ImageDataset(
    Path("dc1/data/X_train.npy"),
    Path("dc1/data/Y_train.npy"),
    transform=transform_augment
)
    
    # Choose an index to visualize the effect of cropping.
idx = 4000
label_original = data2[idx]
    
   

num_samples = 5  # Number of augmented samples to generate for the same image
    
plt.figure(figsize=(15, 3))
    
for i in range(num_samples):
    aug_image_tensor, label_aug = aug_dataset[idx]
        
        # Extract the single-channel image from the tensor
    aug_image = aug_image_tensor[0].numpy()
        
    plt.subplot(1, num_samples, i + 1)
    plt.imshow(aug_image, cmap='gray')
    plt.title(f"Aug #{i+1}\nLabel: {label_aug}")
    plt.axis('off')
    
plt.tight_layout()
plt.show()