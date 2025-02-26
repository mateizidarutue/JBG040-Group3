import matplotlib.pyplot as plt
import numpy as np

def bar_chart_ylabel(data):
    unique_values, counts = np.unique(data, return_counts=True)

    plt.figure(figsize=(10, 6))
    plt.bar(unique_values, counts)

    plt.title('Distribution of Classes in data2')
    plt.xlabel('Class Value')
    plt.ylabel('Count')
    plt.xticks([0, 1, 2, 3, 4, 5], ["Atelectasis", "Effusion", "Infiltration", "No finding", "Nodule", "Pneumothorax"])

    plt.grid(True, alpha=0.3)

    return plt

class_labels = {
    0: "Atelectasis",
    1: "Effusion",
    2: "Infiltration",
    3: "No finding",
    4: "Nodule",
    5: "Pneumothorax"
}

def show_images_around_index(data_images, data_labels, center_index):
    if center_index < 0 or center_index >= 16841:
        print(f"Error: Index must be between 0 and {16841-1}")
        return
    
    start_idx = max(0, center_index - 2)
    end_idx = min(16841, center_index + 3)
    indices = list(range(start_idx, end_idx))
    
    plt.figure(figsize=(15, 4))
    
    for i, idx in enumerate(indices):
        plt.subplot(1, 5, i+1)
        plt.imshow(data_images[idx, 0, :, :], cmap='gray')
        plt.axis('off')
        
        # Convert numerical label to class name
        label_name = class_labels.get(data_labels[idx], "Unknown")
        
        if idx == center_index:
            plt.title(f'Index: {idx}\nLabel: {label_name}', fontsize=10, fontweight='bold')
        else:
            plt.title(f'Index: {idx}\nLabel: {label_name}', fontsize=10)
    
    plt.tight_layout()
    return plt