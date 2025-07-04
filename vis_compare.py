import os
import cv2
import matplotlib.pyplot as plt

vis_results_dir = r"D:\Writing\vis_results"
methods = os.listdir(vis_results_dir)
methods = [os.path.join(vis_results_dir, method) for method in methods]

images = os.listdir(methods[3])  # Assuming all directories have the same structure and images.

for image_name in images:
    compare_images = [os.path.join(method, image_name) for method in methods]
    plt.figure(figsize=(20, 5))  # Adjust figure size as needed.
    for i, image_path in enumerate(compare_images):
        if not os.path.exists(image_path):  # Check if the file exists to avoid errors.
            print(f"Warning: {image_path} does not exist.")
            continue
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image at {image_path}")
            continue
        # Convert from BGR (OpenCV format) to RGB (Matplotlib format).
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.subplot(1, len(compare_images), i + 1)
        plt.imshow(image_rgb)
        plt.title(os.path.basename(methods[i]))  # Use method name as title for clarity.
        plt.axis('off')  # Hide axis for better visualization.
    plt.tight_layout()
    plt.show()
    plt.close()  # Close the figure after showing it to free up memory.