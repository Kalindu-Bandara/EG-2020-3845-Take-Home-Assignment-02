
# RegNo: EG/2020/3845
# Take Home Assignment 2 - Computer Vision
# Q1

import numpy as np
import matplotlib.pyplot as plt
import os

# Ensure 'Results' folder exists
os.makedirs('Results', exist_ok=True)

# Step 1: Create a grayscale image with larger objects and 3 pixel values
def create_image(height=300, width=300):
    image = np.ones((height, width), dtype=np.uint8) * 60 

    #Circle(pixel value 130)
    rr, cc = np.ogrid[:height, :width]
    mask_circle = (rr - 220)**2 + (cc - 70)**2 <= 60**2 
    image[mask_circle] = 130

    #Square (pixel value 220)
    square_top, square_left = 30, 170
    image[square_top:square_top+80, square_left:square_left+80] = 220 

    return image

# Step 2: Add Gaussian noise
def add_gaussian_noise(image, std=20):
    noise = np.random.normal(0, std, image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

# Step 3: Manual Otsu's thresholding
def manual_otsu_threshold(image):
    hist, _ = np.histogram(image.ravel(), bins=256, range=(0, 256))
    total_pixels = image.size
    max_var = 0
    best_thresh = 0

    for t in range(1, 256):
        w0 = np.sum(hist[:t])
        w1 = np.sum(hist[t:])
        if w0 == 0 or w1 == 0:
            continue
        mu0 = np.sum(np.arange(t) * hist[:t]) / w0
        mu1 = np.sum(np.arange(t, 256) * hist[t:]) / w1
        between_var = w0 * w1 * (mu0 - mu1) ** 2

        if between_var > max_var:
            max_var = between_var
            best_thresh = t

    return best_thresh

# Step 4: Apply steps
original_image = create_image()
noisy_image = add_gaussian_noise(original_image)
otsu_thresh = manual_otsu_threshold(noisy_image)
binary_image = (noisy_image > otsu_thresh).astype(np.uint8) * 255

# Step 5: Visualization
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(original_image, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(noisy_image, cmap='gray')
axes[1].set_title('Image with Gaussian Noise')
axes[1].axis('off')

axes[2].imshow(binary_image, cmap='gray')
axes[2].set_title(f"Otsu Threshold Result (t={otsu_thresh})")
axes[2].axis('off')

plt.tight_layout()
plt.savefig("Results/task1.png", dpi=300)
plt.show()

print("Task completed")
print("Output saved at: Results/task1.png")
