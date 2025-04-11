import numpy as np
import matplotlib.pyplot as plt
import cv2

# Load and preprocess the image
image = cv2.imread('enemy.jpg', cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError("The image file 'logo.jpg' was not found.")

# Convert image to binary using thresholding
_, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Define a 3x3 structuring element
struct_elem = np.ones((3, 3), np.uint8)

# Perform erosion and dilation
eroded_image = cv2.erode(binary_image, struct_elem)
dilated_image = cv2.dilate(binary_image, struct_elem)

# Compute absolute differences (edge detection)
eroded_diff = cv2.absdiff(binary_image, eroded_image)
dilated_diff = cv2.absdiff(dilated_image, binary_image)

# Display results
titles = ['Original Binary', 'Eroded', 'Dilated', 'Erosion Diff', 'Dilation Diff']
images = [binary_image, eroded_image, dilated_image, eroded_diff, dilated_diff]

plt.figure(figsize=(16, 6))
for i, img in enumerate(images):
    plt.subplot(2, 3, i + 1)  # Arranges images in a 2-row, 3-column layout
    plt.imshow(img, cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
