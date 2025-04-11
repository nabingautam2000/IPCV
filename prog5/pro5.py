import cv2
import numpy as np
import matplotlib.pyplot as plt
 
image = cv2.imread('spider.jpg', cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError ("image not found")

_, binary = cv2.threshold (image, 127, 255, cv2.THRESH_BINARY)

kernel = np.ones((3, 3), np.uint8)


eroded = cv2.erode(binary, kernel)
dilated = cv2.dilate(binary, kernel)

edges_from_erosion = cv2.absdiff (binary, eroded)
edges_from_dilatin = cv2.absdiff (dilated ,binary)

titles = ['Binary', 'Eroded','Dilated' ,'Edge (errision) ', 'Edge (Dilation)']
images = [binary ,eroded, dilated, edges_from_erosion, edges_from_dilatin]

plt.figure (figsize=(12, 6))
for i ,(title ,img) in enumerate (zip (titles, images)):
    plt.subplot(2 ,3 ,i+1)
    plt.imshow (img, cmap='grey')
    plt.title = (title)
    plt.axis = ('off')
plt.tight_layout()
plt.show()


 

