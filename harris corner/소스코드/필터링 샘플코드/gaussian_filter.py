import cv2
import matplotlib.pyplot as plt
import numpy as np

# Read image
filename = 'cameraman.jpg'
img = plt.imread(filename)


# Get gaussian filter
filter = cv2.getGaussianKernel(ksize=5, sigma=1)


print(filter)
# Make 2D gaussian filter ( Option )
filter = filter * filter.T

print(filter)


# Apply filter to image
dst = cv2.filter2D(img, -1, filter)
cv2.imwrite('gaussian_filter_image.jpg', dst)

# Plot images
plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(dst, cmap='gray'), plt.title('Gaussian_blur')
plt.xticks([]), plt.yticks([])
plt.show()
