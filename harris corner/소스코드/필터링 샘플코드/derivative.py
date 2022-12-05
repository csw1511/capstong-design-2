import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
filename = 'cameraman.jpg'
img = plt.imread(filename)


# Make derivative filter
filter_x = np.array([[-1, 0, 1]])
filter_y = np.array([[-1],
                     [0],
                     [1]])


# Apply filter to image
dst = cv2.filter2D(img, -1, filter_y)
cv2.imwrite('derivative_image.jpg', dst)


# Plot images
plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(dst, cmap='gray'), plt.title('derivative')
plt.xticks([]), plt.yticks([])
plt.show()



