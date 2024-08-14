import numpy as np
import matplotlib.pyplot as plt
from time import time
import cv2
from scipy import signal
from filters import conv_fast, zero_pad


img_grey = np.array([[1, 2, 0, 3],
                    [1, 2, 0, 3],
                    [19, 0, 5, 1],
                    [3, 1, 7, 2]])
temp_grey = np.array([[3, 3, 3],
                     [5, 0, 1],
                     [1, 2, 8]])

img_grey = np.array([
    [1, 2, 3, 1, 2, 3, 1, 2, 3],
    [2, 2, 2, 3, 3, 3, 1, 1, 1],
    [0, 0, 0, 1, 1, 2, 2, 3, 3],
    [0, 2, 1, 3, 4, 1, -2, 1, 1],
    [-1, -2, 0, 0, -2, 1, 1, 1, 1],
    [0, 0, 2, 3, 3, 1, 1, -1, -1],
    [1, 2, 3, 1, 2, 3, 1, 2, 3],
    [-1, -2, -3, 0, -1, -2, 1, 0, -1],
    [0, 1, 2, 0, 0, 1, 2, 2, 0]
])

temp_grey = np.array([
    [0, 2, 1],
    [-1, -2, 0],
    [0, 0, 2]
])

def cross_correlation(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    pad_w = (Wk - 1) // 2
    pad_h = (Hk - 1) // 2
    padded_img = zero_pad(image, pad_h, pad_w)
    Hi, Wi = padded_img.shape
    
    sub_shape = (Hk, Wk)
    view_shape = tuple(np.subtract(padded_img.shape, sub_shape) + 1) + sub_shape
    strides = padded_img.strides + padded_img.strides
    submatrices = np.lib.stride_tricks.as_strided(padded_img, view_shape, strides)

    out = np.einsum('ij,klij->kl', kernel, submatrices)
    ### END YOUR CODE

    return out


# Perform cross-correlation between the image and the template
out = cross_correlation(img_grey, temp_grey)
# out = signal.correlate2d(img_grey, temp_grey)

# Find the location with maximum similarity
y, x = np.unravel_index(out.argmax(), out.shape)

plt.figure(figsize=(25, 20))

# Display product template
plt.subplot(311), plt.imshow(temp_grey), plt.title('Template'), plt.axis('off') 

# Display image
plt.subplot(312), plt.imshow(img_grey), plt.title('Result (blue marker on the detected location)'), plt.axis('off')

# Display cross-correlation output
plt.subplot(313), plt.imshow(out), plt.title('Cross-correlation (white means more correlated)'), plt.axis('off')

# Draw marker at detected location
plt.plot(x, y, 'bx', ms=20, mew=5)

plt.show()
