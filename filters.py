import numpy as np
from scipy import signal


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

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
    kernel = np.flipud(kernel)
    kernel = np.fliplr(kernel)
    pad_w = Wk // 2
    pad_h = Hk // 2
    padded_img = zero_pad(image, pad_h, pad_w)

    start_i = pad_w
    start_j = pad_h
    finish_i = pad_w + Hi
    finish_j = pad_h + Wi
    for i in range(Hi):
        for j in range(Wi):
            for k in range(Hk):
                for n in range(Wk):
                    out[i, j] += padded_img[i + k, j + n] * kernel[k, n]
    ## END YOUR CODE

    return out


def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """

    H, W = image.shape

    ### YOUR CODE HERE
    out = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), "constant", constant_values=(0,0))
    ### END YOUR CODE
    return out


def conv_fast(image, kernel): # Я попробовала вообще без циклов сделать, через np.einsum.
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
    kernel = np.flipud(kernel)
    kernel = np.fliplr(kernel)
    
    sub_shape = (Hk, Wk)
    view_shape = tuple(np.subtract(padded_img.shape, sub_shape) + 1) + sub_shape
    strides = padded_img.strides + padded_img.strides
    submatrices = np.lib.stride_tricks.as_strided(padded_img, view_shape, strides)

    out = np.einsum('ij,klij->kl', kernel, submatrices)
    ### END YOUR CODE

    return out


def conv_faster(image, kernel): # Незаметно на маленьких kernel, но если kernel становится по размеру сравним с самим изображением, то разница с fast значительна.
    """
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
    out = signal.fftconvolve(image, kernel, mode = 'same')
    ### END YOUR CODE

    return out


# Я создала изображения shelf1, shelf2 и shelf3 (вариации изображения shelf) для удобного тестирования.
# Они в папке img.
def cross_correlation(f, g): # Почему-то все-таки промахивается мимо нужного места. Причем, если ее дополнить нулевым средним, то все становится хорошо.
    """ Cross-correlation of f and g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    ### YOUR CODE HERE
    Hi, Wi = f.shape
    Hk, Wk = g.shape

    Hf = Hi - Hk + 1
    Wf = Wi - Wk + 1
    out = np.zeros((Hf, Wf))

    for i in range(Hf):
        for j in range(Wf):
            sub = f[i:i + Hk, j:j + Wk]

            out[i, j] = np.sum(sub * g)
    ### END YOUR CODE

    return out

# А эта моя изначальная функция, которая похожа на свертку, только без переворачивания kernel.
# Она выдает тот же результат, что и scipy.signal.correlate2d с mode="same", но он какой-то странный, не работает на изображении.
# Я не очень поняла, что все-таки неправильно, но решила сделать с нуля через циклы (это вот то, что выше).
# def cross_correlation(image, kernel):
#     """ An efficient implementation of convolution filter.

#     This function uses element-wise multiplication and np.sum()
#     to efficiently compute weighted sum of neighborhood at each
#     pixel.

#     Hints:
#         - Use the zero_pad function you implemented above
#         - There should be two nested for-loops
#         - You may find np.flip() and np.sum() useful

#     Args:
#         image: numpy array of shape (Hi, Wi).
#         kernel: numpy array of shape (Hk, Wk).

#     Returns:
#         out: numpy array of shape (Hi, Wi).
#     """
#     Hi, Wi = image.shape
#     Hk, Wk = kernel.shape
#     Hf = Hk + Hk - 1
#     Wf = Wi + Wk - 1 
#     out = np.zeros((Hf, Wf))

#     ### YOUR CODE HERE
#     pad_w = (Wk - 1) // 2
#     pad_h = (Hk - 1) // 2
#     padded_img = zero_pad(image, pad_h, pad_w)
#     Hi, Wi = padded_img.shape
    
#     sub_shape = (Hk, Wk)
#     view_shape = tuple(np.subtract(padded_img.shape, sub_shape) + 1) + sub_shape
#     strides = padded_img.strides + padded_img.strides
#     submatrices = np.lib.stride_tricks.as_strided(padded_img, view_shape, strides)

#     out = np.einsum('kl,ijkl->ij', kernel, submatrices)
#     ### END YOUR CODE

#     return out


def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    ### YOUR CODE HERE
    mean1 = np.mean(f)
    mean2 = np.mean(g)

    f_centered = f - mean1
    g_centered = g - mean2

    Hi, Wi = f_centered.shape
    Hk, Wk = g_centered.shape

    Hf = Hi - Hk + 1
    Wf = Wi - Wk + 1
    out = np.zeros((Hf, Wf))

    for i in range(Hf):
        for j in range(Wf):
            sub = f_centered[i:i + Hk, j:j + Wk]

            out[i, j] = np.sum(sub * g_centered)
    ### END YOUR CODE

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    ### YOUR CODE HERE
    Hi, Wi = f.shape
    Hk, Wk = g.shape

    Hf = Hi - Hk + 1
    Wf = Wi - Wk + 1
    out = np.zeros((Hf, Wf))

    mean1 = np.mean(f) # Среднее
    mean2 = np.mean(g)

    std1 = np.std(f) # Стандартное отклонение
    std2 = np.std(g)

    for i in range(Hf):
        for j in range(Wf):
            sub = f[i:i + Hk, j:j + Wk]

            out[i, j] = np.sum((sub - mean1) * (g - mean2))

            out[i, j] /= (std1 * std2) # Нормализация
    ### END YOUR CODE

    return out
