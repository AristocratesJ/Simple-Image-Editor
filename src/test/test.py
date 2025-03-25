import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the image using PIL (Python Imaging Library)
img = Image.open('imgs/dog.jpg')
kernel = np.array([[1,  4,  7,  4, 1],
                                 [4, 16, 26, 16, 4],
                                 [7, 26, 41, 26, 7],
                                 [4, 16, 26, 16, 4],
                                 [1,  4,  7,  4, 1]], dtype=np.float32)
print(kernel.sum())
# ## GRAYSCALE
# # Convert the image to a NumPy array
# img_array = np.array(img)
# img_processed1 = np.dot (img_array[..., :3], [0.299, 0.587, 0.114])
# img_processed2 = img_array.mean(axis=2)
# # Display the original RGB image
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 3, 1)
# plt.imshow(img_array)
# plt.title('Original RGB Image')
# plt.axis('off')
#
# # Display the converted grayscale image
# plt.subplot(1, 3, 2)
# plt.imshow(img_processed1, cmap='gray')
# plt.title('Grayscale Image Internet')
# plt.axis('off')
#
# # Display the converted grayscale image
# plt.subplot(1, 3, 3)
# plt.imshow(img_processed2, cmap='gray')
# plt.title('Grayscale Image Mean')
# plt.axis('off')
#
# plt.tight_layout()
# plt.show()



# ## BINARIZE
# # Convert the image to grayscale
# print(np.array(img).shape)
# img_gray = img.convert('L')
# print(np.array(img_gray).shape)
# # Convert the grayscale image to a NumPy array
# img_array = np.array(img_gray)
#
# # Binarize the image using a threshold
# threshold = 128
# binary_img = np.where(img_array < threshold, 0, 255).astype(np.uint8)
#
# # Display the original and binarized images
# plt.figure(figsize= (10, 5))
#
# plt.subplot(1, 2, 1)
# plt.imshow(img_array, cmap='gray')
# plt.title('Original Grayscale Image')
# plt.axis('off')
#
# plt.subplot(1, 2, 2)
# plt.imshow(binary_img, cmap='gray')
# plt.title('Binarized Image (Threshold = 128)')
# plt.axis('off')
#
# plt.tight_layout()
# plt.show()



# ## FILTER
# kernel_dim = (3, 3)
# img = np.array(img)
# new_shape = list(img.shape)
# extend_factor = [np.nan, np.nan]
# extend_factor[0] = kernel_dim[0] - 1
# extend_factor[1] = kernel_dim[1] - 1
# new_shape[0] += 2 * extend_factor[0]
# new_shape[1] += 2 * extend_factor[1]
# print(extend_factor)
# print(img.shape)
# extended_img = np.full(new_shape, 0)
# print(extended_img.shape)
# print(extended_img[extend_factor[0]:(new_shape[0] - extend_factor[0]), extend_factor[1]:(new_shape[1] - extend_factor[1]), :].shape)
# extended_img[extend_factor[0]:(new_shape[0] - extend_factor[0]), extend_factor[1]:(new_shape[1] - extend_factor[1]), :]=img.copy()
# print(img)
# print(extended_img)
#
# # UP
# print(extended_img[:extend_factor[0], extend_factor[1]:(new_shape[1] - extend_factor[1]), :].shape)
# arr = img[0, :, :]
# print(np.repeat(arr[np.newaxis, :, :], extend_factor[0], axis=0).shape)
# extended_img[:extend_factor[0], extend_factor[1]:(new_shape[1] - extend_factor[1]), :] = np.repeat(arr[np.newaxis, :, :], extend_factor[0], axis=0)
# # DOWN
# print(extended_img[(new_shape[0] - extend_factor[0]):, extend_factor[1]:(new_shape[1] - extend_factor[1]), :].shape)
# arr = img[-1, :, :]
# print(np.repeat(arr[np.newaxis, :, :], extend_factor[0], axis=0).shape)
# extended_img[(new_shape[0] - extend_factor[0]):, extend_factor[1]:(new_shape[1] - extend_factor[1]), :] = np.repeat(arr[np.newaxis, :, :], extend_factor[0], axis=0)
# # LEFT
# print(extended_img[extend_factor[0]:(new_shape[0] - extend_factor[0]), :extend_factor[1], :].shape)
# arr = img[:, 0, :]
# extended_img[extend_factor[0]:(new_shape[0] - extend_factor[0]), :extend_factor[1], :] = np.repeat(arr[:, np.newaxis, :], extend_factor[1], axis=1)
# # RIGHT
# print(extended_img[extend_factor[0]:(new_shape[0] - extend_factor[0]), :extend_factor[1], :].shape)
# arr = img[:, -1, :]
# extended_img[extend_factor[0]:(new_shape[0] - extend_factor[0]), (new_shape[1] - extend_factor[1]):, :] = np.repeat(arr[:, np.newaxis, :], extend_factor[1], axis=1)
#
# # UP-LEFT CORNER
# print(extended_img[:extend_factor[0], :extend_factor[1], :].shape)
# extended_img[:extend_factor[0], :extend_factor[1], :] = np.full((extend_factor[0], extend_factor[1], 3), img[0, 0])
# # UP-RIGHT CORNER
# extended_img[:extend_factor[0], (new_shape[1] - extend_factor[1]):, :] = np.full((extend_factor[0], extend_factor[1], 3), img[0, -1])
# # DOWN-LEFT CORNER
# extended_img[(new_shape[0] - extend_factor[0]):, :extend_factor[1], :] = np.full((extend_factor[0], extend_factor[1], 3), img[-1, 0])
# # DOWN-RIGHT CORNER
# extended_img[(new_shape[0] - extend_factor[0]):, (new_shape[1] - extend_factor[1]):, :] = np.full((extend_factor[0], extend_factor[1], 3), img[-1, -1])


kernel = np.full((3, 3), np.nan)
kernel[:, :] = 1/9
print(kernel)
matrix = np.arange(27).reshape([3, 3, 3])
print(matrix)
print("___________")
print((matrix[:, :, 0] * kernel))

print(kernel**(1/2))

def convolution(img, kernel):
    # Define shape for extended image
    new_shape = list(img.shape)
    extend_factor = [np.nan, np.nan]
    extend_factor[0] = kernel.shape[0] - 1
    extend_factor[1] = kernel.shape[1] - 1
    new_shape[0] += 2 * extend_factor[0]
    new_shape[1] += 2 * extend_factor[1]

    # Create extended image
    extended_img = np.full(new_shape, 0, dtype=np.int16)
    # Fill center of and extended image with actual image
    extended_img[extend_factor[0]:(new_shape[0] - extend_factor[0]), extend_factor[1]:(new_shape[1] - extend_factor[1]), :] = img.copy()

    # Fill paddings of extended image
    # UP
    arr = img[0, :, :]
    extended_img[:extend_factor[0], extend_factor[1]:(new_shape[1] - extend_factor[1]), :] = np.repeat(arr[np.newaxis, :, :], extend_factor[0], axis=0)
    # DOWN
    arr = img[-1, :, :]
    extended_img[(new_shape[0] - extend_factor[0]):, extend_factor[1]:(new_shape[1] - extend_factor[1]), :] = np.repeat(arr[np.newaxis, :, :], extend_factor[0], axis=0)
    # LEFT
    arr = img[:, 0, :]
    extended_img[extend_factor[0]:(new_shape[0] - extend_factor[0]), :extend_factor[1], :] = np.repeat(arr[:, np.newaxis, :], extend_factor[1], axis=1)
    # RIGHT
    arr = img[:, -1, :]
    extended_img[extend_factor[0]:(new_shape[0] - extend_factor[0]), (new_shape[1] - extend_factor[1]):, :] = np.repeat(arr[:, np.newaxis, :], extend_factor[1], axis=1)
    # UP-LEFT CORNER
    extended_img[:extend_factor[0], :extend_factor[1], :] = np.full((extend_factor[0], extend_factor[1], 3), img[0, 0])
    # UP-RIGHT CORNER
    extended_img[:extend_factor[0], (new_shape[1] - extend_factor[1]):, :] = np.full((extend_factor[0], extend_factor[1], 3), img[0, -1])
    # DOWN-LEFT CORNER
    extended_img[(new_shape[0] - extend_factor[0]):, :extend_factor[1], :] = np.full((extend_factor[0], extend_factor[1], 3), img[-1, 0])
    # DOWN-RIGHT CORNER
    extended_img[(new_shape[0] - extend_factor[0]):, (new_shape[1] - extend_factor[1]):, :] = np.full((extend_factor[0], extend_factor[1], 3), img[-1, -1])

    # Do the convolution
    result = np.full(img.shape, 0, dtype=np.uint8)
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            tmp = extended_img[i:(i + kernel.shape[0]), j:(j + kernel.shape[1]), :] # fragment of img which is convoluted in this iteration
            r = (tmp[:, :, 0] * kernel).sum()
            g = (tmp[:, :, 1] * kernel).sum()
            b = (tmp[:, :, 2] * kernel).sum()
            pixel = np.clip(np.array([r, g, b]), a_min=0, a_max=255) # clip to range 0 - 255 if it exceeds (0 when number<0 and 255 when number>255
            result[i, j, :] = pixel
    return result


def convolution_gray(img, kernel):
    # Define shape for extended image
    new_shape = list(img.shape)
    pad_top = kernel.shape[0] // 2
    pad_bottom = kernel.shape[0] - 1 - pad_top
    pad_left = kernel.shape[1] // 2
    pad_right = kernel.shape[1] - 1 - pad_left
    new_shape[0] += pad_top + pad_bottom
    new_shape[1] += pad_left + pad_right

    # Create extended image
    extended_img = np.full(new_shape, 0, dtype=img.dtype)
    # Fill center of and extended image with actual image
    extended_img[pad_top:(new_shape[0] - pad_bottom), pad_left:(new_shape[1] - pad_right)] = img.copy()

    # Fill paddings of extended image
    # UP
    arr = img[0, :]
    extended_img[:pad_top, pad_left:(new_shape[1] - pad_right)] = np.repeat(arr[np.newaxis, :], pad_top, axis=0)
    # DOWN
    arr = img[-1, :]
    extended_img[(new_shape[0] - pad_bottom):, pad_left:(new_shape[1] - pad_right)] = np.repeat(arr[np.newaxis, :], pad_bottom, axis=0)
    # LEFT
    arr = img[:, 0]
    extended_img[pad_top:(new_shape[0] - pad_bottom), :pad_left] = np.repeat(arr[:, np.newaxis], pad_left, axis=1)
    # RIGHT
    arr = img[:, -1]
    extended_img[pad_top:(new_shape[0] - pad_bottom), (new_shape[1] - pad_right):] = np.repeat(arr[:, np.newaxis], pad_right, axis=1)
    # UP-LEFT CORNER
    extended_img[:pad_top, :pad_left] = np.full((pad_top, pad_left), img[0, 0])
    # UP-RIGHT CORNER
    extended_img[:pad_top, (new_shape[1] - pad_right):] = np.full((pad_top, pad_right), img[0, -1])
    # DOWN-LEFT CORNER
    extended_img[(new_shape[0] - pad_bottom):, :pad_left] = np.full((pad_bottom, pad_left), img[-1, 0])
    # DOWN-RIGHT CORNER
    extended_img[(new_shape[0] - pad_bottom):, (new_shape[1] - pad_right):] = np.full((pad_bottom, pad_right), img[-1, -1])

    # Do the convolution
    result = np.full(img.shape, 0, dtype=img.dtype)
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            fragment = extended_img[i:(i + kernel.shape[0]),
                       j:(j + kernel.shape[1])]  # fragment of img which is convoluted in this iteration
            pixel = np.sum(fragment * kernel)
            pixel = np.clip(pixel, a_min=0,
                            a_max=255)  # clip to range 0 - 255 if it exceeds (0 when number<0 and 255 when number>255
            result[i, j] = pixel

    print("Done")
    return result

kernel = np.array([[-1, -1, -1, -1],
                   [-1, 11, -1, -1],
                   [-1, -1, -1, -1]])
img_after_filter = convolution_gray(np.array(img).mean(axis=2), kernel)
print(img_after_filter)

plt.figure(figsize= (10, 5))

plt.subplot(1, 2, 1)
plt.imshow(np.array(img))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_after_filter, cmap='gray')
plt.title('After Filter')
plt.axis('off')

plt.tight_layout()
plt.show()