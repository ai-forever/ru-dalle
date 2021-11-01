# -*- coding: utf-8 -*-
import numpy as np


def pad_reflect(image, pad_size):
    imsize = image.shape
    height, width = imsize[:2]
    new_img = np.zeros([height + pad_size * 2, width + pad_size * 2, imsize[2]]).astype(np.uint8)
    new_img[pad_size:-pad_size, pad_size:-pad_size, :] = image
    new_img[0:pad_size, pad_size:-pad_size, :] = np.flip(image[0:pad_size, :, :], axis=0)  # top
    new_img[-pad_size:, pad_size:-pad_size, :] = np.flip(image[-pad_size:, :, :], axis=0)  # bottom
    new_img[:, 0:pad_size, :] = np.flip(new_img[:, pad_size:pad_size * 2, :], axis=1)  # left
    new_img[:, -pad_size:, :] = np.flip(new_img[:, -pad_size * 2:-pad_size, :], axis=1)  # right
    return new_img


def unpad_image(image, pad_size):
    return image[pad_size:-pad_size, pad_size:-pad_size, :]


def pad_patch(image_patch, padding_size, channel_last=True):
    """ Pads image_patch with with padding_size edge values. """
    if channel_last:
        return np.pad(
            image_patch,
            ((padding_size, padding_size), (padding_size, padding_size), (0, 0)),
            'edge',
        )
    else:
        return np.pad(
            image_patch,
            ((0, 0), (padding_size, padding_size), (padding_size, padding_size)),
            'edge',
        )


def unpad_patches(image_patches, padding_size):
    return image_patches[:, padding_size:-padding_size, padding_size:-padding_size, :]


def split_image_into_overlapping_patches(image_array, patch_size, padding_size=2):
    """ Splits the image into partially overlapping patches.
    The patches overlap by padding_size pixels.
    Pads the image twice:
        - first to have a size multiple of the patch size,
        - then to have equal padding at the borders.
    Args:
        image_array: numpy array of the input image.
        patch_size: size of the patches from the original image (without padding).
        padding_size: size of the overlapping area.
    """
    xmax, ymax, _ = image_array.shape
    x_remainder = xmax % patch_size
    y_remainder = ymax % patch_size

    # modulo here is to avoid extending of patch_size instead of 0
    x_extend = (patch_size - x_remainder) % patch_size
    y_extend = (patch_size - y_remainder) % patch_size

    # make sure the image is divisible into regular patches
    extended_image = np.pad(image_array, ((0, x_extend), (0, y_extend), (0, 0)), 'edge')

    # add padding around the image to simplify computations
    padded_image = pad_patch(extended_image, padding_size, channel_last=True)

    xmax, ymax, _ = padded_image.shape
    patches = []

    x_lefts = range(padding_size, xmax - padding_size, patch_size)
    y_tops = range(padding_size, ymax - padding_size, patch_size)

    for x in x_lefts:
        for y in y_tops:
            x_left = x - padding_size
            y_top = y - padding_size
            x_right = x + patch_size + padding_size
            y_bottom = y + patch_size + padding_size
            patch = padded_image[x_left:x_right, y_top:y_bottom, :]
            patches.append(patch)

    return np.array(patches), padded_image.shape


def stich_together(patches, padded_image_shape, target_shape, padding_size=4):
    """ Reconstruct the image from overlapping patches.
    After scaling, shapes and padding should be scaled too.
    Args:
        patches: patches obtained with split_image_into_overlapping_patches
        padded_image_shape: shape of the padded image contructed in split_image_into_overlapping_patches
        target_shape: shape of the final image
        padding_size: size of the overlapping area.
    """

    xmax, ymax, _ = padded_image_shape
    patches = unpad_patches(patches, padding_size)
    patch_size = patches.shape[1]
    n_patches_per_row = ymax // patch_size

    complete_image = np.zeros((xmax, ymax, 3))

    row = -1
    col = 0
    for i in range(len(patches)):
        if i % n_patches_per_row == 0:
            row += 1
            col = 0
        complete_image[
            row * patch_size: (row + 1) * patch_size, col * patch_size: (col + 1) * patch_size, :
        ] = patches[i]
        col += 1
    return complete_image[0: target_shape[0], 0: target_shape[1], :]
