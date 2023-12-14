import os
import sys
import time
import math
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import tensorflow as tf
import matplotlib.pyplot as plt
import torch.nn.functional as F

from scipy.ndimage.interpolation import rotate

#######################################################


# def _convert(im):
#     return ((im + 1) * 127.5).astype(np.uint8)


# def show(im):
#     plt.axis("off")
#     plt.imshow(_convert(im), interpolation="nearest")
#     plt.show()


# def _transform_vector(width, x_shift, y_shift, im_scale, rot_in_degrees):
#     """
#     If one row of transforms is [a0, a1, a2, b0, b1, b2, c0, c1],
#     then it maps the output point (x, y) to a transformed input point
#     (x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k),
#     where k = c0 x + c1 y + 1.
#     The transforms are inverted compared to the transform mapping input points to output points.
#     """

#     rot = float(rot_in_degrees) / 90.0 * (math.pi / 2)

#     # Standard rotation matrix
#     # (use negative rot because tf.contrib.image.transform will do the inverse)
#     rot_matrix = np.array(
#         [[math.cos(-rot), -math.sin(-rot)], [math.sin(-rot), math.cos(-rot)]]
#     )

#     # Scale it
#     # (use inverse scale because tf.contrib.image.transform will do the inverse)
#     inv_scale = 1.0 / im_scale
#     xform_matrix = rot_matrix * inv_scale
#     a0, a1 = xform_matrix[0]
#     b0, b1 = xform_matrix[1]

#     # At this point, the image will have been rotated around the top left corner,
#     # rather than around the center of the image.
#     #
#     # To fix this, we will see where the center of the image got sent by our transform,
#     # and then undo that as part of the translation we apply.
#     x_origin = float(width) / 2
#     y_origin = float(width) / 2

#     x_origin_shifted, y_origin_shifted = np.matmul(
#         xform_matrix,
#         np.array([x_origin, y_origin]),
#     )

#     x_origin_delta = x_origin - x_origin_shifted
#     y_origin_delta = y_origin - y_origin_shifted

#     # Combine our desired shifts with the rotation-induced undesirable shift
#     a2 = x_origin_delta - (x_shift / (2 * im_scale))
#     b2 = y_origin_delta - (y_shift / (2 * im_scale))

#     # Return these values in the order that tf.contrib.image.transform expects
#     return np.array([a0, a1, a2, b0, b1, b2, 0, 0]).astype(np.float32)


# def test_random_transform(min_scale=0.5, max_scale=1.0, max_rotation=22.5):
#     """
#     Scales the image between min_scale and max_scale
#     """
#     img_shape = [100, 100, 3]
#     img = np.ones(img_shape)

#     sess = tf.Session()
#     image_in = tf.placeholder(dtype=tf.float32, shape=img_shape)
#     width = img_shape[0]

#     def _random_transformation():
#         im_scale = np.random.uniform(low=min_scale, high=1.0)

#         padding_after_scaling = (1 - im_scale) * width
#         x_delta = np.random.uniform(-padding_after_scaling, padding_after_scaling)
#         y_delta = np.random.uniform(-padding_after_scaling, padding_after_scaling)

#         rot = np.random.uniform(-max_rotation, max_rotation)

#         return _transform_vector(
#             width,
#             x_shift=x_delta,
#             y_shift=y_delta,
#             im_scale=im_scale,
#             rot_in_degrees=rot,
#         )

#     random_xform_vector = tf.py_func(_random_transformation, [], tf.float32)
#     random_xform_vector.set_shape([8])

#     output = tf.contrib.image.transform(image_in, random_xform_vector, "BILINEAR")

#     xformed_img = sess.run(output, feed_dict={image_in: img})

#     show(xformed_img)


# def _circle_mask(shape, sharpness=40):
#     """Return a circular mask of a given shape"""
#     assert shape[0] == shape[1], "circle_mask received a bad shape: " + shape

#     diameter = shape[0]
#     x = np.linspace(-1, 1, diameter)
#     y = np.linspace(-1, 1, diameter)
#     xx, yy = np.meshgrid(x, y, sparse=True)
#     z = (xx**2 + yy**2) ** sharpness

#     mask = 1 - np.clip(z, -1, 1)
#     mask = np.expand_dims(mask, axis=2)
#     mask = np.broadcast_to(mask, shape).astype(np.float32)
#     return mask


# def _gen_target_ys():
#     label = TARGET_LABEL
#     y_one_hot = np.zeros(1000)
#     y_one_hot[label] = 1.0
#     y_one_hot = np.tile(y_one_hot, (BATCH_SIZE, 1))
#     return y_one_hot


##############################################################################################


def submatrix(arr):
    """This code defines a function called submatrix that takes a 2D array (arr) as input 
    and returns a submatrix containing only the non-zero elements from the input array."""
    # find the indices of non-zero elements in the input array arr
    x, y = np.nonzero(arr) # returns two arrays: x contains the row indices of non-zero elements, and y contains the column indices of non-zero elements.
    # Using the smallest and largest x and y indices of nonzero elements,
    # we can find the desired rectangular bounds.
    # And don't forget to add 1 to the top bound to avoid the fencepost problem.
    # identifies the minimum and maximum row and column indices of the non-zero elements in the input array and returns a submatrix containing only those elements.
    return arr[x.min() : x.max() + 1, y.min() : y.max() + 1] 
    

class ToSpaceBGR(object):
    """This class appears to be a transformation class that, when called, swaps color channels in an input tensor.
      If is_bgr is True, it swaps the red and blue channels in the tensor."""
    def __init__(self, is_bgr):
        self.is_bgr = is_bgr

    def __call__(self, tensor):
        if self.is_bgr:
            new_tensor = tensor.clone()
            new_tensor[0] = tensor[2]
            new_tensor[2] = tensor[0]
            tensor = new_tensor
        return tensor


class ToRange255(object):
    """This class, when called, scales the values of an input tensor by 255 if is_255 is True."""
    def __init__(self, is_255):
        self.is_255 = is_255

    def __call__(self, tensor):
        if self.is_255:
            tensor.mul_(255)
        return tensor


def init_patch_circle(image_size, patch_size):
    """This function generates a circular patch within a square matrix (represented as a NumPy array) filled with random values. 
    It creates three layers (channels) of the patch, representing an RGB image. 
    It generates a circular pattern using random values within the circle and then deletes some rows and columns with zero values."""
    #squares the image_size
    ##? why squaring image_size? - might have been to adjust the scale or emphasize some aspect in the noise generation process??
    image_size = image_size**2
    #Calculates the noise_size as a proportion of the squared image_size and patch_size - overall size of the noise relative to the image size and the desired size of the patch. 
    ##? why is noise calculated like this? - It seems to indicate how much of the squared image's area should be used to create the noise patch.
    noise_size = int(image_size * patch_size)
    # Determines the radius of the circle based on the calculated noise_size and the formula for the area of a circle (pi * radius^2). 
    # It computes the radius by reversing the area formula (radius = sqrt(area / pi)). Noise Size is the area of the circle.
    radius = int(math.sqrt(noise_size / math.pi))
    # Initializes a NumPy array patch filled with zeros. It has a shape of (1, 3, radius * 2, radius * 2), creating a 4-dimensional array. 
    # The dimensions correspond to (1) batch size, (2) number of channels, and (3-4) the square matrix for the circle (radius * 2= vertical/horizontal diameters).
    patch = np.zeros((1, 3, radius * 2, radius * 2))
    # Loop that runs three tiems
    for i in range(3):
        # Initializes a 2D NumPy array a filled with zeros, representing the square matrix for the circle.
        a = np.zeros((radius * 2, radius * 2))
        cx, cy = radius, radius  # Coordinates at the center of circle
        # Generates two meshgrid arrays x and y spanning from -radius to radius. 
        # These grids represent the x and y coordinates around the center of the circle.
        y, x = np.ogrid[-radius:radius, -radius:radius]
        # Creates a boolean mask (index) by checking if the squared distances from the center (x**2 + y**2) are less than or equal to the squared radius. 
        # This identifies the points within the circle. 
        # This means it will always include points up to the edge of the square matrix, effectively capturing the entire circle within that square area.
        index = x**2 + y**2 <= radius**2
        # Assigns random values to the elements within the circle (identified by index) in the a array.
        a[cy - radius : cy + radius, cx - radius : cx + radius][
            index
        ] = np.random.rand()
        ##? why are we doing this: ensure that the resulting circle shape within the square matrix is clean, providing a more accurate representation of the intended circular patch.   
        ##? Will this reduce the size of the patch??     
        # Finds indices where all elements along the rows are zeros within array a.
        idx = np.flatnonzero((a == 0).all((1)))
        # Deletes rows from array a where all elements are zeros, based on the found indices (idx).
        a = np.delete(a, idx, axis=0)
        # Deletes columns from array a where all elements are zeros, based on the found indices (idx),
        # and assigns the modified a to a specific slice in the patch array.
        ##? not sure why it is using ix again for columns - need to use this maybe before idx_columns = np.flatnonzero((a == 0).all(axis=0))
        patch[0][i] = np.delete(a, idx, axis=1)
    return patch, patch.shape


def circle_transform(patch, data_shape, patch_shape, image_size, device="cuda"):
    """This function takes a patch generated previously, rotates it randomly, 
    places it randomly within a dummy image, and creates a mask based on the applied patch."""
    # create dummy image filled with zeros based on the specified data_shape
    x = np.zeros(data_shape)

    # Determines the size of the patch by extracting the last dimension from the patch_shape.
    m_size = patch_shape[-1]

    # Initiates a loop iterating through the first dimension of the dummy image x.
    for i in range(x.shape[0]):
        # Generates a random rotation angle (rot) between 0 and 359 degrees for each iteration.
        rot = np.random.choice(360)
        # Rotates the patch randomly using the previously generated rot angle for the specified channel (patch[0][i]) within the patch.
        ##? What does the patch look like - try print(patch[0][i].shape[0])
        for j in range(patch[0][i].shape[0]):
            patch[0][i] = rotate(patch[0][i], angle=rot, reshape=False)

        # random location
        # Generates random x and y coordinates within the image size, ensuring that the patch, when placed at these coordinates,
        #  fits within the dummy image x without exceeding its boundaries.
        random_x = np.random.choice(image_size)
        if random_x + m_size > x.shape[-1]:
            #If the coordinates exceed the boundary, the code regenerates new random coordinates until they fit within the image boundaries.
            while random_x + m_size > x.shape[-1]:
                random_x = np.random.choice(image_size)
        random_y = np.random.choice(image_size)
        if random_y + m_size > x.shape[-1]:
            while random_y + m_size > x.shape[-1]:
                random_y = np.random.choice(image_size)

        # apply patch to dummy image - question: am I applying this correctly?
        # Integrates the rotated patch into the dummy image x at the random x and y coordinates, overlaying the patch onto the corresponding area within the image.
        # x[i][
        #     random_x : random_x + patch_shape[-1], random_y : random_y + patch_shape[-1]
        # ] = patch[0][i]
        # apply patch to dummy image  - keep this because it assumes that the channels of the image and the patch might not always be the same
        x[i][0][random_x:random_x+patch_shape[-1], random_y:random_y+patch_shape[-1]] = patch[i][0]
        x[i][1][random_x:random_x+patch_shape[-1], random_y:random_y+patch_shape[-1]] = patch[i][1]
        x[i][2][random_x:random_x+patch_shape[-1], random_y:random_y+patch_shape[-1]] = patch[i][2]

    # Creates a mask (mask) from the dummy image x by copying its values and setting non-zero elements to 1.0, 
    # effectively creating a binary mask representing the patch area in the image.
    # why? process of generating a secondary image or array that highlights or isolates specific areas or elements of interest within the original image or data.
    mask = np.copy(x)
    mask[mask != 0] = 1.0

    # x = torch.tensor(x).to(device)
    # mask = torch.tensor(mask).to(device)
    # patch = torch.tensor(patch).to(device)

    return x, mask, patch.shape


def init_patch_square(image_size, patch_size):
    """This function generates a square patch (also as a NumPy array) filled with random values similar to the circular patch function. 
    It generates three layers representing an RGB image."""
    # get mask
    image_size = image_size**2
    noise_size = image_size * patch_size
    noise_dim = int(noise_size ** (0.5)) # dimensions (height and width) of the square patch array 
    patch = np.random.rand(1, 3, noise_dim, noise_dim)
    return patch, patch.shape


def square_transform(patch, data_shape, patch_shape, image_size):
    """This function takes a square patch, performs rotations, places it randomly within a dummy image, and creates a mask."""
    # get dummy image
    x = np.zeros(data_shape)

    # get shape
    m_size = patch_shape[-1]

    for i in range(x.shape[0]):
        # random rotation
        rot = np.random.choice(4)
        for j in range(patch[i].shape[0]):
            patch[i][j] = np.rot90(patch[i][j], rot)

        # random location
        random_x = np.random.choice(image_size)
        if random_x + m_size > x.shape[-1]:
            while random_x + m_size > x.shape[-1]:
                random_x = np.random.choice(image_size)
        random_y = np.random.choice(image_size)
        if random_y + m_size > x.shape[-1]:
            while random_y + m_size > x.shape[-1]:
                random_y = np.random.choice(image_size)

        # apply patch to dummy image
        x[i][0][
            random_x : random_x + patch_shape[-1], random_y : random_y + patch_shape[-1]
        ] = patch[i][0]
        x[i][1][
            random_x : random_x + patch_shape[-1], random_y : random_y + patch_shape[-1]
        ] = patch[i][1]
        x[i][2][
            random_x : random_x + patch_shape[-1], random_y : random_y + patch_shape[-1]
        ] = patch[i][2]

    mask = np.copy(x)
    mask[mask != 0] = 1.0

    return x, mask


def attack(
    x,
    patch,
    mask,
    netClassifier,
    target, # (target label for misclassification)
    conf_target=0.9, # desired confidence level for the adversarial attack. The attack loop runs until the confidence score of the target class exceeds this threshold.
    # min_out and max_out are limits within which the pixel values of the adversarial image will be clamped
    min_out=-1,  # not sure about this one
    max_out=1,  # not sure about this one
    max_count=1000, # maximum number of iterations allowed for the adversarial attack loop
):
    """This function performs an adversarial attack on an input image x. 
    It calculates the probability of the target class in the network's output and 
    generates an adversarial example by adding a patch to the original image while considering a mask. 
    It iterates to modify the image until the confidence of the target class reaches a 
    specified threshold (conf_target) or the maximum count of iterations is reached."""

    #sets the neural network classifier in evaluation mode
    netClassifier.eval()
    # Computes the probabilities of classes for the original input image x using the softmax function through the neural network classifier netClassifier. 
    # target_prob stores the probability of the target class (target) predicted by the classifier for the input image x.
    x_out = F.softmax(netClassifier(x))
    target_prob = x_out.data[0][target]  # QUESTION: why are we only taking one?

    # extend mask to match dimensions of x
    mask = mask.unsqueeze(0).expand(x.shape[0], -1, -1, -1)

    # extend patch to match dimensions of input image x
    # unsqueeze adds an extra dimension, and expand replicates the mask and patch across multiple samples 
    patch = patch.unsqueeze(0).expand(x.shape[0], -1, -1, -1)

    # Generates an adversarial example (adv_x) by blending the original input x with the patch (patch) using the mask. 
    # This operation modifies x by adding the patch where the mask is non-zero.
    adv_x = torch.mul((1 - mask), x) + torch.mul(mask, patch)

    # Loop until the confidence of the target class exceeds a specified threshold or until a maximum count is reached.
    count = 0
    while conf_target > target_prob:
        count += 1
        # it calculates the network's predictions and gradients to update the adversarial image in each iteration.
        adv_x = adv_x.detach().requires_grad_(True)  # CHANGED
        adv_out = F.log_softmax(netClassifier(adv_x))

        adv_out_probs, adv_out_labels = adv_out.max(1)
        
        # Computes the loss, computes gradients, and updates the patch and adversarial image based on the gradients and 
        # specified output bounds (min_out and max_out). Clamps the adversarial image to maintain values within a certain range.
        Loss = -adv_out[0][target]
        Loss.backward()
        adv_grad = adv_x.grad.clone()
        patch = patch.clone()

        adv_x.grad.data.zero_()

        # subtracts the gradient values from the patch matrix element-wise.
        # moving in the direction that reduces the loss (or increases the probability of misclassification) 
        # by reducing the neural network's confidence in predicting the correct class
        patch -= adv_grad

        adv_x = torch.mul((1 - mask), x) + torch.mul(mask, patch)
        adv_x = torch.clamp(adv_x, min_out, max_out)

        #Re-evaluates the modified adversarial image and updates the target class probability for further comparison in the loop.
        out = F.softmax(netClassifier(adv_x))
        target_prob = out.data[0][target]
        # y_argmax_prob = out.data.max(1)[0][0]

        # print(count, conf_target, target_prob, y_argmax_prob)
        #Breaks the loop if the maximum count of iterations is reached
        if count >= max_count:
            break

    return adv_x, mask, patch


def train(
    epoch,
    patch,
    patch_shape,
    netClassifier,
    train_loader,
    target,
    device="cuda",
    patch_type="square",
    image_size=32,
):
    netClassifier.eval()
    success = 0
    total = 0
    recover_time = 0
    # Iterates through the training dataset (train_loader) by batches.
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        prediction = netClassifier(data)
        # only computer adversarial examples on examples that are originally classified correctly
        # Checks if the current prediction is incorrect. If so, it continues to the next batch. If correct, it increments the total count.
        if prediction.data.max(1)[1][0] != labels.data[0]:
            continue

        total += 1

        # Retrieves the shape of the data.
        # transform path
        data_shape = data[0].data.cpu().numpy().shape

        # Depending on the patch type specified (patch_type), transforms the patch and mask accordingly using either a circular or square transformation.
        if patch_type == "circle":
            patch, mask, patch_shape = circle_transform(
                patch, data_shape, patch_shape, image_size
            )
        elif patch_type == "square":
            patch, mask = square_transform(patch, data_shape, patch_shape, image_size)

        # Converts patch and mask to PyTorch tensors and moves them to the specified device (GPU or CPU).
        patch, mask = torch.FloatTensor(patch), torch.FloatTensor(mask)
        patch, mask, data = patch.to(device), mask.to(device), data.to(device)

        # patch, mask = Variable(patch), Variable(mask) - NOT SURE IF THIS SHOULD BE UNCOMMENTED

        # Generates adversarial examples (adv_x) by attacking the data using the patch.
        adv_x, mask, patch = attack(data, patch, mask, netClassifier, target)

        adv_label = netClassifier(adv_x).data.max(1)[1][0] #adv_label
        ori_label = labels.data[0] # original label
        # target is the misclassified class we want

        # Checks if the adversarial example is successful in achieving the target misclassification. 
        # If successful, increments the success count.
        if adv_label == target: 
            success += 1

            # if plot_all == 1: 
            #     # plot source image
            #     vutils.save_image(data.data, "./%s/%d_%d_original.png" %(opt.outf, batch_idx, ori_label), normalize=True)
                
            #     # plot adversarial image
            #     vutils.save_image(adv_x.data, "./%s/%d_%d_adversarial.png" %(opt.outf, batch_idx, adv_label), normalize=True)

        masked_patch = torch.mul(mask, patch)
        patch = masked_patch.data.cpu().numpy()
        new_patch = np.zeros(patch_shape)
        for i in range(new_patch.shape[0]):
            for j in range(new_patch.shape[1]):
                new_patch[i][j] = submatrix(patch[i][j])

        patch = new_patch

        # log to file
    print("Epoch {}: Train Patch Success: {:.3f}".format(epoch, success / total))

    return patch


def test(
    epoch,
    patch,
    patch_shape,
    netClassifier,
    test_loader,
    image_size,
    min_out,
    max_out,
    target,
    device="cuda",
    patch_type="square",
):
    netClassifier.eval()
    success = 0
    total = 0
    for batch_idx, (data, labels) in enumerate(test_loader):
        data, labels = data.to(device), labels.to(device)
        prediction = netClassifier(data)

        # only computer adversarial examples on examples that are originally classified correctly
        if prediction.data.max(1)[1][0] != labels.data[0]:
            continue

        total += 1

        # transform path
        data_shape = data[0].data.cpu().numpy().shape

        if patch_type == "circle":
            patch, mask, patch_shape = circle_transform(
                patch, data_shape, patch_shape, image_size
            )
        elif patch_type == "square":
            patch, mask = square_transform(patch, data_shape, patch_shape, image_size)
        patch, mask = torch.FloatTensor(patch), torch.FloatTensor(mask)
        patch, mask, data = patch.to(device), mask.to(device), data.to(device)

        # patch, mask = Variable(patch), Variable(mask) - NOT SURE IF THIS SHOULD BE UNCOMMENTED

        # extend mask to match dimensions of x
        mask = mask.unsqueeze(0).expand(data.shape[0], -1, -1, -1)

        # extend patch to match dimensions of x
        patch = patch.unsqueeze(0).expand(data.shape[0], -1, -1, -1)

        adv_x = torch.mul((1 - mask), data) + torch.mul(mask, patch)
        adv_x = torch.clamp(adv_x, min_out, max_out)

        adv_label = netClassifier(adv_x).data.max(1)[1][0]
        ori_label = labels.data[0]

        if adv_label == target:
            success += 1

        masked_patch = torch.mul(mask, patch)
        patch_copy = masked_patch.data.cpu().numpy()
        new_patch = np.zeros(patch_shape)
        for i in range(new_patch.shape[0]):
            for j in range(new_patch.shape[1]):
                new_patch[i][j] = submatrix(patch_copy[i][j])

        # patch = new_patch

        # log to file
    print("Epoch {}: Test Patch Success: {:.3f}".format(epoch, success / total))
