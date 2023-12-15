# Import Required Packages
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

import os
import sys
import time
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable

from scipy.ndimage import rotate
from pretrained_models.resnet20 import ResNetCIFAR
from tqdm import tqdm

#######################################################

## CIRCLE PATCH


def init_patch_circle(patch_size):
    """
    Generates a circular patch filled with random values within the specified patch size.

    Args:
    - patch_size (int): The diameter of the circular patch. It determines the size of the patch.

    Returns:
    - patch (numpy.ndarray): A 3-dimensional NumPy array representing the circular patch
                            with random values inside the circular boundary.
    """
    radius = patch_size // 2
    patch = np.zeros((3, radius * 2, radius * 2))
    for i in range(3):
        a = np.zeros((radius * 2, radius * 2))
        cx, cy = radius, radius  # The center of circle
        y, x = np.ogrid[-radius:radius, -radius:radius]
        index = x**2 + y**2 <= radius**2
        a[cy - radius : cy + radius, cx - radius : cx + radius][
            index
        ] = np.random.rand()
        idx = np.flatnonzero((a == 0).all((1)))
        a = np.delete(a, idx, axis=0)
        patch[i] = np.delete(a, idx, axis=1)
    return patch


## SQUARE PATCH


def init_patch_square(patch_size):
    """
    Generates a square patch filled with random values of the specified patch size.

    Args:
    - patch_size (int): The size of the square patch.

    Returns:
    - patch (numpy.ndarray): A 3-dimensional NumPy array representing the square patch
                            filled with random values.
    """
    patch = np.zeros(
        (3, patch_size, patch_size)
    )  # Initialize the square patch with zeros
    # Fill the square patch with random values
    for i in range(3):  # Iterate over RGB channels
        patch[i] = np.random.rand(patch_size, patch_size)
    return patch


## PATCH TRANSFORMATIONS

# CIFAR-10 image tensor mean and std
NORM_MEAN = [0.4914, 0.4822, 0.4465]
NORM_STD = [0.2023, 0.1994, 0.2010]
# TENSOR_MEANS (torch.Tensor): Tensor representing the means of the ImageNet dataset.
# TENSOR_STD (torch.Tensor): Tensor representing the standard deviations of the ImageNet dataset.
TENSOR_MEANS, TENSOR_STD = (
    torch.FloatTensor(NORM_MEAN)[:, None, None],
    torch.FloatTensor(NORM_STD)[:, None, None],
)


def patch_forward(patch):
    """
    Maps patch values from the range [-infinity, infinity] to ImageNet minimum and maximum values.

    Args:
    - patch (torch.Tensor): The input patch tensor.

    Returns:
    - patch (torch.Tensor): The transformed patch tensor mapped to ImageNet minimum and maximum values.
    """
    # Map patch values from [-infinity, infinity] to ImageNet min and max
    patch = (torch.tanh(patch) + 1 - 2 * TENSOR_MEANS) / (2 * TENSOR_STD)
    return patch


def place_patch(img, patch):
    """
    Places the given patch randomly within the image channels.

    Args:
    - img (numpy.ndarray): The input image as a multi-dimensional NumPy array with shape (channels, height, width).
    - patch (numpy.ndarray): The patch to be placed within the image.

    Returns:
    - img (numpy.ndarray): The modified image with the patch placed randomly within its channels.
    """
    for i in range(img.shape[0]):  # Loop through channels
        # Generate random offsets within the image boundaries for placing the patch
        h_offset = np.random.randint(0, img.shape[2] - patch.shape[1] - 1)
        w_offset = np.random.randint(0, img.shape[3] - patch.shape[2] - 1)
        # Place the patch using patch_forward() function
        img[
            i,
            :,
            h_offset : h_offset + patch.shape[1],
            w_offset : w_offset + patch.shape[2],
        ] = patch_forward(patch)
    return img


## TARGETED ATTACK


def eval_patch_targeted(model, patch, val_loader, target_class):
    """
    Evaluates the performance of the given patch on the model using a validation loader.

    Args:
    - model (torch.nn.Module): The neural network model.
    - patch (numpy.ndarray): The patch to be evaluated on the model.
    - val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
    - target_class (int): The target class index for the attack.

    Returns:
    - acc (torch.Tensor): Accuracy of the patch in fooling the model on non-target class images.
    - top5 (torch.Tensor): Top-5 accuracy of the patch in fooling the model on non-target class images.
    - attack_success_rate (torch.Tensor): Attack success rate of the patch on non-target class images.
    """
    model.eval()
    tp, tp_5, counter = 0.0, 0.0, 0.0
    n = 0  # number of images
    with torch.no_grad():
        for img, img_labels in tqdm(val_loader, desc="Validating...", leave=False):
            # For stability, place the patch at 4 random locations per image, and average the performance
            for _ in range(4):
                patch_img = place_patch(img, patch)
                patch_img = patch_img.to(device)
                img_labels = img_labels.to(device)
                pred = model(patch_img)
                # In the accuracy calculation, we need to exclude the images that are of our target class
                # as we would not "fool" the model into predicting those
                tp += torch.logical_and(
                    pred.argmax(dim=-1) == target_class, img_labels != target_class
                ).sum()
                tp_5 += torch.logical_and(
                    (pred.topk(5, dim=-1)[1] == target_class).any(dim=-1),
                    img_labels != target_class,
                ).sum()
                counter += (img_labels != target_class).sum()
                n += (img_labels != target_class).sum()
    acc = tp / counter
    top5 = tp_5 / counter
    attack_success_rate = tp / n
    return acc, top5, attack_success_rate


def patch_attack_targeted(model, target_class, patch_size, num_epochs=5):
    """
    Performs a targeted patch attack on the given model to deceive it into misclassifying target class images.

    Args:
    - model (torch.nn.Module): The neural network model.
    - target_class (int): The index of the target class for the attack.
    - patch_size (int): The size of the patch.
    - num_epochs (int): The number of epochs for training the patch (default is 5).

    Returns:
    - patch (torch.Tensor): The generated patch tensor.
    - metrics (dict): Dictionary containing evaluation metrics such as accuracy, top-5 accuracy, and attack success rate.
    """

    train_set, val_set = torch.utils.data.random_split(trainset, [0.8, 0.2])
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=32, shuffle=False, drop_last=False, num_workers=4
    )

    patch = nn.Parameter(
        torch.tensor(init_patch_circle(patch_size)), requires_grad=True
    )
    optimizer = torch.optim.SGD([patch], lr=1e-1, momentum=0.8)
    loss_module = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(num_epochs):
        t = tqdm(train_loader, leave=False)
        for img, _ in t:
            img = place_patch(img, patch)
            img = img.to(device)
            pred = model(img)
            labels = torch.zeros(
                img.shape[0], device=pred.device, dtype=torch.long
            ).fill_(target_class)
            loss = loss_module(pred, labels)
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()
            t.set_description(f"Epoch {epoch}, Loss: {loss.item():4.2f}")
        if epoch % 5 == 0:
            acc, top5, attack_success_rate = eval_patch_targeted(
                model, patch, val_loader, target_class
            )
            print(f"Epoch {epoch}, Attack Success Rate: {attack_success_rate.item()}.")

    # Final validation
    attack_success_rate = eval_patch_targeted(model, patch, val_loader)
    print(f"Epoch {epoch}, Attack Success Rate: {attack_success_rate.item()}.")

    return patch.data, {
        "acc": acc.item(),
        "top5": top5.item(),
        "attack_success_rate": attack_success_rate.item(),
    }


## UNTARGETED ATTACK


def eval_patch_untargeted(model, patch, val_loader):
    """
    Evaluates the effectiveness of an untargeted patch attack on the model using a validation loader.

    Args:
    - model (torch.nn.Module): The neural network model.
    - patch (numpy.ndarray): The patch to be evaluated on the model.
    - val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.

    Returns:
    - attack_success_rate (torch.Tensor): Attack success rate of the patch on non-target class images.
    """
    model.eval()
    tp = 0
    n = 0  # number of images
    with torch.no_grad():
        for img, img_labels in tqdm(val_loader, desc="Validating...", leave=False):
            # For stability, place the patch at 4 random locations per image, and average the performance
            for _ in range(4):
                patch_img = place_patch(img, patch)
                patch_img = patch_img.to(device)
                img_labels = img_labels.to(device)
                pred = model(patch_img)
                # get number of incorrect predictions
                tp += (pred.argmax(dim=-1) != img_labels).sum()
                n += img_labels.shape[0]

    attack_success_rate = tp / n
    return attack_success_rate


def patch_attack_untargeted(model, patch_size=16, num_epochs=5):
    """
    Performs an untargeted patch attack on the given model to induce misclassification on non-targeted images.

    Args:
    - model (torch.nn.Module): The neural network model.
    - patch_size (int): The size of the patch (default is 16).
    - num_epochs (int): The number of epochs for training the patch (default is 5).

    Returns:
    - patch (torch.Tensor): The generated patch tensor.
    - attack_success_rate (float): The final attack success rate of the patch on non-target class images.
    """

    train_set, val_set = torch.utils.data.random_split(trainset, [0.8, 0.2])
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=32, shuffle=False, drop_last=False, num_workers=4
    )

    patch = nn.Parameter(
        torch.tensor(init_patch_circle(patch_size)), requires_grad=True
    )
    optimizer = torch.optim.SGD([patch], lr=1e-1, momentum=0.8)
    loss_module = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(num_epochs):
        t = tqdm(train_loader, leave=False)
        for img, labels in t:
            img = place_patch(img, patch)
            img = img.to(device)
            labels = labels.to(device)
            pred = model(img)
            loss = -loss_module(pred, labels)  # make it negative to maximize the loss
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()
            t.set_description(f"Epoch {epoch}, Loss: {loss.item():4.2f}")
        if epoch % 5 == 0:
            attack_success_rate = eval_patch_untargeted(model, patch, val_loader)
            print(f"Epoch {epoch}, Attack Success Rate: {attack_success_rate.item()}.")

    # Final validation
    attack_success_rate = eval_patch_untargeted(model, patch, val_loader)
    print(f"Epoch {epoch}, Attack Success Rate: {attack_success_rate.item()}.")

    return patch.data, attack_success_rate.item()