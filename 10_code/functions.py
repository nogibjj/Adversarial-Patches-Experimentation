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
import json

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
    Maps patch values from the range [-infinity, infinity] to CIFAR-10 minimum and maximum values.

    Args:
    - patch (torch.Tensor): The input patch tensor.

    Returns:
    - patch (torch.Tensor): The transformed patch tensor mapped to CIFAR-10 minimum and maximum values.
    """
    # Map patch values from [-infinity, infinity] to ImageNet min and max
    patch = (torch.tanh(patch) + 1 - 2 * TENSOR_MEANS) / (2 * TENSOR_STD)
    return patch


def place_patch(img, patch, apply_rotation = False):
    """
    Places the given patch randomly within the image channels.

    Args:
    - img (numpy.ndarray): The input image as a multi-dimensional NumPy array with shape (channels, height, width).
    - patch (numpy.ndarray): The patch to be placed within the image.

    Returns:
    - img (numpy.ndarray): The modified image with the patch placed randomly within its channels.
    """

    # apply rotation
    if apply_rotation:    
        rot = np.random.choice(4) # apply same rotation to all patches - elisa
        rot = [0,90,180,270][rot] # should preserve the dimensions... - elisa
        patch = torch.from_numpy(rotate(patch.detach().numpy(), rot, axes = (1,2), reshape=False))  # - shifted elisa. 

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


def eval_patch_targeted(model, patch, val_loader, target_class, device):
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


def patch_attack_targeted(
    model,
    device,
    train_loader,
    val_loader,
    target_class,
    patch_size,
    patch_type,
    num_epochs,
):
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

    if patch_type == "circle":
        patch = nn.Parameter(
            torch.tensor(init_patch_circle(patch_size)), requires_grad=True
        )
    elif patch_type == "square":
        patch = nn.Parameter(
            torch.tensor(init_patch_square(patch_size)), requires_grad=True
        )

    # patch = nn.Parameter(
    #     torch.tensor(init_patch_circle(patch_size)), requires_grad=True
    # )
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
                model, patch, val_loader, target_class, device
            )
            print(f"Epoch {epoch}, Attack Success Rate: {attack_success_rate.item()}.")

    # # Final validation
    # attack_success_rate = eval_patch_targeted(
    #     model, patch, val_loader, target_class, device
    # )
    # print(f"Epoch {epoch}, Attack Success Rate: {attack_success_rate.item()}.")

    return patch.data, {
        "acc": acc.item(),
        "top5": top5.item(),
        "attack_success_rate": attack_success_rate.item(),
    }


## UNTARGETED ATTACK


def eval_patch_untargeted(model, patch, val_loader, device):
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


def patch_attack_untargeted(
    model, device, train_loader, val_loader, patch_size, patch_type, num_epochs
):
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

    if patch_type == "circle":
        patch = nn.Parameter(
            torch.tensor(init_patch_circle(patch_size)), requires_grad=True
        )
    elif patch_type == "square":
        patch = nn.Parameter(
            torch.tensor(init_patch_square(patch_size)), requires_grad=True
        )

    # patch = nn.Parameter(
    #     torch.tensor(init_patch_circle(patch_size)), requires_grad=True
    # )
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
            attack_success_rate = eval_patch_untargeted(
                model, patch, val_loader, device
            )
            print(f"Epoch {epoch}, Attack Success Rate: {attack_success_rate.item()}.")

    # Final validation
    attack_success_rate = eval_patch_untargeted(model, patch, val_loader, device)
    print(f"Epoch {epoch}, Attack Success Rate: {attack_success_rate.item()}.")

    return patch.data, attack_success_rate.item()

def generate_patch(patch_size, patch_type, targeted, train_loader, val_loader, target_class = 1, device = 'cuda', num_epochs = 20):
   
    # load the Pre-Trained ResNet-20 model
    resnet = ResNetCIFAR(num_layers=20, Nbits=None)
    resnet = resnet.to(device)
    resnet.load_state_dict(torch.load("./pretrained_models/pretrained_model.pt"))

    if targeted:
        patch, results = patch_attack_targeted(resnet, device, train_loader, val_loader, target_class=target_class, patch_size=patch_size, patch_type=patch_type, num_epochs=num_epochs)
    else:
        patch, results = patch_attack_untargeted(resnet, device, train_loader, val_loader, patch_size=patch_size, patch_type=patch_type, num_epochs=num_epochs)
    
    return patch, results

# plot the ASR as a function of the patch size

def plot_asr(patches, results, val_loader, patch_size, target = None, device = 'cuda', attack_type = 'Targeted', shape = 'circle', model = None, model_name = None):
    # Numbers of pairs of bars you want
    # attack_type should equal to 'Targeted' or 'Untargeted'
    N = 4

    # Data on X-axis

    # Specify the values of blue bars (height)

    validation_ASR = []
    training_ASR = []
    validation_resnet_ASR = []
    three_bars = True
    classes = ('Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck')

    if model is None:
        three_bars = False

    for i in range(len(patches)):
        # load the Pre-Trained ResNet-20 model
        if model is None:
            model = ResNetCIFAR(num_layers=20, Nbits=None)
            model = model.to(device)
            model.load_state_dict(torch.load("./pretrained_models/pretrained_model.pt"))
        if attack_type == 'Targeted':
            _, _, attack_success_rate = eval_patch_targeted(model, patches[i], val_loader, target_class = target, device = device)
        else:
            attack_success_rate = eval_patch_untargeted(model, patches[i], val_loader, device = device)
        validation_ASR.append(attack_success_rate.item())
        if attack_type == 'Targeted':
            training_ASR.append(results[i]['attack_success_rate'])
        else:
            training_ASR.append(results[i])

    # Position of bars on x-axis
    ind = np.arange(N)

    # Figure size
    plt.figure(figsize=(10,6))

    # Width of a bar 
    width = 0.3       

    # Plotting
    if three_bars:
        for i in range(len(patches)):
            resnet_model = ResNetCIFAR(num_layers=20, Nbits=None)
            resnet_model = resnet_model.to(device)
            resnet_model.load_state_dict(torch.load("./pretrained_models/pretrained_model.pt"))
            if attack_type == 'Targeted':
                _, _, attack_success_rate = eval_patch_targeted(resnet_model, patches[i], val_loader, target_class = target, device = device)
            else:
                attack_success_rate = eval_patch_untargeted(resnet_model, patches[i], val_loader, device = device)
            validation_resnet_ASR.append(attack_success_rate.item())
    
    plt.bar(ind, training_ASR , width, label='Training ASR (ResNet20)')
    plt.bar(ind + width, validation_ASR, width, label=f'Validation ASR ({model_name})')
    if three_bars:
        plt.bar(ind + 2*width, validation_resnet_ASR, width, label='Validation ASR (ResNet20)')

    # plot the baseline as a horizontal line
    baseline_acc = baseline(model, val_loader, device)
    plt.axhline(y=baseline_acc, color='r', linestyle='-', label = 'Baseline (No Patch) Incorr. Pred.')

    # add value labels
    if not three_bars:
        for i in range(len(validation_ASR)):
            plt.text(i - 0.1, training_ASR[i] + 0.02, str(round(training_ASR[i], 3)))
            plt.text(i + 0.2, validation_ASR[i] + 0.02, str(round(validation_ASR[i], 3)))
    else:
        for i in range(len(validation_ASR)):
            plt.text(i - 0.1, training_ASR[i] + 0.02, str(round(training_ASR[i], 3)))
            plt.text(i + 0.2, validation_ASR[i] + 0.02, str(round(validation_ASR[i], 3)))
            plt.text(i + 0.5, validation_resnet_ASR[i] + 0.02, str(round(validation_resnet_ASR[i], 3)))

    plt.xlabel('Patch Size')
    plt.ylabel('Attack Success Rate (ASR)')
    if target is None:
        plt.title(f'{attack_type} Attack Success Rate (ASR) As A Function of Patch Size w/ {model_name}')
    else:
        plt.title(f'{attack_type} Attack Success Rate (ASR) For Class {classes[target]} As A Function of Patch Size w/ {model_name}')

    # xticks()
    # First argument - A list of positions at which ticks should be placed
    # Second argument -  A list of labels to place at the given locations
    plt.xticks(ind + width, patch_size)

    # Finding the best position for legends and putting it
    plt.legend(loc='best')
    plt.savefig(f"../20_output_files/asr_{model_name}_{attack_type}_{target}.png")
    plt.show()

def baseline(net, testloader, device = 'cuda'):

    criterion = nn.CrossEntropyLoss()

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    num_val_steps = len(testloader)
    val_acc = correct / total
    print("Test Loss=%.4f, Test accuracy=%.4f" % (test_loss / (num_val_steps), val_acc))
    return 1 - val_acc

def write_to_json(patch, results, patch_size, attack_type, target = None, path = '../20_output_files/model_results/'):
    patch_copy = patch.detach().numpy().tolist()
    if target is None:
        filename = f'{attack_type}_{patch_size}.json'
    else:
        filename = f'{attack_type}_{patch_size}_{target}.json'

    with open(path + filename, 'w') as outfile:
        json.dump({'patch': patch_copy, 'results': results}, outfile)