import os
import sys
import time
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import tensorflow as tf
import matplotlib.pyplot as plt

from scipy.ndimage.interpolation import rotate

def _convert(im):
  return ((im + 1) * 127.5).astype(np.uint8)

def show(im):
  plt.axis('off')
  plt.imshow(_convert(im), interpolation="nearest")
  plt.show()

def _transform_vector(width, x_shift, y_shift, im_scale, rot_in_degrees):
  """
   If one row of transforms is [a0, a1, a2, b0, b1, b2, c0, c1], 
   then it maps the output point (x, y) to a transformed input point 
   (x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k), 
   where k = c0 x + c1 y + 1. 
   The transforms are inverted compared to the transform mapping input points to output points.
  """

  rot = float(rot_in_degrees) / 90. * (math.pi/2)
  
  # Standard rotation matrix
  # (use negative rot because tf.contrib.image.transform will do the inverse)
  rot_matrix = np.array(
      [[math.cos(-rot), -math.sin(-rot)],
      [math.sin(-rot), math.cos(-rot)]]
  )
  
  # Scale it
  # (use inverse scale because tf.contrib.image.transform will do the inverse)
  inv_scale = 1. / im_scale 
  xform_matrix = rot_matrix * inv_scale
  a0, a1 = xform_matrix[0]
  b0, b1 = xform_matrix[1]
  
  # At this point, the image will have been rotated around the top left corner,
  # rather than around the center of the image. 
  #
  # To fix this, we will see where the center of the image got sent by our transform,
  # and then undo that as part of the translation we apply.
  x_origin = float(width) / 2
  y_origin = float(width) / 2
  
  x_origin_shifted, y_origin_shifted = np.matmul(
      xform_matrix,
      np.array([x_origin, y_origin]),
  )

  x_origin_delta = x_origin - x_origin_shifted
  y_origin_delta = y_origin - y_origin_shifted
  
  # Combine our desired shifts with the rotation-induced undesirable shift
  a2 = x_origin_delta - (x_shift/(2*im_scale))
  b2 = y_origin_delta - (y_shift/(2*im_scale))
    
  # Return these values in the order that tf.contrib.image.transform expects
  return np.array([a0, a1, a2, b0, b1, b2, 0, 0]).astype(np.float32)

def test_random_transform(min_scale=0.5, max_scale=1.0,  max_rotation=22.5):
  """
  Scales the image between min_scale and max_scale
  """
  img_shape = [100,100,3]
  img = np.ones(img_shape)
  
  sess = tf.Session()
  image_in = tf.placeholder(dtype=tf.float32, shape=img_shape)
  width = img_shape[0]
  
  def _random_transformation():
    im_scale = np.random.uniform(low=min_scale, high=1.0)
    
    padding_after_scaling = (1-im_scale) * width
    x_delta = np.random.uniform(-padding_after_scaling, padding_after_scaling)
    y_delta = np.random.uniform(-padding_after_scaling, padding_after_scaling)
    
    
    rot = np.random.uniform(-max_rotation, max_rotation)
    
    return _transform_vector(width, 
                                     x_shift=x_delta,
                                     y_shift=y_delta,
                                     im_scale=im_scale, 
                                     rot_in_degrees=rot)

  random_xform_vector = tf.py_func(_random_transformation, [], tf.float32)
  random_xform_vector.set_shape([8])

  output = tf.contrib.image.transform(image_in, random_xform_vector , "BILINEAR")
  
  xformed_img = sess.run(output, feed_dict={
      image_in: img
  })
  
  show(xformed_img)

for i in range(2):
  print("Test image with random transform: %s" % (i+1))
  test_random_transform(min_scale=0.25, max_scale=2.0, max_rotation=22.5)

##############################################################################################


def submatrix(arr):
    x, y = np.nonzero(arr)
    # Using the smallest and largest x and y indices of nonzero elements, 
    # we can find the desired rectangular bounds.  
    # And don't forget to add 1 to the top bound to avoid the fencepost problem.
    return arr[x.min():x.max()+1, y.min():y.max()+1]


class ToSpaceBGR(object):
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
    def __init__(self, is_255):
        self.is_255 = is_255
    def __call__(self, tensor):
        if self.is_255:
            tensor.mul_(255)
        return tensor


def init_patch_circle(image_size, patch_size):
    image_size = image_size**2
    noise_size = int(image_size*patch_size)
    radius = int(math.sqrt(noise_size/math.pi))
    patch = np.zeros((1, 3, radius*2, radius*2))    
    for i in range(3):
        a = np.zeros((radius*2, radius*2))    
        cx, cy = radius, radius # The center of circle 
        y, x = np.ogrid[-radius: radius, -radius: radius]
        index = x**2 + y**2 <= radius**2
        a[cy-radius:cy+radius, cx-radius:cx+radius][index] = np.random.rand()
        idx = np.flatnonzero((a == 0).all((1)))
        a = np.delete(a, idx, axis=0)
        patch[0][i] = np.delete(a, idx, axis=1)
    return patch, patch.shape


def circle_transform(patch, data_shape, patch_shape, image_size):
    # get dummy image 
    x = np.zeros(data_shape)
   
    # get shape
    m_size = patch_shape[-1]
    
    for i in range(x.shape[0]):

        # random rotation
        rot = np.random.choice(360)
        for j in range(patch[i].shape[0]):
            patch[i][j] = rotate(patch[i][j], angle=rot, reshape=False)
        
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
        x[i][0][random_x:random_x+patch_shape[-1], random_y:random_y+patch_shape[-1]] = patch[i][0]
        x[i][1][random_x:random_x+patch_shape[-1], random_y:random_y+patch_shape[-1]] = patch[i][1]
        x[i][2][random_x:random_x+patch_shape[-1], random_y:random_y+patch_shape[-1]] = patch[i][2]
    
    mask = np.copy(x)
    mask[mask != 0] = 1.0
    
    return x, mask, patch.shape


def init_patch_square(image_size, patch_size):
    # get mask
    image_size = image_size**2
    noise_size = image_size*patch_size
    noise_dim = int(noise_size**(0.5))
    patch = np.random.rand(1,3,noise_dim,noise_dim)
    return patch, patch.shape


def square_transform(patch, data_shape, patch_shape, image_size):
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
        x[i][0][random_x:random_x+patch_shape[-1], random_y:random_y+patch_shape[-1]] = patch[i][0]
        x[i][1][random_x:random_x+patch_shape[-1], random_y:random_y+patch_shape[-1]] = patch[i][1]
        x[i][2][random_x:random_x+patch_shape[-1], random_y:random_y+patch_shape[-1]] = patch[i][2]
    
    mask = np.copy(x)
    mask[mask != 0] = 1.0
    
    return x, mask