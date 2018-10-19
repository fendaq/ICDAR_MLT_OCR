import tensorflow as tf
import functools

def augment_image(image):
  """Augmentation the image with a random modification.
  Args:
    image: input Tensor image of rank 3, with the last dimension
           of size 3.
  Returns:
    Distorted Tensor image of the same shape.
  """
  with tf.variable_scope('AugmentImage'):
    height = image.get_shape().dims[0].value
    width = image.get_shape().dims[1].value

    # Random crop cut from the street sign image, resized to the same size.
    # Assures that the crop is covers at least 0.8 area of the input image.
    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        bounding_boxes=tf.zeros([0, 0, 4]),
        min_object_covered=0.8,
        aspect_ratio_range=[0.8, 1.2],
        area_range=[0.8, 1.0],
        use_image_if_no_bounding_boxes=True)
    distorted_image = tf.slice(image, bbox_begin, bbox_size)

    # Randomly chooses one of the 4 interpolation methods
    
    distorted_image = tf.clip_by_value(distorted_image, -1.5, 1.5)

  return distorted_image


def central_crop(image, crop_size):
  """Returns a central crop for the specified size of an image.
  Args:
    image: A tensor with shape [height, width, channels]
    crop_size: A tuple (crop_width, crop_height)
  Returns:
    A tensor of shape [crop_height, crop_width, channels].
  """
  with tf.variable_scope('CentralCrop'):
    target_width, target_height = crop_size
    image_height, image_width = tf.shape(image)[0], tf.shape(image)[1]
    assert_op1 = tf.Assert(
        tf.greater_equal(image_height, target_height),
        ['image_height < target_height', image_height, target_height])
    assert_op2 = tf.Assert(
        tf.greater_equal(image_width, target_width),
        ['image_width < target_width', image_width, target_width])
    with tf.control_dependencies([assert_op1, assert_op2]):
      offset_width = (image_width - target_width) / 2
      offset_height = (image_height - target_height) / 2
      return tf.image.crop_to_bounding_box(image, offset_height, offset_width,
                                           target_height, target_width)


def preprocess_image(image, augment=False, central_crop_size=None,
                     num_towers=4):
  """Normalizes image to have values in a narrow range around zero.
  Args:
    image: a [H x W x 3] uint8 tensor.
    augment: optional, if True do random image distortion.
    central_crop_size: A tuple (crop_width, crop_height).
    num_towers: optional, number of shots of the same image in the input image.
  Returns:
    A float32 tensor of shape [H x W x 3] with RGB values in the required
    range.
  """
  with tf.variable_scope('PreprocessImage'):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    if augment or central_crop_size:
      if num_towers == 1:
        images = [image]
      else:
        images = tf.split(value=image, num_or_size_splits=num_towers, axis=1)
      if central_crop_size:
        view_crop_size = (central_crop_size[0] / num_towers,
                          central_crop_size[1])
        images = [central_crop(img, view_crop_size) for img in images]
      if augment:
        images = [augment_image(img) for img in images]
      image = tf.concat(images, 1)

    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.5)

  return image