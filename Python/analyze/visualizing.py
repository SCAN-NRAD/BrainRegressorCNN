import numpy as np
import logging
import os
import glob
import imageio
import re
import tensorflow as tf


logger = logging.getLogger()


def get_grid_size(kernel_shape) -> tuple:
    size = int(np.ceil(np.sqrt(kernel_shape[len(kernel_shape)-1])))
    space = 1
    patch_h = kernel_shape[0]
    patch_w = kernel_shape[1]

    w = size * patch_w + (size - 1) * space
    h = size * patch_h + (size - 1) * space

    return h, w


def make_grid(kernels: np.ndarray) -> np.ndarray:
    """
    Arrange kernel weights on a grid for visualization
    :param kernels: N kernel (weights) as [N, h, w, 1]
    :return: 2D ndarray (gray-scale image)
    """
    size = int(np.ceil(np.sqrt(kernels.shape[0])))
    space = 1
    patch_h = kernels.shape[1]
    patch_w = kernels.shape[2]

    w = size * patch_w + (size-1) * space
    h = size * patch_h + (size - 1) * space

    grid = np.zeros([h, w])
    idx = 0
    for row in range(size):
        y = row * (patch_h + space)
        for col in range(size):
            x = col * (patch_w + space)
            if idx < kernels.shape[0]:
                patch = kernels[idx, :, :, 0]

                # normalize patch
                patch = (patch - np.max(patch)) / -np.ptp(patch)

                grid[y:y + patch_h, x:x + patch_w] = patch
                idx = idx + 1

    return grid


def make_kernel_gif(checkpoint_path: str, kernel_tensor_names: list):
    # find events file
    candidates = glob.glob(os.path.join(checkpoint_path, 'tb_logs', 'events.out.tfevents.*'))

    if len(candidates) == 0:
        logger.warning('No event file found')
        return

    candidates.sort()
    image_str = tf.placeholder(tf.string)
    img_tf = tf.image.decode_image(image_str)

    for kernel in kernel_tensor_names:
        m = re.match('NET/(layer\d+_\w+)/(kernel|bias):0', kernel)
        tag = m.group(1)
        count = 0
        gif_name = '{}.gif'.format(tag)
        with imageio.get_writer(os.path.join(checkpoint_path, gif_name), mode='I') as gif_writer:
            for event_file in candidates:
                try:
                    for e in tf.train.summary_iterator(event_file):
                        for v in e.summary.value:
                            if str.startswith(v.tag, '{}/image'.format(tag)):
                                count = count + 1
                                img = img_tf.eval({image_str: v.image.encoded_image_string})
                                gif_writer.append_data(img)
                except Exception as ex:
                    logging.error('Error reading {}: {}'.format(event_file, ex))

        logging.info('Wrote %s (%i images)', gif_name, count)

