# coding: utf-8
"""
Author: Jay Meng
E-mail: jalymo@126.com
Wechat：345238818
"""

import os
import tensorflow as tf
from six.moves import urllib
import tarfile
from config import cfg

def download_cifar10(data_dir):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    cifar10_url = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'    
    # Check if file exists, otherwise download it
    data_file = os.path.join(data_dir, 'cifar-10-binary.tar.gz')
    if os.path.isfile(data_file):
        pass
    else:
        # Download file
        def progress(block_num, block_size, total_size):
            progress_info = float(block_num * block_size) / float(total_size) * 100.0  
            if progress_info > 100:  
                progress_info = 100
            print('Url=%s, downloading process: %.2f%%' % (cifar10_url, progress_info))            
            
        filepath, _ = urllib.request.urlretrieve(cifar10_url, data_file, progress)
        # Extract file
        tarfile.open(filepath, 'r:gz').extractall(data_dir)
        os.remove(data_file)

# Define CIFAR reader
def read_cifar_files(filename_queue, distort_images = True):
    # Extract model parameters
    image_vec_length = cfg.image_height * cfg.image_width * cfg.num_channels
    record_length = 1 + image_vec_length # ( + 1 for the 0-9 label)
    
    reader = tf.FixedLengthRecordReader(record_bytes=record_length)
    key, record_string = reader.read(filename_queue)
    record_bytes = tf.decode_raw(record_string, tf.uint8)
    image_label = tf.cast(tf.slice(record_bytes, [0], [1]), tf.int32)
  
    # Extract image
    image_extracted = tf.reshape(tf.slice(record_bytes, [1], [image_vec_length]),
                                 [cfg.num_channels, cfg.image_height, cfg.image_width])
    
    # Reshape image
    image_uint8image = tf.transpose(image_extracted, [1, 2, 0])
    reshaped_image = tf.cast(image_uint8image, tf.float32)
    # Randomly Crop image
    final_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, cfg.crop_width, cfg.crop_height)
    
    if distort_images:
        # Randomly flip the image horizontally, change the brightness and contrast
        final_image = tf.image.random_flip_left_right(final_image)
        final_image = tf.image.random_brightness(final_image,max_delta=63)
        final_image = tf.image.random_contrast(final_image,lower=0.2, upper=1.8)

    # Normalize whitening
    final_image = tf.image.per_image_standardization(final_image)
    return(final_image, image_label)

# Create a CIFAR image pipeline from reader
def input_pipeline(data_dir, extract_folder, batch_size, train_logical=True):      
    if train_logical:
        files = [os.path.join(data_dir, extract_folder, 'data_batch_{}.bin'.format(i)) for i in range(1,6)]
    else:
        files = [os.path.join(data_dir, extract_folder, 'test_batch.bin')]
    filename_queue = tf.train.string_input_producer(files)
    image, label = read_cifar_files(filename_queue)
    
    # min_after_dequeue defines how big a buffer we will randomly sample
    #   from -- bigger means better shuffling but slower start up and more
    #   memory used.
    # capacity must be larger than min_after_dequeue and the amount larger
    #   determines the maximum we will prefetch.  Recommendation:
    #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
    min_after_dequeue = 5000
    capacity = min_after_dequeue + 3 * batch_size
    example_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                        batch_size=batch_size,
                                                        capacity=capacity,
                                                        min_after_dequeue=min_after_dequeue)

    return(example_batch, label_batch)


def load_cifar10(data_dir):    
    extract_folder = 'cifar-10-batches-bin'     
    if not os.path.exists(data_dir + '/' + extract_folder):
        download_cifar10(data_dir)    
        
    # Initialize the data pipeline
    images, targets = input_pipeline(data_dir, extract_folder, cfg.batch_size, train_logical=True)
    # Get batch test images and targets from pipline
    test_images, test_targets = input_pipeline(data_dir, extract_folder, cfg.batch_size, train_logical=False)    
    return images, targets, test_images, test_targets
        
