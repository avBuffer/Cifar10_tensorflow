# coding: utf-8
"""
Author: Jay Meng
E-mail: jalymo@126.com
Wechat：345238818
"""

import tensorflow as tf
from config import cfg


class Cifar10Net(object):
    def __init__(self):   
        tf.logging.info('Seting up the main structure')

    # Define the model architecture, this will return logits from images
    def cifar_cnn_model(self, input_images, batch_size, train_logical=True):
        def truncated_normal_var(name, shape, dtype):
            return(tf.get_variable(name=name, shape=shape, dtype=dtype, initializer=tf.truncated_normal_initializer(stddev=0.05)))
        def zero_var(name, shape, dtype):
            return(tf.get_variable(name=name, shape=shape, dtype=dtype, initializer=tf.constant_initializer(0.0)))
        
        # First Convolutional Layer
        with tf.variable_scope('conv1') as scope:
            # Conv_kernel is 5x5 for all 3 colors and we will create 64 features
            conv1_kernel = truncated_normal_var(name='conv_kernel1', shape=[5, 5, 3, 64], dtype=tf.float32)
            # We convolve across the image with a stride size of 1
            conv1 = tf.nn.conv2d(input_images, conv1_kernel, [1, 1, 1, 1], padding='SAME')
            # Initialize and add the bias term
            conv1_bias = zero_var(name='conv_bias1', shape=[64], dtype=tf.float32)
            conv1_add_bias = tf.nn.bias_add(conv1, conv1_bias)
            # ReLU element wise
            relu_conv1 = tf.nn.relu(conv1_add_bias)
        
        # Max Pooling
        pool1 = tf.nn.max_pool(relu_conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],padding='SAME', name='pool_layer1')
        
        # Local Response Normalization (parameters from paper)
        # paper: http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks
        norm1 = tf.nn.lrn(pool1, depth_radius=5, bias=2.0, alpha=1e-3, beta=0.75, name='norm1')
    
        # Second Convolutional Layer
        with tf.variable_scope('conv2') as scope:
            # Conv kernel is 5x5, across all prior 64 features and we create 64 more features
            conv2_kernel = truncated_normal_var(name='conv_kernel2', shape=[5, 5, 64, 64], dtype=tf.float32)
            # Convolve filter across prior output with stride size of 1
            conv2 = tf.nn.conv2d(norm1, conv2_kernel, [1, 1, 1, 1], padding='SAME')
            # Initialize and add the bias
            conv2_bias = zero_var(name='conv_bias2', shape=[64], dtype=tf.float32)
            conv2_add_bias = tf.nn.bias_add(conv2, conv2_bias)
            # ReLU element wise
            relu_conv2 = tf.nn.relu(conv2_add_bias)
        
        # Max Pooling
        pool2 = tf.nn.max_pool(relu_conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool_layer2')    
        
         # Local Response Normalization (parameters from paper)
        norm2 = tf.nn.lrn(pool2, depth_radius=5, bias=2.0, alpha=1e-3, beta=0.75, name='norm2')
        
        # Reshape output into a single matrix for multiplication for the fully connected layers
        reshaped_output = tf.reshape(norm2, [batch_size, -1])
        reshaped_dim = reshaped_output.get_shape()[1].value
        
        # First Fully Connected Layer
        with tf.variable_scope('full1') as scope:
            # Fully connected layer will have 384 outputs.
            full_weight1 = truncated_normal_var(name='full_mult1', shape=[reshaped_dim, 384], dtype=tf.float32)
            full_bias1 = zero_var(name='full_bias1', shape=[384], dtype=tf.float32)
            full_layer1 = tf.nn.relu(tf.add(tf.matmul(reshaped_output, full_weight1), full_bias1))
    
        # Second Fully Connected Layer
        with tf.variable_scope('full2') as scope:
            # Second fully connected layer has 192 outputs.
            full_weight2 = truncated_normal_var(name='full_mult2', shape=[384, 192], dtype=tf.float32)
            full_bias2 = zero_var(name='full_bias2', shape=[192], dtype=tf.float32)
            full_layer2 = tf.nn.relu(tf.add(tf.matmul(full_layer1, full_weight2), full_bias2))
    
        # Final Fully Connected Layer -> 10 categories for output (num_targets)
        with tf.variable_scope('full3') as scope:
            # Final fully connected layer has 10 (num_targets) outputs.
            full_weight3 = truncated_normal_var(name='full_mult3', shape=[192, cfg.num_targets], dtype=tf.float32)
            full_bias3 =  zero_var(name='full_bias3', shape=[cfg.num_targets], dtype=tf.float32)
            final_output = tf.add(tf.matmul(full_layer2, full_weight3), full_bias3)
            
        return(final_output)


    # Loss function
    def cifar_loss(self, logits, targets):
        # Get rid of extra dimensions and cast targets into integers
        targets = tf.squeeze(tf.cast(targets, tf.int32))
        # Calculate cross entropy from logits and targets
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets)
        # Take the average loss across batch size
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        return(cross_entropy_mean)
    
    
    # Train step
    def train_step(self, loss_value, generation_num):
        # Our learning rate is an exponential decay after we wait a fair number of generations
        model_learning_rate = tf.train.exponential_decay(cfg.learning_rate, generation_num,
                                                         cfg.num_gens_to_wait, cfg.lr_decay, staircase=True)
        # Create optimizer
        my_optimizer = tf.train.GradientDescentOptimizer(model_learning_rate)
        # Initialize train step
        train_step = my_optimizer.minimize(loss_value)
        return(train_step)
    
    
    # Accuracy function
    def accuracy_of_batch(self, logits, targets):
        # Make sure targets are integers and drop extra dimensions
        targets = tf.squeeze(tf.cast(targets, tf.int32))
        # Get predicted values by finding which logit is the greatest
        batch_predictions = tf.cast(tf.argmax(logits, 1), tf.int32)
        # Check if they are equal across the batch
        predicted_correctly = tf.equal(batch_predictions, targets)
        # Average the 1's and 0's (True's and False's) across the batch size
        accuracy = tf.reduce_mean(tf.cast(predicted_correctly, tf.float32))
        return(accuracy)
    
    