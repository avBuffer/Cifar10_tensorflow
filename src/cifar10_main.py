# coding: utf-8
"""
Author: Jay Meng
E-mail: jalymo@126.com
Wechat：345238818
"""

# Introductory CNN Model: MNIST Digits
# In this example, we will load the MNIST handwritten
# digits and create a simple CNN network to predict the
# digit category (0-9)

import os
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import shutil

from config import cfg
from utils import load_cifar10
from cifar10Net import Cifar10Net

from tensorflow.python.framework import ops
ops.reset_default_graph()

# Change Directory
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

    
def main(_):    
    print('BEGIN Cifar10 ...')
    cifar10Net = Cifar10Net()
       
    # Get data
    print('Getting/Transforming Data.')
    images, targets, test_images, test_targets = load_cifar10(cfg.data_dir)
    
    # Declare Model
    print('Creating the CIFAR10 Model.')
    with tf.variable_scope('model_definition') as scope:
        # Declare the training network model
        model_output = cifar10Net.cifar_cnn_model(images, cfg.batch_size)
        # This is very important!!!  We must set the scope to REUSE the variables,
        #  otherwise, when we set the test network model, it will create new random
        #  variables.  Otherwise we get random evaluations on the test batches.
        scope.reuse_variables()
        test_output = cifar10Net.cifar_cnn_model(test_images, cfg.batch_size)
    
    # Declare loss function
    print('Declare Loss Function.')
    loss = cifar10Net.cifar_loss(model_output, targets)
    
    # Create accuracy function
    accuracy = cifar10Net.accuracy_of_batch(test_output, test_targets)
    
    # Create training operations
    print('Creating the Training Operation.')
    generation_num = tf.Variable(0, trainable=False)
    train_op = cifar10Net.train_step(loss, generation_num)
    
    print('Initializing the Variables.')    
    # Initialize Variables
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)        
        # Initialize queue (This queue will feed into the model, so no placeholders necessary)
        tf.train.start_queue_runners(sess=sess)
        
        # Train CIFAR Model
        print('Starting Training')
        train_loss = []
        test_accuracy = []
        
        # Saving accuracy results
        path = cfg.result_dir + '/accuracy.csv'
        if not os.path.exists(cfg.result_dir):
            os.mkdir(cfg.result_dir)
        elif os.path.exists(path):
            os.remove(path)
    
        fd_results = open(path, 'w')
        fd_results.write('Epoch,TrainLoss,TestAcc,CostTime\n')
    
        if not os.path.exists(cfg.model_dir):
            os.mkdir(cfg.model_dir)
        elif os.path.exists(cfg.model_dir):
            shutil.rmtree(cfg.model_dir)
        
        for epoch in range(cfg.epochs):
            startTime = time.time()
            _, loss_value = sess.run([train_op, loss])
            
            if (epoch+1) % cfg.train_sum_freq == 0:
                train_loss.append(loss_value)
                output = 'Epoch {}: Loss = {:.5f}'.format((epoch+1), loss_value)
                print(output)
            
            if (epoch+1) % cfg.test_sum_freq == 0:
                [temp_accuracy] = sess.run([accuracy])
                test_accuracy.append(temp_accuracy)
                acc_output = ' --- Test Accuracy = {:.2f}%.'.format(100.*temp_accuracy)
                print(acc_output)
                 
                endTime = time.time()
                costTime = (endTime - startTime)*1000
                fd_results.write(str(epoch+1) + ',' + str(loss_value) + ',' + 
                                 str(temp_accuracy) + ',' + str(costTime) + '\n')
                fd_results.flush()
        
            if (epoch) % cfg.save_freq == 0:
                saver.save(sess, cfg.model_dir + '/model_epoch_%04d' % (epoch)) 
        
        fd_results.close()            
        tf.logging.info('Training done')
                    
        if cfg.if_showplt:    
            # Print loss and accuracy
            # Matlotlib code to plot the loss and accuracies
            eval_indices = range(0, cfg.epochs, cfg.test_sum_freq)
            output_indices = range(0, cfg.epochs, cfg.train_sum_freq)   
            # Plot loss over time
            plt.plot(output_indices, train_loss, 'k-')
            plt.title('Softmax Loss per Generation')
            plt.xlabel('Generation')
            plt.ylabel('Softmax Loss')
            plt.show()
            
            # Plot accuracy over time
            plt.plot(eval_indices, test_accuracy, 'k-')
            plt.title('Test Accuracy')
            plt.xlabel('Generation')
            plt.ylabel('Accuracy')
            plt.show()
            
    print('END Cifar10')    
    
if __name__ == "__main__":
    tf.app.run()
    