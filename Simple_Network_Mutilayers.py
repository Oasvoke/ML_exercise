
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


# In[2]:


def initialize_parameter(layers_n):
    parameters = {}
    L = len(layers_n)
    for i in range(1,L):
        parameters["W"+str(i)] = tf.get_variable(name="W"+str(i),shape=(layers_n[i],layers_n[i-1]),initializer=tf.contrib.layers.xavier_initializer())
        parameters["b"+str(i)] = tf.get_variable(name="b"+str(i),shape=(layers_n[i],1),initializer=tf.zeros_initializer())
    return parameters
    
#    n_hidden_1 = 256 # 1st layer number of neurons
#    n_hidden_2 = 256 # 2nd layer number of neurons
#   num_input = 784 # MNIST data input (img shape: 28*28)
#    num_classes = 10 # MNIST total classes (0-9 digits)


# In[3]:



"""
tf.reset_default_graph()
with tf.Session() as sess:
    parameters = initialize_parameter([784,256,256,10])
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"])) 
    print("W3 = " + str(parameters["W3"]))
    print("b3 = " + str(parameters["b3"])) 
"""


# In[4]:


def one_step_forward(X,W,b):
    Z = tf.add(tf.matmul(W,X),b)
#    A = Z
    A = tf.nn.relu(Z)
    return A


# In[5]:


def forward(X,parameters):

    L=len(parameters)//2
    A_prev = X
    for i in range(1,L):
        A = one_step_forward(A_prev , parameters["W"+str(i)] , parameters["b"+str(i)]) 
        A_prev = A
    Z_last = tf.add(tf.matmul(parameters["W"+str(L)],A_prev),parameters["b"+str(L)])
    return Z_last
        


# In[6]:


def com_cost(Z_last,Y):
    logits = tf.transpose(Z_last)
    labels = tf.transpose(Y)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels))
    return cost


# In[7]:


def test_acc(Z3,Y):
    # Evaluate model (with test logits, for dropout to be disabled)
    correct_pred = tf.equal(tf.argmax(Z3), tf.argmax(Y))
    acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return acc


# In[8]:


def full_model(layers_n,learning_rate):
    L = len(layers_n)
    batch_size = 128
    num_steps = 1500
    display_step = 100
    X = tf.placeholder(dtype="float",shape=(layers_n[0],None))
    Y = tf.placeholder(dtype="float",shape=(layers_n[L-1],None))
    parameters = initialize_parameter(layers_n)
    Z_last = forward(X,parameters)
    cost = com_cost(Z_last,Y)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    acc = test_acc(Z_last,Y)
    
    
    init = tf.global_variables_initializer()
    sess = tf.Session() 
    sess.run(init)
    for step in range(1, num_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.T
        batch_y = batch_y.T
    
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            cost_batch , acc_batch = sess.run([cost,acc], feed_dict={X: batch_x,Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " +                    "{:.4f}".format(cost_batch) + ", Training Accuracy= " +                   "{:.3f}".format(acc_batch))
            
    print("Optimization Finished!")
        
    X_test = mnist.test.images.T
    Y_test = mnist.test.labels.T
    # Calculate accuracy for MNIST test images
    print("Testing Accuracy:",     sess.run(acc, feed_dict={X: X_test,Y: Y_test}))
    parameters = sess.run(parameters)
    
    return parameters


# In[9]:


layers_n = [784,256,256,10]
learning_rate = 0.001
parameters = full_model(layers_n,learning_rate)

